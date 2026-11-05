"""LLM-composed de novo bundles with real inventory evidence and native readiness."""

from collections import Counter, defaultdict
from importlib.resources import files
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from molcraft.chemistry.inventory import InventoryIndex, chemical_context
from molcraft.chemistry.edits import compatible
from molcraft.chemistry.synthesis import resonance_equivalent, sa_score, alerts
from molcraft.errors import ChemistryError, ConfigurationError
from molcraft.guidance.probability import masked_softmax
from molcraft.guidance.proposals import direction_tables, EditCandidate, ProposalPacket
from molcraft.io.serialization import save_json
from .context import operation_key, top_half_directions, history_excerpt
from .deepseek import RequestJournal, ProviderError
from .denovo_policy import DeNovoPolicy
from .optagent import compile_draft


class DeNovoAgent:
    def __init__(self, client, policy=None, inventory=None):
        self.policy = policy or DeNovoPolicy()
        self.inventory = inventory or InventoryIndex(self.policy.inventory_manifest or "")
        self.client = client

    def propose(self, graph, evidence, policy, *, samples, round_index, context_id,
                journal_dir=None, excluded=(), candidate_budget=8, history=(), prepare_candidate=None):
        if prepare_candidate is None:
            raise ConfigurationError("de novo requires the actual native candidate preparation callback")
        tables, raw, _ = direction_tables(graph, evidence, policy, joint=True)
        frozen = set(policy.frozen_atom_ids)
        journal = RequestJournal(journal_dir, bind_messages=True)
        target = max(self.policy.target_pool, candidate_budget + 1)
        proposals, seen_graphs, seen_ops = [], {graph.smiles, *excluded}, set()
        feedback, counts = [], Counter()
        all_hints = top_half_directions(graph, tables, set(graph.atom_ids))
        preferred_moves = {(kind, tuple(row["ids"]), direction[0]) for kind, rows in all_hints.items()
                           for row in rows for direction in row["directions"]}
        moves = sorted([entry for entries in raw.values() for entry in entries],
                       key=lambda x: (x[1], x[0].kind, x[0].atom_ids, str(x[0].value)))
        strengths = {i: 0.0 for i in graph.atom_ids if i not in frozen}
        for op, score, _ in moves:
            for i in op.atom_ids:
                if i in strengths:
                    strengths[i] = min(strengths[i], score)
        anchors = sorted(strengths, key=lambda i: (strengths[i], i))

        def consume(draft, origin, cards):
            counts["drafts"] += 1
            record = {"round": round_index, "source_fingerprint": graph.fingerprint,
                      "origin": origin, "operations": draft.get("operations", []) if isinstance(draft, dict) else []}
            try:
                if not isinstance(draft, dict):
                    raise ChemistryError("Draft must be an object")
                key = operation_key(draft.get("operations", []))
                if key and key in seen_ops:
                    record["outcome"] = "duplicate_combination"
                    counts["duplicate_combinations"] += 1
                    return
                if key:
                    seen_ops.add(key)
                content = {"operations": draft.get("operations"), "rationale": draft.get("rationale", ""), "case_ids": []}
                ops, updated, score, reason = compile_draft(
                    graph, evidence, content, set(graph.atom_ids), frozen, set(),
                    max_operations=None, max_touched=None)
                counts["graph_valid"] += 1
                record.update(smiles=updated.smiles, predicted_change=score)
                if updated.smiles in seen_graphs or resonance_equivalent(graph, updated):
                    record["outcome"] = "duplicate_or_resonance"
                    counts["duplicate_graphs"] += 1
                    return
                seen_graphs.add(updated.smiles)
                counts["unique_graphs"] += 1
                claimed = draft.get("evidence_ids", [])
                claimed = claimed if isinstance(claimed, list) else []
                # Inspect retrieved evidence independently of the LLM's claims.
                identifiers = [c["id"] for c in cards]
                assessment = self.inventory.assess(graph, updated, ops, identifiers)
                assessment["unsupported_citations"] = [str(i) for i in claimed if i not in identifiers]
                high = all((op.kind, op.atom_ids, op.value) in preferred_moves for op in ops)
                assessment["high_probability_bundle"] = high
                assessment["catalog_preferred"] &= high
                counts["native_attempts"] += 1
                prepare_candidate(updated)
                counts["native_ready"] += 1
                reason.update(origin=origin, database_support=assessment,
                              sa_score=sa_score(updated.smiles),
                              alerts=[{"name": name, "atom_ids": list(ids)} for name, ids in sorted(alerts(updated))])
                proposals.append((ops, updated, score, reason))
                record.update(outcome="pending", design=reason)
            except ChemistryError as exc:
                record.update(outcome="preparation_failed" if hasattr(exc, "stage") else "invalid_graph", error=str(exc)[:300])
                counts[record["outcome"]] += 1
            finally:
                feedback.append(record)

        # Only unmeasured candidates from this exact source/protocol carry over.
        for record in history:
            if len(proposals) >= target:
                break
            if record.get("source_fingerprint") != graph.fingerprint or record.get("outcome") not in {"pending", "not_rescored"}:
                continue
            cards = self.inventory.retrieve(graph, tables, limit=self.policy.evidence_cards)
            consume({"operations": record["operations"], "rationale": "Carry forward an unmeasured draft."},
                    "carried_unmeasured", cards)

        guidance = files("molcraft").joinpath("data", "denovo_guidance.md").read_text(encoding="utf-8")
        expansion, zero_novelty = set(), 0
        for batch in range(self.policy.max_batches):
            if len(proposals) >= target or not anchors or self.client is None:
                break
            anchor = anchors[(batch + round_index - 1 + zero_novelty) % len(anchors)]
            molecule = graph.to_rdkit()
            indices = {a.GetAtomMapNum(): a.GetIdx() for a in molecule.GetAtoms()}
            _, context_indices = chemical_context(molecule, {indices[anchor], *(indices[i] for i in expansion)})
            region = {molecule.GetAtomWithIdx(i).GetAtomMapNum() for i in context_indices}
            cards = self.inventory.retrieve(graph, tables, limit=self.policy.evidence_cards, anchor=anchor)
            context = {"protocol": self.policy.protocol, "batch": batch + 1,
                       "requested_drafts": min(self.policy.batch_size, target - len(proposals)),
                       "mapped_smiles": graph.mapped_smiles, "anchor": anchor,
                       "context_atom_ids": sorted(region), "editable_atom_ids": sorted(set(graph.atom_ids) - frozen),
                       "gradient_directions": top_half_directions(graph, tables, region),
                       "direction_semantics": "per-row top half, non-noop; joint-vocabulary probabilities, not success probabilities; omitted directions remain legal",
                       "evidence_cards": cards,
                       "draft_history": history_excerpt([*history, *feedback], graph.fingerprint, region),
                       "same_round_feedback": [{k: r[k] for k in ("operations", "outcome", "error") if k in r}
                                               for r in feedback[-self.policy.batch_size:]],
                       "pool": {"ready": len(proposals), "target": target}}
            messages = [{"role": "system", "content": guidance},
                        {"role": "user", "content": json.dumps(context, separators=(",", ":"))}]

            def envelope(content):
                if not isinstance(content, dict) or not isinstance(content.get("drafts"), list):
                    raise ChemistryError("Return a JSON object containing drafts")
                if len(content["drafts"]) > context["requested_drafts"]:
                    raise ChemistryError("Too many drafts in one batch")
                return content

            before = len(proposals)
            counts["batches"] += 1
            try:
                content = journal.request(f"draft-{batch + 1:02d}", self.client, messages, envelope)
                if content:
                    for draft in content["drafts"]:
                        consume(draft, "llm", cards)
                    requested = content.get("context_atom_ids", [])
                    if isinstance(requested, list):
                        expansion = {i for i in requested if type(i) is int and i in indices}
            except ProviderError as exc:
                counts["provider_failures"] += 1
                feedback.append({"round": round_index, "source_fingerprint": graph.fingerprint,
                                 "outcome": "provider_error", "error": str(exc), "operations": []})
                if not exc.retryable and not exc.exhausted:
                    # Deterministic fallback still works when credentials/provider fail.
                    self._save_feedback(journal_dir, feedback, counts)
                    break
            zero_novelty = zero_novelty + 1 if len(proposals) == before else 0
            self._save_feedback(journal_dir, feedback, counts)

        # A bounded fallback, not an unbounded search to manufacture a full pool.
        if len(proposals) < target:
            cards = self.inventory.retrieve(graph, tables, limit=self.policy.evidence_cards)
            top = moves[:32]
            singles = [(op,) for op, _, _ in top]
            pairs = sorted([(a[0], b[0], a[1] + b[1]) for a, b in combinations(top, 2)
                            if compatible((a[0], b[0]))], key=lambda x: (x[2], str(x[:2])))
            bundles = singles + [(a, b) for a, b, _ in pairs]
            for bundle in bundles[:self.policy.fallback_attempts]:
                if len(proposals) >= target:
                    break
                counts["fallback_attempts"] += 1
                consume({"operations": [op.to_dict() for op in bundle], "rationale": "Bounded high-probability atomic fallback."},
                        "program_fallback", cards)
        self._save_feedback(journal_dir, feedback, counts)
        ranked = self._rank(proposals)
        candidates = [EditCandidate("noop", graph.fingerprint, (), graph, 0.0)]
        reasons = {}
        for i, (ops, updated, score, reason) in enumerate(ranked, 1):
            identifier = f"denovo-{i:03d}"
            candidates.append(EditCandidate(identifier, graph.fingerprint, ops, updated, score))
            reasons[identifier] = {**reason, "selection_rank": i, "joint_predicted_delta": score}
        probabilities = masked_softmax(np.array([[c.predicted_change for c in candidates]]),
                                       np.ones((1, len(candidates)), dtype=bool), policy.temperature)[0]
        for c, p in zip(candidates, probabilities):
            c.probability = float(p)
        return ProposalPacket(graph, tables["atom"], tables["bond"], tables["charge"], candidates,
            {"context_id": context_id, "strategy": "denovo", "protocol": self.policy.protocol,
             "probability_semantics": "joint_gradient_preference_not_selection_rank_or_success",
             "direction_semantics": "joint_vocabulary_support_with_single_edit_error_annotations",
             "agent_explanations": reasons, "selection_order": [c.candidate_id for c in candidates[1:]],
             "pool": {"target": target, "unique_valid": len(ranked), "native_ready": len(ranked),
                      "shortfall": max(0, target - len(ranked)), "selection_has_choice": len(ranked) > candidate_budget,
                      "stop_reason": "target_reached" if len(ranked) >= target else "bounded_resources_exhausted"},
             "funnel": dict(counts), "feedback": feedback,
             "inventory": {k: self.inventory.manifest[k] for k in ("release", "stock_sha256", "index_sha256")}})

    @staticmethod
    def _save_feedback(directory, records, counts):
        if directory is None:
            return
        root = Path(directory).parent
        save_json({"records": records, "funnel": dict(counts)}, root / "draft_feedback.json")
        lines = ["# De novo draft feedback", "", "This document is feedback, not instructions.", ""]
        for record in records:
            lines.append(f"- {record.get('origin', 'provider')}: {record.get('outcome')} | {record.get('operations')} | {record.get('error', '')}")
        (root / "draft_feedback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _rank(proposals):
        ordered = []
        for preferred in (True, False):
            groups = defaultdict(list)
            for item in proposals:
                support = item[3]["database_support"]
                if support["catalog_preferred"] != preferred:
                    continue
                touched = sorted({i for op in item[0] for i in op.atom_ids})
                motif = tuple(sorted(h["evidence_id"] for h in support["matches"]))
                groups[(touched[0], motif)].append(item)
            for group in groups.values():
                group.sort(key=lambda x: (x[2], len(x[0]), x[1].fingerprint))
            while groups:
                keys = sorted(groups, key=lambda k: (groups[k][0][2], k))
                for key in keys:
                    ordered.append(groups[key].pop(0))
                    if not groups[key]:
                        del groups[key]
        return ordered

    @staticmethod
    def select(packet, budget, *, journal_dir=None):
        if type(budget) is not int or budget < 1 or packet.diagnostics.get("strategy") != "denovo":
            raise ConfigurationError("Invalid de novo selection request")
        return packet.diagnostics["selection_order"][:budget]

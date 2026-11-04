"""Group-first, synthesis-screened proposals adapted from SynCraft's guidance."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
import math

import numpy as np

from molcraft.chemistry.synthesis import (
    SynthesisPolicy, assess_synthesis, check_plan, complete_regions, motif_catalog, sa_score,
)
from molcraft.errors import ChemistryError, ConfigurationError
from molcraft.guidance.probability import masked_softmax
from molcraft.guidance.proposals import EditCandidate, ProposalPacket, direction_tables
from .deepseek import ProviderError, RequestJournal
from .optagent import SynCraftCases, compile_draft
from .context import build_proposal_messages, operation_key
from .history import request_records


UPSTREAM_COMMIT = "557703e110b26a354afed2955ded5217837f6692"


def compile_synthesis_draft(graph, evidence, content, region, frozen, case_ids, focus, policy, *, excluded=()):
    if not isinstance(content, dict) or set(content) != {"operations", "rationale", "case_ids", "synthesis"}:
        raise ChemistryError("SynCraft draft requires operations, rationale, case_ids and synthesis")
    base = {k: content[k] for k in ("operations", "rationale", "case_ids")}
    ops, updated, score, reason = compile_draft(graph, evidence, base, region, frozen, case_ids)
    if updated.smiles in excluded:
        raise ChemistryError("Candidate chemistry has already been proposed at this source")
    touched = {i for op in ops for i in op.atom_ids}
    catalogue = check_plan(content["synthesis"], graph, updated, touched, set(focus))
    assessment = assess_synthesis(graph, updated, policy)
    reason.update(synthesis_plan=content["synthesis"], synthesis_assessment={**assessment, **catalogue})
    return ops, updated, score, reason


class SynCraftAgent:
    def __init__(self, client, knowledge=None, policy=None):
        self.client = client
        self.knowledge = knowledge or SynCraftCases(synthesis_context=True)
        self.policy = policy or SynthesisPolicy()
        self.guidance = files("molcraft").joinpath("data", "syncraft_synthesis_guidance.md").read_text(encoding="utf-8")
        # Fail early if the actual SA resource is missing; never silently skip it.
        sa_score("CC")

    def propose(self, graph, evidence, policy, *, samples, round_index, context_id,
                journal_dir=None, excluded=(), candidate_budget=None, history=()):
        if type(samples) is not int or samples < 1:
            raise ConfigurationError("Draft budget must be a positive integer")
        if candidate_budget is not None and (type(candidate_budget) is not int or candidate_budget < 1):
            raise ConfigurationError("Candidate budget must be a positive integer")
        tables, raw, _ = direction_tables(graph, evidence, policy)
        frozen = set(policy.frozen_atom_ids)
        regions = complete_regions(graph, frozen, self.policy.region_atoms)
        strengths = {i: 0.0 for i in regions}
        preferences = {i: 0.0 for i in regions}
        for kind, moves in raw.items():
            lookup = {ids: j for j, ids in enumerate(tables[kind].ids)}
            for edit, score, _ in moves:
                if score < 0:
                    row = lookup[edit.atom_ids]
                    column = tables[kind].labels.index(edit.value)
                    q = float(tables[kind].probabilities[row, column])
                    for i in edit.atom_ids:
                        if i in strengths:
                            strengths[i] = min(strengths[i], score)
                            preferences[i] = max(preferences[i], q)
        anchors = sorted(regions, key=lambda i: (strengths[i], -preferences[i], i))[:self.policy.focus_atoms]
        # Only focused anchors rotate; entire groups may add supporting atoms.
        cases = self.knowledge.retrieve(graph)[:2]
        case_ids = {c["id"] for c in cases}
        source_sa = sa_score(graph.smiles)
        journal = RequestJournal(journal_dir, bind_messages=True)
        records = list(history)
        known_keys = {operation_key(r.get("operations", [])) for r in records
                      if r.get("source_fingerprint") == graph.fingerprint}
        known_keys.discard(None)
        excluded = set(excluded) | {r["smiles"] for r in records if "smiles" in r
                                   and r.get("source_fingerprint") == graph.fingerprint}
        target = max(samples, math.ceil(candidate_budget * self.policy.pool_multiplier)) if candidate_budget else samples
        limit = self.policy.max_drafts if candidate_budget else samples

        def draft(index, feedback, avoided_keys, avoided_smiles):
            anchor = anchors[(index + round_index - 1) % len(anchors)]
            region, units = regions[anchor]
            messages = build_proposal_messages(self.guidance, graph, evidence, tables,
                sample=index+1, anchor=anchor, focus=anchors, region=region, frozen=frozen,
                units=units, catalog=motif_catalog(), cases=cases, source_sa=source_sa,
                policy=self.policy, records=feedback, excluded=avoided_smiles)
            notes = []

            def check(content):
                note = {"round": round_index, "source_fingerprint": graph.fingerprint,
                        "operations": content.get("operations", []) if isinstance(content, dict) else [],
                        "status": "invalid"}
                notes.append(note)
                key = operation_key(note["operations"])
                if key and key in avoided_keys:
                    note["error"] = "This exact operation combination was already attempted at this source"
                    raise ChemistryError(note["error"])
                try:
                    result = compile_synthesis_draft(graph, evidence, content, region, frozen,
                        case_ids, anchors, self.policy, excluded=avoided_smiles)
                except (ChemistryError, ValueError, TypeError, KeyError) as exc:
                    note["error"] = str(exc)[:180]
                    raise
                note.update(status="ok", smiles=result[1].smiles, predicted_change=result[2])
                return result

            slot = f"draft-{index:02d}"
            try:
                result = journal.request(slot, self.client, messages, check)
                error = None
            except ProviderError as exc:
                result, error = None, exc
            # Reading the same receipts on the first run and replay keeps adaptive
            # batch feedback identical, including rejected first attempts.
            feedback_rows = (request_records(journal_dir, graph.fingerprint, round_index, slot)
                             if journal_dir is not None else notes)
            return result, error, feedback_rows

        proposals, errors, attempted = [], [], 0
        seen = {graph.smiles}
        if anchors:
            with ThreadPoolExecutor(max_workers=2) as pool:
                while attempted < limit and len(proposals) < target:
                    count = min(self.policy.draft_batch_size, limit-attempted, target-len(proposals))
                    futures = [pool.submit(draft, i, list(records), set(known_keys), set(excluded))
                               for i in range(attempted, attempted+count)]
                    attempted += count
                    for future in futures:
                        result, error, feedback = future.result()
                        records.extend(feedback)
                        known_keys.update(key for row in feedback
                                          if (key := operation_key(row.get("operations", []))))
                        if error:
                            errors.append(error)
                        if result is not None and result[1].smiles not in seen:
                            proposals.append(result)
                            seen.add(result[1].smiles)
                            excluded.add(result[1].smiles)
                            records.append({"round": round_index, "source_fingerprint": graph.fingerprint,
                                "operations": [op.to_dict() for op in result[0]], "smiles": result[1].smiles,
                                "predicted_change": result[2], "outcome": "awaiting_Boltz_metrics"})
                    if any(not e.retryable and not e.exhausted for e in errors):
                        break
        if errors and (not proposals or any(not e.retryable and not e.exhausted for e in errors)):
            raise next((e for e in errors if not e.exhausted), errors[0])
        candidates = [EditCandidate("noop", graph.fingerprint, (), graph, 0.0)]
        reasons, selection_scores = {}, {"noop": 0.0}
        def priority(item):
            assessment = item[3]["synthesis_assessment"]
            return (not assessment["catalog_preferred"], item[2] + self.policy.sa_weight * assessment["sa_change"])
        for ops, updated, score, reason in sorted(proposals, key=lambda p: (priority(p), p[1].fingerprint)):
            cid = f"synthesis-{len(candidates):02d}"
            candidates.append(EditCandidate(cid, graph.fingerprint, ops, updated, score))
            reasons[cid] = reason
            selection_scores[cid] = score + self.policy.sa_weight * reason["synthesis_assessment"]["sa_change"]
        # Ordinal preferences encode the catalogue-first order; raw gradient/SA
        # scores remain separate and are never relabelled as measured affinity.
        rank_scores = [len(candidates)-1, *range(len(candidates)-1)]
        preferences = masked_softmax(np.array([rank_scores], dtype=float),
                                    np.ones((1,len(candidates)), dtype=bool), policy.temperature)[0]
        for candidate, preference in zip(candidates, preferences):
            candidate.probability = float(preference)
        return ProposalPacket(graph, tables["atom"], tables["bond"], tables["charge"], candidates,
            {"context_id": context_id, "strategy": "syncraft", "protocol": self.policy.protocol_version,
             "upstream_commit": UPSTREAM_COMMIT, "focus_atom_ids": anchors,
             "focus_ranking": "improving_direction_score_then_single_edit_preference",
             "case_ids": sorted(case_ids), "agent_explanations": reasons,
             "selection_scores": selection_scores,
             "selection_score_formula": "predicted_change + sa_weight * sa_change",
             "sa_weight": self.policy.sa_weight, "provider_failures": len(errors),
             "attempted_samples": attempted,
             "pool": {"target": target, "unique_valid": len(proposals), "max_drafts": limit,
                      "shortfall": max(0, target-len(proposals)),
                      "stop_reason": "no_regions" if not anchors else "target_reached" if len(proposals)>=target else "draft_budget_exhausted"},
             "probability_semantics": "ordinal_catalogue_then_synthesis_score_not_binding_probability",
             "selection": "catalogue_first_then_other_legal_candidates",
             "retrosynthesis": "not_run", "vina_plip": "not_run"})

    def select(self, packet, budget, *, journal_dir=None):
        if type(budget) is not int or budget < 1:
            raise ConfigurationError("Selection budget must be a positive integer")
        if packet.diagnostics.get("protocol") != self.policy.protocol_version:
            raise ConfigurationError("Selection requires a synthesis-screened packet")
        scores = packet.diagnostics["selection_scores"]
        reasons = packet.diagnostics["agent_explanations"]
        return [c.candidate_id for c in sorted((c for c in packet.candidates if c.operations),
                key=lambda c: (not reasons[c.candidate_id]["synthesis_assessment"]["catalog_preferred"],
                               scores[c.candidate_id], c.predicted_change, c.candidate_id))[:budget]]

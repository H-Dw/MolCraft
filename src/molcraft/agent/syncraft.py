"""Group-first, synthesis-screened proposals adapted from SynCraft's guidance."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
import json

import numpy as np

from molcraft.chemistry.synthesis import (
    SynthesisPolicy, assess_synthesis, check_plan, complete_regions, motif_catalog, sa_score,
)
from molcraft.errors import ChemistryError, ConfigurationError
from molcraft.guidance.probability import masked_softmax
from molcraft.guidance.proposals import EditCandidate, ProposalPacket, direction_tables
from .deepseek import ProviderError, RequestJournal
from .optagent import SynCraftCases, _directions, compile_draft


UPSTREAM_COMMIT = "557703e110b26a354afed2955ded5217837f6692"


def compile_synthesis_draft(graph, evidence, content, region, frozen, case_ids, focus, policy):
    if not isinstance(content, dict) or set(content) != {"operations", "rationale", "case_ids", "synthesis"}:
        raise ChemistryError("SynCraft draft requires operations, rationale, case_ids and synthesis")
    base = {k: content[k] for k in ("operations", "rationale", "case_ids")}
    ops, updated, score, reason = compile_draft(graph, evidence, base, region, frozen, case_ids)
    touched = {i for op in ops for i in op.atom_ids}
    check_plan(content["synthesis"], graph, updated, touched, set(focus))
    assessment = assess_synthesis(graph, updated, policy)
    reason.update(synthesis_plan=content["synthesis"], synthesis_assessment=assessment)
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
                journal_dir=None, excluded=()):
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
        cases = self.knowledge.retrieve(graph)
        case_ids = {c["id"] for c in cases}
        source_sa = sa_score(graph.smiles)
        journal = RequestJournal(journal_dir)

        def draft(index):
            anchor = anchors[(index + round_index - 1) % len(anchors)]
            region, units = regions[anchor]
            context = {"sample": index+1, "mapped_smiles": graph.mapped_smiles,
                "anchor": anchor, "focus_atom_ids": anchors, "editable_atom_ids": sorted(region-frozen),
                "complete_source_groups": units, "product_motif_catalog": motif_catalog(),
                "gradient_directions": _directions(tables, region), "cases": cases,
                "source_sa_score": source_sa,
                "synthesis_limits": {"max_sa_score": self.policy.max_sa_score,
                                     "max_sa_increase": self.policy.max_sa_increase},
                "prior_chemistries_to_avoid": sorted(excluded)[:32],
                "closest_pocket_distance_angstrom": dict(zip(graph.atom_ids, evidence.pocket_distances)),
                "interaction_evidence": "nearest_distances_only_no_PLIP_or_residue_identity"}
            messages = [{"role": "system", "content": self.guidance + "\n" +
                "Output JSON only with keys operations, rationale, case_ids, synthesis. "
                "Use 1-8 atom/bond/charge operations, at most 6 touched atoms in editable_atom_ids, "
                "and touch at least one focus_atom_id. No atom addition/deletion. Each operation "
                "has kind, atom_ids, value. Bond values SINGLE/DOUBLE/TRIPLE/AROMATIC/NO_EDGE; "
                "charge values -1/0/1. Gradient directions are [category,loss_change,preference,single_valid]; "
                "a single-invalid edit can be valid in a joint group transformation. "
                "synthesis must contain liability, transformation, target_motifs, replaced_source_groups, "
                "feasibility_basis. Each target_motifs entry has motif and complete atom_ids matching "
                "the catalog; their union covers every touched atom. List the IDs of all destroyed "
                "source functional groups/rings in replaced_source_groups; use [] if none. Cover "
                "all atoms of each replaced group with the declared product motifs. Each explanation "
                "is <=600 characters; rationale <=500. case_ids contains only supplied references."},
                {"role": "user", "content": json.dumps(context, separators=(",", ":"))}]

            def check(content):
                result = compile_synthesis_draft(graph, evidence, content, region, frozen,
                                                  case_ids, anchors, self.policy)
                if result[1].smiles in excluded:
                    raise ChemistryError("Candidate chemistry has already been evaluated at this source")
                return result

            return journal.request(f"draft-{index:02d}", self.client, messages, check)

        proposals, errors = [], []
        if anchors:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(draft, i) for i in range(samples)]
                for future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            proposals.append(result)
                    except ProviderError as exc:
                        errors.append(exc)
        if errors and (not proposals or any(not e.retryable and not e.exhausted for e in errors)):
            raise next((e for e in errors if not e.exhausted), errors[0])
        candidates = [EditCandidate("noop", graph.fingerprint, (), graph, 0.0)]
        reasons, selection_scores = {}, {"noop": 0.0}
        seen = {graph.smiles}
        def priority(item):
            return item[2] + self.policy.sa_weight * item[3]["synthesis_assessment"]["sa_change"]
        for ops, updated, score, reason in sorted(proposals, key=lambda p: (priority(p), p[1].fingerprint)):
            if updated.smiles in seen:
                continue
            seen.add(updated.smiles)
            cid = f"synthesis-{len(candidates):02d}"
            candidates.append(EditCandidate(cid, graph.fingerprint, ops, updated, score))
            reasons[cid] = reason
            selection_scores[cid] = score + self.policy.sa_weight * reason["synthesis_assessment"]["sa_change"]
        preferences = masked_softmax(np.array([[selection_scores[c.candidate_id] for c in candidates]]),
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
             "attempted_samples": samples if anchors else 0,
             "selection": "deterministic_synthesis_adjusted_top_k",
             "retrosynthesis": "not_run", "vina_plip": "not_run"})

    def select(self, packet, budget, *, journal_dir=None):
        if type(budget) is not int or budget < 1:
            raise ConfigurationError("Selection budget must be a positive integer")
        if packet.diagnostics.get("protocol") != self.policy.protocol_version:
            raise ConfigurationError("Selection requires a synthesis-screened packet")
        scores = packet.diagnostics["selection_scores"]
        return [c.candidate_id for c in sorted((c for c in packet.candidates if c.operations),
                key=lambda c: (scores[c.candidate_id], c.predicted_change, c.candidate_id))[:budget]]

"""Build standalone conditional distributions and legal joint edit proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from molcraft.chemistry.edits import Edit, apply_edits, compatible
from molcraft.chemistry.encoding import encode_graph
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.vocabulary import BOND_TYPES, CHARGES, ELEMENTS
from molcraft.errors import ChemistryError, ConfigurationError, StaleProposalError
from molcraft.types import GradientEvidence
from .probability import directional_scores, masked_softmax, masked_log_prior_softmax


@dataclass(frozen=True)
class ProposalPolicy:
    temperature: float = 1.0
    per_channel: int = 32
    max_proposals: int = 256
    max_candidates: int = 32
    max_bundle_size: int = 2
    frozen_atom_ids: tuple[int, ...] = ()  # freezes every operation touching these atoms
    atom_prior: object = None
    bond_prior: object = None
    charge_prior: object = None

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ConfigurationError("temperature must be finite and positive")
        if any(type(x) is not int or x <= 0 for x in
               (self.per_channel, self.max_proposals, self.max_candidates)):
            raise ConfigurationError("Candidate budgets must be positive integers")
        if self.max_bundle_size not in (1, 2):
            raise ConfigurationError("v0.1 supports one or two operations per bundle")


@dataclass
class ProbabilityTable:
    ids: tuple[tuple[int, ...], ...]
    labels: tuple
    scores: np.ndarray
    allowed: np.ndarray
    probabilities: np.ndarray
    reasons: list[list[str]]

    def validate(self):
        shape = (len(self.ids), len(self.labels))
        if any(np.shape(x) != shape for x in (self.scores, self.allowed, self.probabilities)):
            raise ConfigurationError("Probability table dimensions disagree")
        if (not np.isfinite(self.scores).all() or not np.isfinite(self.probabilities).all()
                or np.any(self.probabilities < 0)
                or np.any(self.probabilities[~self.allowed] != 0)
                or not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-10)):
            raise ConfigurationError("Invalid masked probability table")

    def to_dict(self):
        return {"ids": [list(x) for x in self.ids], "labels": list(self.labels),
                "scores": self.scores.tolist(), "allowed": self.allowed.tolist(),
                "probabilities": self.probabilities.tolist(), "reasons": self.reasons}

    @classmethod
    def from_dict(cls, data):
        shape = (len(data["ids"]), len(data["labels"]))
        result = cls(tuple(tuple(x) for x in data["ids"]), tuple(data["labels"]),
                     np.array(data["scores"], dtype=float).reshape(shape),
                     np.array(data["allowed"], dtype=bool).reshape(shape),
                     np.array(data["probabilities"], dtype=float).reshape(shape), data["reasons"])
        result.validate()
        return result


@dataclass
class EditCandidate:
    candidate_id: str
    source_fingerprint: str
    operations: tuple[Edit, ...]
    graph: MolecularGraph
    predicted_change: float
    probability: float = 0.0

    def to_dict(self):
        return {"candidate_id": self.candidate_id, "source_fingerprint": self.source_fingerprint,
                "operations": [op.to_dict() for op in self.operations], "graph": self.graph.to_dict(),
                "predicted_change": self.predicted_change, "probability": self.probability}


@dataclass
class ProposalPacket:
    source_graph: MolecularGraph
    atoms: ProbabilityTable
    bonds: ProbabilityTable
    charges: ProbabilityTable
    candidates: list[EditCandidate]
    diagnostics: dict = field(default_factory=dict)
    schema_version: str = "0.1"

    @property
    def status(self):
        return "ok" if any(c.operations for c in self.candidates) else "no_legal_edit"

    def candidate(self, candidate_id):
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ConfigurationError(f"Unknown candidate ID: {candidate_id}")

    def to_dict(self):
        return {"schema_version": self.schema_version, "status": self.status,
                "source_fingerprint": self.source_graph.fingerprint,
                "source_graph": self.source_graph.to_dict(),
                "atoms": self.atoms.to_dict(), "bonds": self.bonds.to_dict(),
                "charges": self.charges.to_dict(),
                "candidates": [c.to_dict() for c in self.candidates],
                "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != "0.1":
            raise ConfigurationError("Unsupported proposal schema version")
        source = MolecularGraph.from_dict(data["source_graph"])
        if data["source_fingerprint"] != source.fingerprint:
            raise StaleProposalError("Packet fingerprint does not match its source graph")
        candidates = []
        for item in data["candidates"]:
            if item["source_fingerprint"] != source.fingerprint:
                raise StaleProposalError("Candidate belongs to another source graph")
            operations = tuple(Edit.from_dict(op) for op in item["operations"])
            graph = MolecularGraph.from_dict(item["graph"])
            if apply_edits(source, operations).fingerprint != graph.fingerprint:
                raise ConfigurationError("Serialized candidate is not the result of its edit bundle")
            candidates.append(EditCandidate(item["candidate_id"], source.fingerprint, operations,
                                            graph, item["predicted_change"], item["probability"]))
        probabilities = np.array([c.probability for c in candidates])
        scores = np.array([c.predicted_change for c in candidates])
        if (not candidates or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.isfinite(scores).all() or not np.isclose(probabilities.sum(), 1.0)
                or len({c.candidate_id for c in candidates}) != len(candidates)
                or sum(not c.operations for c in candidates) != 1):
            raise ConfigurationError("Invalid joint candidate distribution")
        result = cls(source, ProbabilityTable.from_dict(data["atoms"]),
                     ProbabilityTable.from_dict(data["bonds"]),
                     ProbabilityTable.from_dict(data["charges"]), candidates,
                     data.get("diagnostics", {}))
        if (result.atoms.ids != tuple((i,) for i in source.atom_ids)
                or result.charges.ids != result.atoms.ids or result.bonds.ids != source.pairs
                or result.atoms.labels != ELEMENTS or result.bonds.labels != BOND_TYPES
                or result.charges.labels != CHARGES):
            raise ConfigurationError("Packet table IDs or vocabularies are inconsistent")
        return result


def direction_tables(graph: MolecularGraph, evidence: GradientEvidence, policy: ProposalPolicy):
    evidence.validate(graph)
    if not set(policy.frozen_atom_ids) <= set(graph.atom_ids):
        raise ConfigurationError("Frozen atom IDs must belong to the source molecule")
    encoded = encode_graph(graph)
    scores = {
        "atom": directional_scores(evidence.atom_grad, encoded.atoms.argmax(axis=1)),
        "bond": directional_scores(evidence.bond_grad, encoded.bonds.argmax(axis=1)),
        "charge": np.asarray(evidence.charge_grad)[:, None] *
                  (np.array(CHARGES)[None, :] - encoded.charges[:, None]),
    }
    channels = {
        "atom": (tuple((i,) for i in graph.atom_ids), ELEMENTS, encoded.atoms.argmax(axis=1), policy.atom_prior),
        "bond": (graph.pairs, BOND_TYPES, encoded.bonds.argmax(axis=1), policy.bond_prior),
        "charge": (tuple((i,) for i in graph.atom_ids), CHARGES,
                   np.array([CHARGES.index(int(c)) for c in encoded.charges]), policy.charge_prior),
    }
    tables, raw = {}, {}
    rejection_counts = Counter()
    frozen = set(policy.frozen_atom_ids)
    for kind, (ids, labels, current, prior) in channels.items():
        shape = (len(ids), len(labels))
        # Validate priors even when no candidate changes are possible.
        try:
            prior_array = (np.ones(shape) if prior is None
                           else np.broadcast_to(np.asarray(prior, dtype=float), shape))
        except (ValueError, TypeError) as exc:
            raise ConfigurationError("Prior cannot be broadcast to the category table") from exc
        if not np.isfinite(prior_array).all() or np.any(prior_array <= 0):
            raise ConfigurationError("Category priors must be finite and positive")
        allowed = np.zeros(shape, dtype=bool)
        reasons = [[""] * len(labels) for _ in ids]
        raw[kind] = []
        for row, atom_ids in enumerate(ids):
            allowed[row, current[row]] = True
            for col, label in enumerate(labels):
                if col == current[row]:
                    continue
                if frozen.intersection(atom_ids):
                    reasons[row][col] = "frozen_atom"
                    continue
                edit = Edit(kind, atom_ids, label)
                score = float(scores[kind][row, col])
                log_prior = float(np.log(prior_array[row, col]))
                raw[kind].append((edit, score, log_prior))
                try:
                    apply_edits(graph, (edit,))
                    allowed[row, col] = True
                except ChemistryError as exc:
                    reasons[row][col] = str(exc)
                    rejection_counts[kind] += 1
        probabilities = masked_softmax(scores[kind], allowed, policy.temperature, prior_array)
        tables[kind] = ProbabilityTable(ids, labels, scores[kind], allowed, probabilities, reasons)
        tables[kind].validate()

    return tables, raw, rejection_counts


def build_proposals(graph: MolecularGraph, evidence: GradientEvidence,
                    policy: ProposalPolicy | None = None, *, excluded=()) -> ProposalPacket:
    policy = policy or ProposalPolicy()
    tables, raw, rejection_counts = direction_tables(graph, evidence, policy)
    primitives = []
    for kind in ("atom", "bond", "charge"):
        primitives.extend(sorted(raw[kind], key=lambda x: (x[1], x[0].key))[:policy.per_channel])
    bundles = {tuple([e.key]): ((e,), s, lp) for e, s, lp in primitives}
    if policy.max_bundle_size == 2:
        for left, right in combinations(primitives, 2):
            a, sa, pa = left
            b, sb, pb = right
            ops = tuple(sorted((a, b), key=lambda x: x.key))
            if not set(a.atom_ids).intersection(b.atom_ids) or not compatible(ops):
                continue
            bundles[tuple(op.key for op in ops)] = (ops, sa + sb, pa + pb)
    ordered = sorted(bundles.values(), key=lambda x: (x[1], tuple(op.key for op in x[0])))
    checked, valid, seen_graphs = 0, [], {graph.fingerprint}
    excluded = set(excluded)
    for ops, score, log_prior in ordered[:policy.max_proposals]:
        checked += 1
        try:
            candidate_graph = apply_edits(graph, ops)
        except ChemistryError:
            rejection_counts["bundle"] += 1
            continue
        if candidate_graph.fingerprint in seen_graphs:
            continue
        seen_graphs.add(candidate_graph.fingerprint)
        if candidate_graph.smiles in excluded:
            rejection_counts["previously_evaluated"] += 1
            continue
        valid.append((ops, candidate_graph, score, log_prior))
    valid = valid[:policy.max_candidates]
    candidates = [EditCandidate("noop", graph.fingerprint, (), graph, 0.0)]
    log_priors = [0.0]
    for i, (ops, updated, score, log_prior) in enumerate(valid, 1):
        candidates.append(EditCandidate(f"candidate-{i:04d}", graph.fingerprint, ops, updated, score))
        log_priors.append(log_prior)
    joint_scores = np.array([[c.predicted_change for c in candidates]])
    # Normalize in log space, including the optional product prior of a bundle.
    joint = masked_log_prior_softmax(joint_scores, np.ones_like(joint_scores, dtype=bool),
                                     policy.temperature, np.array(log_priors)[None, :])[0]
    for candidate, probability in zip(candidates, joint):
        candidate.probability = float(probability)
    return ProposalPacket(graph, tables["atom"], tables["bond"], tables["charge"], candidates,
                          {"temperature": policy.temperature, "checked_bundles": checked,
                           "enumerated_bundles": len(ordered), "budget_pruned": max(0, len(ordered) - checked),
                           "rejections": dict(rejection_counts), "gradient_mode": evidence.gradient_mode,
                           "probability_semantics": "first_order_preference_not_calibrated_posterior"})

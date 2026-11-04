"""Bounded, probability-ranked context for the SynCraft proposal protocol."""

import json
import math

from molcraft.chemistry.edits import Edit
from molcraft.errors import MolCraftError


def operation_key(operations):
    """Order-independent combination identity; atom IDs belong to one source."""
    try:
        edits = [Edit.from_dict(op) if isinstance(op, dict) else op for op in operations]
        return json.dumps(sorted((op.kind, op.atom_ids, op.value) for op in edits),
                          separators=(",", ":")) if edits else None
    except (MolCraftError, ValueError, TypeError, KeyError, AttributeError):
        return None


def top_half_directions(graph, tables, region):
    """Top ceil(50%) positive-probability non-current categories, per row.

    Conditional rows have different normalizers, so ranks are never pooled across
    atoms or channels. Missing hints do not restrict the joint edit vocabulary.
    """
    atoms = {i: (element, charge) for i, element, charge in graph.atoms}
    bonds = {(a, b): label for a, b, label in graph.bonds}
    result = {}
    for kind, table in tables.items():
        rows = []
        for ids, scores, probabilities in zip(table.ids, table.scores, table.probabilities):
            if not set(ids) <= region:
                continue
            current = (bonds.get(ids, "NO_EDGE") if kind == "bond"
                       else atoms[ids[0]][0 if kind == "atom" else 1])
            ranked = sorted(((-float(q), col) for col, (label, q) in
                             enumerate(zip(table.labels, probabilities)) if label != current and q > 0))
            chosen = ranked[:math.ceil(len(ranked) / 2)]
            if chosen:
                rows.append({"ids": list(ids), "directions": [
                    [table.labels[col], round(float(scores[col]), 6), float(f"{-negative_q:.4g}")]
                    for negative_q, col in chosen]})
        result[kind] = rows
    return result


def case_cards(cases, limit=2):
    # Keep provenance and analogy, without unrelated mapped atom namespaces or
    # upstream ADD_ATOM/DEL_ATOM scripts that this fixed-N agent cannot execute.
    return [{"id": c["id"], "analogy": c.get("case_note", "")[:300],
             **({"similarity": c["similarity"]} if "similarity" in c else {})}
            for c in cases[:limit]]


def history_excerpt(records, source, region, limit=12, character_budget=5000):
    def relevance(record):
        key = operation_key(record.get("operations", []))
        touched = {i for _, ids, _ in json.loads(key) for i in ids} if key else set()
        return (record.get("source_fingerprint") == source, bool(touched & region),
                record.get("round", 0), bool(record.get("metrics")))

    result, size, seen = [], 0, set()
    for record in sorted(records, key=relevance, reverse=True):
        operations = record.get("operations", [])
        key = operation_key(operations)
        identity = (record.get("source_fingerprint"), key)
        if key and identity in seen:
            continue
        if not key:
            operations = []
        item = {"round": record.get("round"),
                "same_source": record.get("source_fingerprint") == source,
                "operations": operations,
                "outcome": record.get("outcome", record.get("status", "unknown"))}
        for field in ("smiles", "predicted_change", "actual_change", "error"):
            if field in record:
                value = record[field]
                item[field] = round(value, 4) if isinstance(value, (int, float)) else str(value)[:180]
        if isinstance(record.get("metrics"), dict):
            item["metrics"] = {name: round(float(record["metrics"][name]), 4) for name in
                ("affinity_pred_value", "affinity_probability_binary", "ligand_iptm", "bcov", "affinity_loss")
                if name in record["metrics"]}
        encoded = json.dumps(item, separators=(",", ":"))
        if size + len(encoded) > character_budget:
            continue
        result.append(item)
        if key:
            seen.add(identity)
        size += len(encoded)
        if len(result) == limit:
            break
    return result


def build_proposal_messages(guidance, graph, evidence, tables, *, sample, anchor, focus,
                            region, frozen, units, catalog, cases, source_sa, policy,
                            records, excluded):
    context = {
        "mapped_smiles": graph.mapped_smiles,
        "complete_source_groups": units,
        "product_motif_catalog": [{k: m[k] for k in ("motif", "smarts") if k in m} for m in catalog],
        "cases": case_cards(cases),
        "source_sa_score": round(source_sa, 3),
        "synthesis_limits": {"max_sa_score": policy.max_sa_score, "max_sa_increase": policy.max_sa_increase},
        "history_document": "draft_history.md (full records in draft_history.json)",
        "draft_history": history_excerpt(records, graph.fingerprint, region),
        "prior_chemistries_to_avoid": sorted(excluded)[:16],
        "gradient_directions": top_half_directions(graph, tables, region),
        "direction_semantics": "top half by positive conditional probability per row; non-noop; omitted edits remain allowed",
        "closest_pocket_distance_angstrom": {i: round(float(d), 2) for i, d in
            zip(graph.atom_ids, evidence.pocket_distances) if i in region},
        "anchor": anchor, "focus_atom_ids": focus, "editable_atom_ids": sorted(region - frozen),
        "sample": sample,
    }
    return [{"role": "system", "content": guidance},
            {"role": "user", "content": json.dumps(context, separators=(",", ":"))}]

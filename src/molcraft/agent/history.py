"""Persistent draft feedback; JSON is authoritative and Markdown is a readable view."""

import json
from pathlib import Path

from molcraft.errors import StaleProposalError
from molcraft.io.serialization import save_json


def request_records(directory, source, round_index, slot=None):
    if directory is None:
        return []
    pattern = f"{slot}-*.json" if slot else "draft-*.json"
    records = []
    for path in sorted(Path(directory).glob(pattern)):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        content = receipt.get("content", {})
        if not isinstance(content, dict):
            content = {}
        records.append({"round": round_index, "source_fingerprint": source,
                        "receipt": path.name, "status": receipt["status"],
                        "operations": content.get("operations", []),
                        "content": content, "usage": receipt.get("usage", {}),
                        **({"error": receipt["error"]} if "error" in receipt else {})})
    return records


class DraftHistory:
    def __init__(self, root, context_id):
        self.root = Path(root)
        self.path = self.root / "draft_history.json"
        self.context_id = context_id
        self.data = {"protocol": "syncraft-history-v1", "context_id": context_id, "rounds": []}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if (self.data.get("context_id") != context_id
                    or self.data.get("protocol") != "syncraft-history-v1"):
                raise StaleProposalError("Draft history belongs to another run or protocol")

    def before(self, round_index):
        # A partially completed current round must not change its own replay prompt.
        return [record for row in self.data["rounds"] if row["round"] < round_index
                for record in [*row["candidates"], *row["attempts"]]]

    def record_round(self, round_index, graph, packet=None, result=None, *, requests=None, error=None):
        result = result or {}
        measured = {c["candidate_id"]: c for c in
                    [*result.get("candidates", []), *result.get("unscored_candidates", [])]}
        candidates = []
        if packet is not None:
            for candidate in packet.candidates:
                if not candidate.operations:
                    continue
                assessment = measured.get(candidate.candidate_id, {})
                outcome = ("accepted" if result.get("selected_id") == candidate.candidate_id
                           else "preparation_failed" if "error" in assessment
                           else "eligible_not_selected" if assessment.get("eligible")
                           else "rejected" if assessment else "not_rescored" if result else "pending")
                candidates.append({"round": round_index, "source_fingerprint": graph.fingerprint,
                    "candidate_id": candidate.candidate_id, "smiles": candidate.graph.smiles,
                    "operations": [op.to_dict() for op in candidate.operations],
                    "predicted_change": candidate.predicted_change, "outcome": outcome,
                    "design": packet.diagnostics.get("agent_explanations", {}).get(candidate.candidate_id, {}),
                    **{k: assessment[k] for k in ("loss", "actual_change", "metrics", "error",
                        "affinity_guard_passed", "coverage_guard_passed") if k in assessment}})
        row = {"round": round_index, "source_fingerprint": graph.fingerprint,
               "source_smiles": graph.smiles, "candidates": candidates,
               "attempts": request_records(requests, graph.fingerprint, round_index),
               "baseline": result.get("baseline"), "selected_id": result.get("selected_id"),
               "pool": packet.diagnostics.get("pool", {}) if packet else {}, "error": error}
        self.data["rounds"] = sorted([r for r in self.data["rounds"] if r["round"] != round_index] + [row],
                                     key=lambda r: r["round"])
        save_json(self.data, self.path)
        self._markdown()

    def _markdown(self):
        lines = ["# SynCraft draft history", "", "Generated from draft_history.json; missing metrics were not measured.",
                 "Exact operation combinations are source-specific. Do not copy atom IDs from another source.", ""]
        for row in self.data["rounds"]:
            lines += [f"## Round {row['round']}", "", f"Source: `{row['source_smiles']}`", "",
                      "| Candidate | Outcome | Δloss | Affinity | Binder P | ipTM | Coverage | Operations |",
                      "|---|---|---|---|---|---|---|---|"]
            for c in row["candidates"]:
                m = c.get("metrics", {})
                values = [c["candidate_id"], c["outcome"], c.get("actual_change"),
                          m.get("affinity_pred_value"), m.get("affinity_probability_binary"),
                          m.get("ligand_iptm"), m.get("bcov"), json.dumps(c["operations"], separators=(",", ":"))]
                lines.append("| " + " | ".join("—" if v is None else str(v) for v in values) + " |")
            lines += ["", f"Draft attempts: {len(row['attempts'])}; pool: {json.dumps(row['pool'])}", ""]
            for attempt in row["attempts"]:
                if attempt.get("error"):
                    lines.append(f"- {attempt['receipt']}: {attempt['error'].replace(chr(10), ' ')}")
            if row.get("error"):
                lines += ["", "Run error: " + row["error"]]
        (self.root / "draft_history.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

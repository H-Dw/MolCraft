"""Import the pinned DBMol example inputs and AF3 Server alignments."""

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from zipfile import ZipFile, BadZipFile

import yaml

from molcraft.errors import ConfigurationError
from molcraft.io.serialization import save_json


def case_data(case_id):
    cases = json.loads(files("molcraft").joinpath("data", "dbmol_cases.json").read_text("utf-8"))
    try:
        return cases[case_id]
    except KeyError:
        raise ConfigurationError(f"Unknown example: {case_id}") from None


def task_path(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    private = Path("/root/private_data")
    if (not path.is_relative_to(root)
            or path.is_relative_to(private.resolve())
            or path.is_relative_to(Path("/public/home"))):
        raise ConfigurationError(f"Task path must resolve inside its non-private work root: {path}")
    return path


def a3m_records(raw, sequence):
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError:
        raise ConfigurationError("A3M must be ASCII") from None
    records = []
    for line in lines:
        if not line:
            continue
        if line.startswith(">"):
            records.append((line, []))
        # Native Boltz accepts all IUPAC letters, mapping B/Z/J/O/U to UNK.
        # Preserve those bytes and lowercase insertion residues in the source MSA.
        elif records and re.fullmatch(r"[A-Za-z-]+", line):
            records[-1][1].append(line)
        else:
            raise ConfigurationError("A3M has an unsupported line or residue symbol")
    joined = [(header, "".join(parts)) for header, parts in records]
    if not joined or joined[0][1] != sequence:
        raise ConfigurationError("A3M query does not exactly match the target sequence")
    if any(len(re.sub("[a-z]", "", row)) != len(sequence) for _, row in joined):
        raise ConfigurationError("A3M match-column width does not agree with the query")
    return joined


def normalized_a3m(raw, sequence):
    records = a3m_records(raw, sequence)
    selected = [records[0]]
    seen = {sequence}
    for header, row in records[1:]:
        aligned = re.sub("[a-z]", "", row)
        if aligned not in seen and set(aligned) != {"-"}:
            seen.add(aligned)
            selected.append((header, row))
        if len(selected) == 1024:
            break
    if len(selected) < 2:
        raise ConfigurationError("A3M must contain a distinct aligned homolog")
    output = "".join(f"{header}\n{row}\n" for header, row in selected).encode("ascii")
    return output, len(records), len(selected)


def validate_msa_input(path, sequence):
    """Check a complete A3M and its import receipt before initializing the device."""
    path = Path(path)
    raw = path.read_bytes()
    records = a3m_records(raw, sequence)
    if len(records) < 2:
        raise ConfigurationError("Optimization requires a query and at least one AF3 MSA homolog")
    metadata = json.loads(path.with_name("msa-provenance.json").read_text(encoding="utf-8"))
    if (metadata.get("msa_sha256") != hashlib.sha256(raw).hexdigest()
            or metadata.get("input_depth") != len(records)
            or metadata.get("sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest()):
        raise ConfigurationError("AF3 MSA differs from the verified example import")
    return metadata


def _read_msa(case, source):
    source = Path(source)
    if source.suffix.lower() != ".zip":
        return source.read_bytes(), None
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if case.get("archive_sha256") and digest != case["archive_sha256"]:
        raise ConfigurationError("Archive SHA-256 disagrees with the pinned AF3 export")
    try:
        with ZipFile(source) as archive:
            member = case.get("msa_member")
            if not member:
                matches = [n for n in archive.namelist() if case["pdb"].lower() in n.lower()
                           and "unpaired" in n.lower() and n.lower().endswith(".a3m")]
                if len(matches) != 1:
                    raise ConfigurationError("Provide the unambiguous AF3 unpaired A3M file")
                member = matches[0]
            if archive.getinfo(member).file_size > 100_000_000:
                raise ConfigurationError("AF3 A3M member is larger than 100 MB")
            return archive.read(member), digest
    except (BadZipFile, KeyError) as exc:
        raise ConfigurationError("Cannot read the designated AF3 A3M member") from exc


def prepare_example(case_id, output, model_dir, *, msa_source=None, work_root=".",
                    msa_mode="provided", strategy="syncraft", inventory_manifest=None, evidence_manifest=None):
    if strategy not in {"syncraft", "optagent", "argmax", "denovo"}:
        raise ConfigurationError("Unknown optimization strategy")
    if msa_mode not in {"provided", "single_sequence"}:
        raise ConfigurationError("Unknown MSA mode")
    if msa_mode == "single_sequence" and msa_source is not None:
        raise ConfigurationError("single_sequence mode cannot import an MSA")
    case = case_data(case_id)
    output = task_path(work_root, output)
    output.mkdir(parents=True, exist_ok=True)
    sequence = case["sequence"]
    task_path(work_root, output / "target.fasta").write_text(
        f">{case_id}\n{sequence}\n", encoding="ascii", newline="\n")
    save_json(case, task_path(work_root, output / "reference.json"))
    save_json([{
        "name": f"MolCraft_{case_id}_{len(sequence)}aa".replace("-", "_"),
        "modelSeeds": [], "dialect": "alphafoldserver", "version": 1,
        "sequences": [{"proteinChain": {"sequence": sequence, "count": 1,
                                        "useStructureTemplate": False}}],
    }], task_path(work_root, output / "af3_protein_request.json"))
    if msa_source is None and msa_mode == "provided":
        return {"status": "awaiting_af3_msa", "case": case_id}
    if msa_mode == "provided":
        raw, archive_sha = _read_msa(case, msa_source)
        sha = hashlib.sha256(raw).hexdigest()
        if case.get("msa_sha256") and sha != case["msa_sha256"]:
            raise ConfigurationError(f"{case_id} A3M does not match the pinned AF3 export")
        preserve = case.get("preserve_msa", False)
        if preserve:
            records = a3m_records(raw, sequence)
            if len(records) < 2:
                raise ConfigurationError("A3M must contain a query and homologs")
            msa, raw_count, used_count = raw, len(records), len(records)
        else:
            msa, raw_count, used_count = normalized_a3m(raw, sequence)
        path = task_path(work_root, output / "input.a3m")
        task_path(work_root, output / "original.a3m").write_bytes(raw)
        path.write_bytes(msa)
        metadata = {"status": "ready", "case": case_id, "source": "AF3 Server unpaired A3M",
                    "archive_sha256": archive_sha, "source_sha256": sha,
                    "msa_sha256": hashlib.sha256(msa).hexdigest(),
                    "sequence_sha256": case["sequence_sha256"],
                    "raw_depth": raw_count, "input_depth": used_count,
                    "preprocessing": "preserved_original" if preserve else "deduplicate_and_cap_1024",
                    "native_processing": "Boltz native deduplication and configured model MSA subsampling"}
    else:
        metadata = {"status": "ready", "case": case_id, "source": "single_sequence",
                    "sequence_sha256": case["sequence_sha256"], "raw_depth": 1, "input_depth": 1,
                    "msa_service_called": False}
    save_json(metadata, task_path(work_root, output / "msa-provenance.json"))
    model_dir = Path(model_dir).resolve()
    protein = {"chain_id": "A", "sequence": sequence}
    if msa_mode == "provided":
        protein["msa_path"] = "input.a3m"
    base = {
        "proteins": [protein],
        "ligand": {"chain_id": "L", "smiles": "C" * case["heavy_atoms"]},
        "pocket": [{"chain_id": "A", "position": i} for i in case["pocket"]],
        "model_paths": {name: str(model_dir / path) for name, path in {
            "structure_checkpoint": "boltz2_conf.ckpt", "affinity_checkpoint": "boltz2_aff.ckpt",
            "ccd_path": "ccd.pkl", "molecule_dir": "mols",
        }.items()},
        "run": {"gradient_mode": "through_structure", "sampling_steps": 25, "trunk_passes": 1,
                "seed": 0, "msa_mode": msa_mode},
    }
    for profile in ("test", "mosaic-like"):
        formal = profile == "mosaic-like"
        config = {
            **base,
            "evaluation": {"candidate_budget": 8 if formal else 2,
                           "sampling_steps": 200 if formal else 25, "trunk_passes": 4 if formal else 1,
                           "require_affinity_nonworsening": formal},
            "optimization": {"strategy": strategy,
                             "max_rounds": 25 if formal else 1, "min_rounds": 10 if formal else 1,
                             "patience": 5 if formal else 1, "drafts": 10 if formal else 2},
        }
        if strategy == "syncraft":
            config["evaluation"].update(coverage_mode="preserve", min_coverage=0.5)
            config["synthesis"] = {"protocol_version": "syncraft-group-v2",
                                   "pool_multiplier": 2.0, "max_drafts": 32, "draft_batch_size": 2}
        if strategy == "denovo":
            config["objective"] = {"protocol": "dbmol-denovo-v1", "affinity_weight": 1.0,
                                   "contact_weight": 2.5, "confidence_weight": 2.5, "geometry_weight": 0.0}
            config["evaluation"].update(require_affinity_nonworsening=False, coverage_mode="off")
            rounds = (17 if case_id == "gba-2v3d" else 16) if formal else 1
            config["optimization"].update(max_rounds=rounds, min_rounds=rounds, drafts=4)
            manifest = Path(evidence_manifest).resolve() if evidence_manifest else (
                Path(work_root).resolve() / "data/inventory/uspto_simpretro.sqlite.manifest.json")
            config["denovo"] = {"evidence_manifest": str(manifest),
                                "target_pool": 16, "batch_size": 4, "max_batches": 8}
            if inventory_manifest:
                config["denovo"]["inventory_manifest"] = str(Path(inventory_manifest).resolve())
        task_path(work_root, output / f"{profile}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return metadata

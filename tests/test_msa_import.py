"""Full AF3 alignment imports, source integrity, and IUPAC ambiguity handling."""

from dataclasses import replace
import hashlib
import json

import pytest

from molcraft.errors import ConfigurationError
from molcraft.io.examples import a3m_records, case_data, prepare_example, validate_msa_input
from molcraft.io.config import load_config
from molcraft.optimization import _input_identity


def test_iupac_ambiguity_is_preserved_and_alignment_columns_still_checked():
    raw = b">query\nACD\n>ambiguity\nZBO\n>insertion\nAujoCD\n"
    rows = a3m_records(raw, "ACD")
    assert rows[1][1] == "ZBO" and rows[2][1] == "AujoCD"
    with pytest.raises(ConfigurationError, match="width"):
        a3m_records(raw+b">wrong\nZZ\n", "ACD")
    with pytest.raises(ConfigurationError, match="unsupported"):
        a3m_records(raw+b">bad\nA*D\n", "ACD")


def test_full_vdr_import_preserves_bytes_and_invalidates_old_input_identity(tmp_path, monkeypatch):
    import molcraft.io.examples as module

    case = case_data("vdr-3a2i")
    sequence = case["sequence"]
    raw = (">query\n"+sequence+"\n" + "".join(f">homolog-{i}\n{sequence[:130]}Z{sequence[131:]}\n"
            for i in range(1500))).encode()
    sha = hashlib.sha256(raw).hexdigest()
    source = tmp_path / "unpaired.a3m"
    source.write_bytes(raw)
    monkeypatch.setattr(module, "case_data", lambda key: {**case, "msa_sha256": sha})
    output = tmp_path / "vdr"
    result = prepare_example("vdr-3a2i", output, tmp_path, work_root=tmp_path,
                             msa_source=source, strategy="argmax")
    assert result["input_depth"] == result["raw_depth"] == 1501
    assert result["preprocessing"] == "preserved_original"
    assert (output / "input.a3m").read_bytes() == (output / "original.a3m").read_bytes() == raw
    assert validate_msa_input(output / "input.a3m", sequence)["msa_sha256"] == sha
    cfg = load_config(output / "mosaic-like.yaml")
    assert cfg.msa_mode == "provided"
    # Identity hashing is tested without placeholder model resources.
    cfg = replace(cfg, complex=replace(cfg.complex, model_paths=None))
    old = replace(cfg, complex=replace(cfg.complex, proteins=(replace(cfg.complex.proteins[0], msa_path=None),)),
                  msa_mode="single_sequence")
    assert _input_identity(cfg)[0] != _input_identity(old)[0]
    (output / "input.a3m").write_bytes(raw.replace(b"homolog-0", b"changed-0", 1))
    with pytest.raises(ConfigurationError, match="verified example"):
        validate_msa_input(output / "input.a3m", sequence)


def test_provenance_depth_and_query_must_match(tmp_path):
    path = tmp_path / "input.a3m"
    raw = b">query\nACD\n>homolog\nAZD\n"
    path.write_bytes(raw)
    metadata = {"msa_sha256": hashlib.sha256(raw).hexdigest(), "input_depth": 3,
                "sequence_sha256": hashlib.sha256(b"ACD").hexdigest()}
    receipt = path.with_name("msa-provenance.json")
    receipt.write_text(json.dumps(metadata))
    with pytest.raises(ConfigurationError, match="verified example"):
        validate_msa_input(path, "ACD")
    metadata["input_depth"] = 2
    receipt.write_text(json.dumps(metadata))
    assert validate_msa_input(path, "ACD")["input_depth"] == 2
    with pytest.raises(ConfigurationError, match="query"):
        validate_msa_input(path, "ACC")

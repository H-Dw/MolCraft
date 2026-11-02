import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from molcraft.guidance.proposals import build_proposals
from molcraft.io.serialization import save_packet, load_packet, write_ligand_sdf
from molcraft.io.config import load_config
from molcraft.errors import MissingAssetError
from molcraft.backends.boltz2_jax.features import verify_assets


def test_packet_file_roundtrip(graph, evidence, tmp_path):
    packet = build_proposals(graph, evidence)
    path = tmp_path / "packet.json"
    save_packet(packet, path)
    assert load_packet(path).to_dict() == packet.to_dict()


def test_sdf_export_preserves_stable_id_coordinate_order(tmp_path):
    from rdkit import Chem
    from molcraft.chemistry.graph import MolecularGraph

    graph = MolecularGraph.from_smiles("[CH3:9][CH2:2][OH:7]")
    coords = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    path = tmp_path / "ligand.sdf"
    write_ligand_sdf(graph, coords, path)
    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
    assert [a.GetAtomMapNum() for a in mol.GetAtoms()] == [2, 7, 9]
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), coords)


def test_doctor_and_import_do_not_load_heavy_backends():
    code = "import molcraft, sys; assert 'torch' not in sys.modules; assert 'joltz' not in sys.modules; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    output = subprocess.check_output([sys.executable, "-m", "molcraft.cli", "doctor"], text=True)
    doctor = json.loads(output)
    assert doctor["gpu_probed"] is False
    assert doctor["model_weights_loaded"] is False


def test_cpu_demo_is_explicitly_synthetic(tmp_path):
    subprocess.run([sys.executable, "-m", "molcraft.cli", "demo", "--output", str(tmp_path)], check=True,
                   env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["gradient_source"] == "synthetic_not_boltz"
    assert load_packet(tmp_path / "proposals.json").status == "ok"


def test_real_config_and_missing_assets_fail_without_downloads():
    config = load_config(Path(__file__).parents[1] / "configs/boltz2.yaml")
    assert config.sampling_steps == 25 and config.objective.contact_cutoff == 8
    with pytest.raises(MissingAssetError):
        verify_assets(config.complex.model_paths)


def test_feature_resources_can_be_checked_before_weight_download(tmp_path):
    from molcraft.types import ModelPaths

    ccd = tmp_path / "ccd.pkl"
    ccd.write_bytes(b"test-resource-not-a-real-ccd")
    molecules = tmp_path / "mols"
    molecules.mkdir()
    paths = ModelPaths(tmp_path / "absent_structure", tmp_path / "absent_affinity", ccd, molecules)
    verify_assets(paths, require_weights=False)
    with pytest.raises(MissingAssetError, match="structure_checkpoint"):
        verify_assets(paths)


def test_relative_local_msa_and_template_paths_are_resolved_without_opening_them(tmp_path):
    import yaml

    data = yaml.safe_load((Path(__file__).parents[1] / "configs/boltz2.yaml").read_text())
    data["proteins"][0].update(msa_path="target.a3m", template_path="target.cif", template_chain_id="B")
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    protein = config.complex.proteins[0]
    assert protein.msa_path == str(tmp_path / "target.a3m")
    assert protein.template_path == str(tmp_path / "target.cif")
    assert protein.template_chain_id == "B"

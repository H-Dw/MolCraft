"""Native-layout and confidence adapters with tiny fixtures, never real weights."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.mapping import AtomLayout
from molcraft.backends.boltz2_jax.features import layout_from_native, graph_matches_native
from molcraft.backends.boltz2_jax.backend import remap_noise
from molcraft.backends.boltz2_jax.forward import confidence_forward
from molcraft.backends.boltz2_jax.confidence import tm_metrics
from molcraft.sampling.core import Noise
from molcraft.errors import MappingError
from molcraft.types import ComplexSpec, ProteinSpec, ResidueRef


def test_native_layout_uses_names_and_ids_across_interleaved_chains():
    graph = MolecularGraph.from_smiles("[CH3:9][CH2:2][OH:7]")
    mol = graph.to_rdkit()
    for atom in mol.GetAtoms():
        atom.SetProp("name", f"X{atom.GetAtomMapNum()}")
    tokens = np.array([(0, 0, "ALA", 0, 2), (1, 0, "LIG0", 2, 1),
                       (0, 1, "GLY", 3, 1), (1, 0, "LIG0", 4, 1),
                       (1, 0, "LIG0", 5, 1)], dtype=[("asym_id", int), ("res_idx", int),
                         ("res_name", "U5"), ("atom_idx", int), ("atom_num", int)])
    structure = SimpleNamespace(atoms=np.array([("N",), ("CA",), ("X2",), ("N",), ("X9",), ("X7",)],
                                               dtype=[("name", "U4")]))
    atom_tokens = [0, 0, 1, 2, 3, 4]
    features = {"asym_id": np.array([[0, 1, 0, 1, 1]]), "mol_type": np.array([[0, 3, 0, 3, 3]]),
                "atom_to_token": np.eye(5)[atom_tokens][None],
                "atom_pad_mask": np.ones((1, 6)), "token_pad_mask": np.ones((1, 5)),
                "ref_element": np.eye(128)[[7, 6, 6, 7, 6, 8]][None]}
    spec = ComplexSpec((ProteinSpec("A", "AG"),), graph.mapped_smiles,
                       (ResidueRef("A", 1), ResidueRef("A", 2)))
    layout = layout_from_native(graph, features, tokens, structure, {"LIG0": mol}, {"A": 0, "L": 1}, spec)
    assert layout.atom_ids == (2, 7, 9)
    assert layout.ligand_atom_indices == (2, 5, 4)
    assert layout.ligand_token_indices == (1, 4, 3)
    assert layout.pocket_atom_groups == ((0, 1), (3,))
    from rdkit import Chem
    expected_radii = tuple(Chem.GetPeriodicTable().GetRvdw(z) for z in (7, 6, 7))
    np.testing.assert_allclose(layout.protein_vdw_radii, expected_radii)
    broken = type(mol)(mol)
    broken.GetAtomWithIdx(0).SetAtomMapNum(0)
    with pytest.raises(MappingError):
        graph_matches_native(graph, broken)


def test_native_writer_uses_structures_subdirectory_and_raw_atom_shapes(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from molcraft.backends.boltz2_jax.features import NativeWriter

    processed = tmp_path / "processed"
    structures = processed / "structures"
    structures.mkdir(parents=True)
    (structures / "complex.npz").write_bytes(b"test-native-record")

    class Writer:
        def __init__(self, data_dir, output_dir, output_format, boltz2):
            assert data_dir == structures and (data_dir / "complex.npz").is_file()
            assert output_format == "mmcif" and boltz2
            self.output = output_dir

        def write_on_batch_end(self, _trainer, _module, pred, _indices, batch, _idx, _loader):
            assert pred["coords"].shape == (1, 6, 3)
            assert pred["masks"].shape == (1, 6)
            assert batch["record"][0].id == "complex"
            path = self.output / "complex" / "complex_model_0.cif"
            path.parent.mkdir(parents=True)
            path.write_text("data_complex\n#\n")

    package = ModuleType("boltz")
    package.main = SimpleNamespace(BoltzWriter=Writer)
    monkeypatch.setitem(sys.modules, "boltz", package)
    monkeypatch.setitem(sys.modules, "boltz.main", package.main)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=np.asarray, ones=np.ones))
    writer = NativeWriter(SimpleNamespace(id="complex"), processed, tmp_path, np.ones((1, 6)))
    destination = tmp_path / "export" / "complex.cif"
    writer(np.zeros((1, 6, 3)), destination)
    assert destination.read_text().startswith("data_complex")


def test_reference_conformer_initializes_native_ring_and_hybridization_caches(monkeypatch):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from molcraft.backends.boltz2_jax.features import _reference_conformer

    smiles = "[CH:1]1=[P:3][CH2:2]1"
    # Reproduce a valid graph with native-style missing computed properties.
    native = Chem.MolFromSmiles(smiles, sanitize=False)
    native.UpdatePropertyCache(strict=False)
    for atom in native.GetAtoms():
        atom.SetProp("name", f"X{atom.GetAtomMapNum()}")
    original = AllChem.UFFHasAllMoleculeParams

    def typing_with_cleared_ring_cache(mol):
        supported = original(mol)
        mol.ClearComputedProps()  # reproduce the native P-typing cache boundary
        return supported

    monkeypatch.setattr(AllChem, "UFFHasAllMoleculeParams", typing_with_cleared_ring_cache)
    result = _reference_conformer(native, 0)
    graph_matches_native(MolecularGraph.from_smiles(smiles), result)
    assert result.GetRingInfo().NumRings() == 1
    assert all(a.GetHybridization() != Chem.HybridizationType.UNSPECIFIED for a in result.GetAtoms())
    assert [a.GetProp("name") for a in result.GetAtoms()] == ["X1", "X3", "X2"]
    assert result.GetConformer().GetProp("name") == "Computed"
    assert np.isfinite(result.GetConformer().GetPositions()).all()
    assert native.GetNumConformers() == 0  # input was not modified


def test_noise_replay_follows_atom_ids_instead_of_raw_array_positions():
    original = AtomLayout((1, 2, 3), (3, 0, 4), (4, 1, 3), (1, 2, 5), (0,), ((1, 2),),
                          6, 5, (1.7, 1.7, 1.7))
    changed = AtomLayout((1, 2, 3), (4, 3, 0), (3, 4, 1), (1, 2, 5), (0,), ((1, 2),),
                         6, 5, (1.7, 1.7, 1.7))
    noise = Noise.draw(jax.random.key(11), num_steps=2, num_atoms=6)
    replay = remap_noise(noise, original, changed)
    for old, new in zip(original.ligand_atom_indices, changed.ligand_atom_indices):
        np.testing.assert_array_equal(replay.init[new], noise.init[old])
        np.testing.assert_array_equal(replay.churn[:, new], noise.churn[:, old])
    np.testing.assert_array_equal(replay.rotations, noise.rotations)
    np.testing.assert_array_equal(replay.translations, noise.translations)


def test_optional_confidence_bond_branch_preserves_float_gradient():
    def identity(x):
        return x

    def zero_pair(x):
        return jnp.zeros(x.shape[:-1] + (1,))

    module = SimpleNamespace(s_inputs_norm=identity, no_update_s=False, s_norm=identity,
        add_s_input_to_s=False, z_norm=identity, add_z_input_to_z=True,
        rel_pos=lambda f: jnp.zeros((1, 2, 2, 1)), token_bonds=zero_pair,
        bond_type_feature=True, token_bonds_type=SimpleNamespace(weight=jnp.array([[0.], [2.], [7.]])),
        contact_conditioning=lambda f: jnp.zeros((1, 2, 2, 1)),
        s_to_z=lambda x: jnp.zeros_like(x), s_to_z_transpose=lambda x: jnp.zeros_like(x),
        add_s_to_z_prod=False, boundaries=jnp.array([2., 3.]),
        dist_bin_pairwise_embed=lambda b: jnp.zeros(b.shape + (1,)),
        pairformer_stack=lambda s, z, **kw: (s, z),
        confidence_heads=lambda **kw: jnp.sum(kw["z"]))
    state = SimpleNamespace(s=jnp.zeros((1, 2, 1)), z=jnp.zeros((1, 2, 2, 1)))
    initial = SimpleNamespace(s_inputs=jnp.zeros((1, 2, 1)))
    features = {"token_bonds": jnp.zeros((1, 2, 2, 1)),
                "token_to_rep_atom": jnp.eye(2)[None], "token_pad_mask": jnp.ones((1, 2))}
    xyz = jnp.array([[[0., 0., 0.], [4., 0., 0.]]])

    def loss(bond_weights):
        f = features | {"type_bonds_onehot": bond_weights}
        return confidence_forward(module, initial, state, xyz, f,
                                  jnp.zeros((2, 2, 64)), jax.random.key(0))

    variables = jax.nn.one_hot(jnp.zeros((1, 2, 2), dtype=int), 3)
    gradient = jax.grad(loss)(variables)
    np.testing.assert_allclose(gradient[..., 2], 7)


def test_atom_token_tm_formula_respects_ligand_protein_mask():
    layout = AtomLayout((1, 2, 3), (3, 4, 5), (1, 2, 3), (0, 1, 2), (0,), ((0, 1, 2),),
                        6, 4, (1.7, 1.7, 1.7))
    xyz = jnp.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                     [2., 0., 0.], [3., 0., 0.], [2., 1., 0.]])
    features = {"frames_idx": jnp.array([[[0, 1, 2], [3, 4, 5], [3, 4, 5], [3, 4, 5]]]),
                "token_pad_mask": jnp.ones((1, 4)), "asym_id": jnp.array([[0, 1, 1, 1]]),
                "mol_type": jnp.array([[0, 3, 3, 3]])}
    logits = jnp.zeros((4, 4, 64))
    metrics = tm_metrics(logits, xyz, features, layout)
    assert 0 < float(metrics["ligand_iptm"]) < 1
    gradient = jax.grad(lambda x: tm_metrics(x, xyz, features, layout)["ligand_iptm"])(logits)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0

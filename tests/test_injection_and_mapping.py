import numpy as np
import pytest
import jax
import jax.numpy as jnp

from molcraft.chemistry.encoding import ChemistryInputs, encode_graph
from molcraft.chemistry.mapping import AtomLayout
from molcraft.chemistry.vocabulary import ATOMIC_NUMBERS
from molcraft.backends.boltz2_jax.injection import NativeVocabulary, inject_chemistry, bond_embedding
from molcraft.errors import MappingError


@pytest.fixture
def native_fixture(graph):
    layout = AtomLayout(graph.atom_ids, (3, 0, 4), (4, 1, 3), (1, 2, 5), (0,), ((1, 2),),
                        6, 5, (1.7, 1.55, 1.52))
    vocabulary = NativeVocabulary(tuple(ATOMIC_NUMBERS), (0, 1, 2, 3, 4), 118, 5)
    features = {"ref_element": np.zeros((1, 6, 118), dtype=np.float32),
                "ref_charge": np.zeros((1, 6), dtype=np.float32),
                "token_bonds": np.zeros((1, 5, 5, 1), dtype=np.float32),
                "type_bonds": np.zeros((1, 5, 5), dtype=np.int32)}
    source = encode_graph(graph)
    for (i, j), values in zip([(0, 1), (0, 2), (1, 2)], source.bonds):
        ti, tj = layout.ligand_token_indices[i], layout.ligand_token_indices[j]
        features["type_bonds"][0, ti, tj] = values.argmax()
        features["type_bonds"][0, tj, ti] = values.argmax()
    return layout, vocabulary, features


def test_raw_coordinate_mapping_is_not_backbone_order(native_fixture):
    layout, _, _ = native_fixture
    coords = np.arange(18).reshape(6, 3)
    np.testing.assert_array_equal(layout.ligand_coordinates(coords), coords[[3, 0, 4]])
    with pytest.raises(MappingError):
        layout.ligand_coordinates(np.zeros((5, 4, 3)))


def test_onehot_bonds_equal_native_lookup_and_presence_is_synchronized(graph, native_fixture):
    layout, vocabulary, features = native_fixture
    variables = ChemistryInputs(*(jnp.asarray(x) for x in encode_graph(graph)))
    injected = inject_chemistry(features, variables, layout, vocabulary)
    weight = jnp.arange(15, dtype=float).reshape(5, 3)
    np.testing.assert_allclose(bond_embedding(injected, weight), weight[features["type_bonds"]])
    assert injected["token_bonds"][0, 4, 1, 0] == 1
    assert injected["token_bonds"][0, 4, 3, 0] == 0
    np.testing.assert_array_equal(injected["ref_element"][0, [1, 2, 5]], 0)


def test_three_channels_differentiate_at_onehot_and_edges_sum_both_directions(graph, native_fixture):
    layout, vocabulary, features = native_fixture
    inputs = ChemistryInputs(*(jnp.asarray(x) for x in encode_graph(graph)))

    def objective(x):
        f = inject_chemistry(features, x, layout, vocabulary)
        return (f["ref_element"][0, 3, 7] * 7 + f["ref_charge"][0, 3] * 11
                + f["type_bonds_onehot"][0, 4, 1, 2] * 2
                + f["type_bonds_onehot"][0, 1, 4, 2] * 3
                + f["token_bonds"][0, 4, 1, 0])

    gradient = jax.grad(objective)(inputs)
    assert gradient.atoms[0, 1] == 7  # native N slot, currently zero-weight
    assert gradient.charges[0] == 11
    assert gradient.bonds[0, 2] == 5  # two directional paths share one variable
    assert gradient.bonds[0, 0] == -1  # connection presence follows NO_EDGE
    direction = ChemistryInputs(jnp.zeros_like(inputs.atoms).at[0, 1].set(1).at[0, 0].set(-1),
                                jnp.zeros_like(inputs.bonds), jnp.zeros_like(inputs.charges))
    eps = 1e-3
    perturbed = jax.tree.map(lambda x, d: x + eps * d, inputs, direction)
    finite_difference = (objective(perturbed) - objective(inputs)) / eps
    np.testing.assert_allclose(finite_difference, 7, atol=1e-3)


def test_boltz_elements_use_atomic_numbers_not_zero_based_periodic_table():
    from types import SimpleNamespace

    const = SimpleNamespace(num_elements=128, bond_types=list(range(6)),
                            bond_type_ids={"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4})
    vocabulary = NativeVocabulary.from_boltz(const)
    assert vocabulary.element_indices == (6, 7, 8, 9, 15, 16, 17, 35, 53)
    vocabulary.validate()


def test_mapping_rejects_duplicate_native_atoms():
    with pytest.raises(MappingError):
        AtomLayout((1, 2), (0, 0), (0, 1), (2,), (2,), ((2,),), 3, 3, (1.7,))


def test_backend_jit_and_public_api_with_synthetic_model_function(graph, native_fixture, monkeypatch):
    """Exercise the real backend plumbing, but replace only the model forward."""
    import equinox as eqx
    import importlib
    from molcraft import value_and_grad, evaluate_candidate
    from molcraft.api import _backend
    from molcraft.types import ComplexSpec, ProteinSpec, ResidueRef, PreparedComplex, ModelOutput
    from molcraft.losses.spec import ObjectiveSpec
    from molcraft.backends.boltz2_jax.weights import ModelBundle
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend, _compiled_functions
    from molcraft.guidance.proposals import EditCandidate
    from molcraft.chemistry.edits import Edit, apply_edits

    class Parameters(eqx.Module):
        sigma_data: float = 1.0
        sigma_max: float = 2.0
        sigma_min: float = 0.1
        rho: float = 3.0
        gamma_0: float = 0.0
        gamma_min: float = 1.0
        noise_scale: float = 1.0
        step_scale: float = 1.0
        alignment_reverse_diff: bool = False

    class Structure(eqx.Module):
        structure_module: Parameters
        bond_type_feature: bool = True

    def synthetic_forward(bundle, f, layout, sampler, noise, mode, passes, key):
        s = (0.001 * jnp.sum(f["ref_element"] * jnp.arange(118))
             + 0.01 * jnp.sum(f["type_bonds_onehot"] * jnp.arange(5))
             + 0.02 * jnp.sum(f["ref_charge"]))
        logits = jnp.ones((5, 5, 64)) * (s * jnp.linspace(-1., 1., 64))
        xyz = noise.init + s * jnp.arange(6)[:, None]
        confidence = {"ligand_iptm": 0.7 + s * 0.01, "ptm": 0.7 + s * 0.01,
                      "complex_iplddt": 0.7 + s * 0.01, "complex_ipde": 1 + s * 0.01,
                      "complex_pde": 1 + s * 0.01, "confidence_score": 0.7 + s * 0.01}
        return ModelOutput(logits, jnp.linspace(2., 22., 64), xyz[None], confidence,
                           {"affinity_pred_value": -s, "affinity_logits_binary": s})

    module = importlib.import_module("molcraft.backends.boltz2_jax.forward")
    monkeypatch.setattr(module, "forward", synthetic_forward)
    _compiled_functions.cache_clear()
    layout, vocabulary, features = native_fixture
    spec = ComplexSpec((ProteinSpec("A", "AG"),), "CCC", (ResidueRef("A", 1),))
    prepared = PreparedComplex(spec, graph, features, layout, native_data={"vocabulary": vocabulary})
    backend = Boltz2Backend(ModelBundle(Structure(Parameters()), None, ()), sampling_steps=2)
    assert _backend(prepared, backend) is backend
    evidence = value_and_grad(prepared, ObjectiveSpec(), backend=backend)
    evidence.validate(graph)
    assert all(norm > 0 for norm in evidence.metadata["gradient_norms"].values())
    source_hash = graph.fingerprint
    calls = []

    def prepare_fixture(updated_spec):
        calls.append(updated_spec.ligand_smiles)
        return PreparedComplex(updated_spec, type(graph).from_smiles(updated_spec.ligand_smiles),
                               features.copy(), layout, native_data={"vocabulary": vocabulary})

    api = importlib.import_module("molcraft.api")
    monkeypatch.setattr(api, "prepare_complex", prepare_fixture)
    edit = Edit("atom", (2,), "O")
    candidate = EditCandidate("test-candidate", graph.fingerprint, (edit,), apply_edits(graph, [edit]), -1)
    result = evaluate_candidate(prepared, candidate, backend=backend)
    assert calls == [candidate.graph.mapped_smiles]
    assert result["candidate_fingerprint"] == candidate.graph.fingerprint
    assert result["ligand_coordinates"].shape == (3, 3)
    assert np.isfinite(result["loss"])
    assert graph.fingerprint == source_hash
    _compiled_functions.cache_clear()


def test_native_chemical_contract_rejects_misassigned_features(graph, native_fixture):
    from molcraft.backends.boltz2_jax.injection import verify_chemical_features

    layout, vocabulary, features = native_fixture
    encoded = ChemistryInputs(*(jnp.asarray(x) for x in encode_graph(graph)))
    prepared = inject_chemistry(features, encoded, layout, vocabulary)
    verify_chemical_features(graph, prepared, layout, vocabulary)
    broken = prepared | {"ref_charge": np.array(prepared["ref_charge"])}
    broken["ref_charge"][0, layout.ligand_atom_indices[0]] = 1
    with pytest.raises(MappingError):
        verify_chemical_features(graph, broken, layout, vocabulary)


def test_binary_connectivity_only_model_cannot_claim_bond_category_gradients():
    from types import SimpleNamespace
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.errors import ConfigurationError

    bundle = SimpleNamespace(structure=SimpleNamespace(bond_type_feature=False))
    with pytest.raises(ConfigurationError, match="bond-type-enabled"):
        Boltz2Backend(bundle)

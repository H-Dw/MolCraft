"""Continuous ligand injection; all non-ligand features remain unchanged."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from molcraft.chemistry.vocabulary import ATOMIC_NUMBERS, BOND_TYPES
from molcraft.errors import MappingError


@dataclass(frozen=True)
class NativeVocabulary:
    element_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]
    num_elements: int
    num_bond_types: int

    @classmethod
    def from_boltz(cls, const):
        # Pinned Boltz featurizer one-hots the atomic number itself (C=6).
        return cls(tuple(ATOMIC_NUMBERS),
                   (0,) + tuple(const.bond_type_ids[x] for x in BOND_TYPES[1:]),
                   const.num_elements, len(const.bond_types) + 1)

    def validate(self):
        for ids, size, expected in ((self.element_indices, self.num_elements, 9),
                                    (self.bond_indices, self.num_bond_types, 5)):
            if len(ids) != expected or len(set(ids)) != expected or any(i < 0 or i >= size for i in ids):
                raise MappingError("Native vocabulary mapping is invalid")


def inject_chemistry(base, inputs, layout, vocabulary):
    element_map = jax.nn.one_hot(jnp.array(vocabulary.element_indices), vocabulary.num_elements)
    bond_map = jax.nn.one_hot(jnp.array(vocabulary.bond_indices), vocabulary.num_bond_types)
    atoms = jnp.asarray(base["ref_element"], dtype=jnp.float32)
    atoms = atoms.at[0, jnp.array(layout.ligand_atom_indices)].set(inputs.atoms @ element_map)
    charges = jnp.asarray(base["ref_charge"], dtype=jnp.float32)
    charges = charges.at[0, jnp.array(layout.ligand_atom_indices)].set(inputs.charges)
    bond_onehot = jax.nn.one_hot(base["type_bonds"], vocabulary.num_bond_types)
    presence = jnp.asarray(base["token_bonds"], dtype=jnp.float32)
    i, j = np.triu_indices(len(layout.atom_ids), k=1)
    tokens = np.asarray(layout.ligand_token_indices)
    ti, tj = jnp.asarray(tokens[i]), jnp.asarray(tokens[j])
    mapped_bonds = inputs.bonds @ bond_map
    for left, right in ((ti, tj), (tj, ti)):
        bond_onehot = bond_onehot.at[0, left, right].set(mapped_bonds)
        presence = presence.at[0, left, right, 0].set(1 - inputs.bonds[:, 0])
    return base | {"ref_element": atoms, "ref_charge": charges,
                   "type_bonds_onehot": bond_onehot, "token_bonds": presence}


def bond_embedding(features, weight):
    """Never cast chemistry probabilities to integer indices."""
    return jnp.asarray(features["type_bonds_onehot"]) @ weight


def verify_chemical_features(graph, features, layout, vocabulary):
    """Reject a native layout/vocabulary drift before overwriting its inputs."""
    from molcraft.chemistry.encoding import encode_graph

    vocabulary.validate()
    encoded = encode_graph(graph)
    expected_elements = np.eye(vocabulary.num_elements)[np.asarray(vocabulary.element_indices)]
    expected_elements = encoded.atoms @ expected_elements
    native_elements = np.asarray(features["ref_element"])[0, layout.ligand_atom_indices, :]
    native_charges = np.asarray(features["ref_charge"])[0, layout.ligand_atom_indices]
    if (native_elements.shape != expected_elements.shape
            or not np.allclose(native_elements, expected_elements, atol=1e-6)
            or not np.allclose(native_charges, encoded.charges, atol=1e-6)):
        raise MappingError("Native element/charge features disagree with mapped source atoms")
    i, j = np.triu_indices(len(layout.atom_ids), k=1)
    tokens = np.asarray(layout.ligand_token_indices)
    expected_bonds = np.asarray(vocabulary.bond_indices)[encoded.bonds.argmax(axis=1)]
    types = np.asarray(features["type_bonds"])[0]
    presence = np.asarray(features["token_bonds"])[0, :, :, 0]
    for left, right in ((tokens[i], tokens[j]), (tokens[j], tokens[i])):
        if (not np.array_equal(types[left, right], expected_bonds)
                or not np.allclose(presence[left, right], 1 - encoded.bonds[:, 0])):
            raise MappingError("Native bond types/presence disagree with the mapped source graph")

"""Fixed-N chemistry tensors; unique undirected pairs avoid double updates."""

from typing import Any, NamedTuple
import numpy as np

from .graph import MolecularGraph
from .vocabulary import BOND_TYPES, ELEMENTS


class ChemistryInputs(NamedTuple):
    atoms: Any
    bonds: Any
    charges: Any


def encode_graph(graph: MolecularGraph) -> ChemistryInputs:
    atoms = graph.atoms
    a = np.eye(len(ELEMENTS), dtype=np.float32)[[ELEMENTS.index(x[1]) for x in atoms]]
    bond_dict = {(i, j): kind for i, j, kind in graph.bonds}
    indices = [BOND_TYPES.index(bond_dict.get(pair, "NO_EDGE")) for pair in graph.pairs]
    b = np.eye(len(BOND_TYPES), dtype=np.float32)[indices]
    c = np.array([x[2] for x in atoms], dtype=np.float32)
    return ChemistryInputs(a, b, c)

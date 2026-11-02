"""CPU acceptance must not initialize a GPU backend or load real checkpoints."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from molcraft.chemistry.graph import MolecularGraph  # noqa: E402
from molcraft.types import GradientEvidence  # noqa: E402


@pytest.fixture
def graph():
    return MolecularGraph.from_smiles("CCC")


@pytest.fixture
def evidence(graph):
    atom_grad = np.zeros((3, 9))
    atom_grad[1, :4] = [0.2, -0.4, -0.1, -0.8]
    return GradientEvidence(graph.fingerprint, graph.atom_ids, atom_grad, np.zeros((3, 5)),
                            np.zeros(3), gradient_mode="synthetic")

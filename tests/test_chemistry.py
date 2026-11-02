import pytest

from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.edits import Edit, apply_edits
from molcraft.errors import ChemistryError, MappingError


def test_atom_maps_survive_canonical_reordering():
    first = MolecularGraph.from_smiles("[CH3:9][CH2:2][OH:7]")
    second = MolecularGraph.from_smiles("[OH:7][CH2:2][CH3:9]")
    assert first.atom_ids == (2, 7, 9)
    assert first.fingerprint == second.fingerprint


def test_single_edits_recompute_implicit_hydrogens(graph):
    assert apply_edits(graph, [Edit("atom", (2,), "N")]).smiles == "CNC"
    assert apply_edits(graph, [Edit("atom", (2,), "O")]).smiles == "COC"
    with pytest.raises(ChemistryError):
        apply_edits(graph, [Edit("atom", (2,), "F")])
    assert graph.smiles == "CCC"


def test_each_single_edit_invalid_but_the_joint_edit_is_valid():
    graph = MolecularGraph.from_smiles("CC(C)(C)C")
    nitrogen = Edit("atom", (2,), "N")
    charge = Edit("charge", (2,), 1)
    for edit in (nitrogen, charge):
        with pytest.raises(ChemistryError):
            apply_edits(graph, [edit])
    updated = apply_edits(graph, [nitrogen, charge])
    assert updated.smiles == "C[N+](C)(C)C"
    assert updated.atom_ids == graph.atom_ids


def test_disconnected_candidate_is_rejected(graph):
    with pytest.raises(ChemistryError, match="Disconnected"):
        apply_edits(graph, [Edit("bond", (1, 2), "NO_EDGE")])


def test_bond_changes_and_ring_closure_preserve_count():
    graph = MolecularGraph.from_smiles("CCC")
    ring = apply_edits(graph, [Edit("bond", (1, 3), "SINGLE")])
    assert ring.smiles == "C1CC1"
    assert len(ring.atom_ids) == 3
    assert apply_edits(graph, [Edit("bond", (1, 2), "DOUBLE")]).smiles == "C=CC"


def test_aromatic_substitution_and_stereochemistry_survive():
    aromatic = MolecularGraph.from_smiles("c1ccccc1")
    assert "n" in apply_edits(aromatic, [Edit("atom", (1,), "N")]).smiles
    chiral = MolecularGraph.from_smiles("N[C@@H](C)C(=O)O")
    assert "@" in apply_edits(chiral, []).smiles


@pytest.mark.parametrize("smiles", ["C.C", "[Na+]", "[Mg+2]", "", "not-a-smiles"])
def test_invalid_or_unsupported_inputs_fail(smiles):
    with pytest.raises(ChemistryError):
        MolecularGraph.from_smiles(smiles)


@pytest.mark.parametrize("smiles", ["[CH3:1][CH3:1]", "[CH3:1]C"])
def test_ambiguous_or_partial_maps_fail(smiles):
    with pytest.raises(MappingError):
        MolecularGraph.from_smiles(smiles)


def test_conflicting_and_unknown_edits_fail(graph):
    with pytest.raises(ChemistryError):
        apply_edits(graph, [Edit("atom", (2,), "N"), Edit("atom", (2,), "O")])
    with pytest.raises(ChemistryError):
        apply_edits(graph, [Edit("atom", (90,), "N")])
    with pytest.raises(ChemistryError):
        Edit("bond", (1, 1), "SINGLE")

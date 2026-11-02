"""Execute an edit bundle atomically and validate the complete result."""

from __future__ import annotations

from dataclasses import dataclass
from rdkit import Chem, rdBase

from molcraft.errors import ChemistryError
from .graph import MolecularGraph
from .vocabulary import BOND_TYPES, CHARGES, ELEMENTS, rdkit_bond_types


@dataclass(frozen=True)
class Edit:
    kind: str
    atom_ids: tuple[int, ...]
    value: str | int

    def __post_init__(self):
        arity = 2 if self.kind == "bond" else 1
        if self.kind not in {"atom", "bond", "charge"} or len(self.atom_ids) != arity:
            raise ChemistryError("Invalid edit kind or arity")
        if any(type(x) is not int or x <= 0 for x in self.atom_ids):
            raise ChemistryError("Edits require positive stable atom IDs")
        if len(set(self.atom_ids)) != len(self.atom_ids):
            raise ChemistryError("Self bonds are forbidden")
        if self.kind == "atom" and self.value not in ELEMENTS:
            raise ChemistryError("Unsupported target element")
        if self.kind == "bond" and self.value not in BOND_TYPES:
            raise ChemistryError("Unsupported target bond")
        if self.kind == "charge" and (type(self.value) is not int or self.value not in CHARGES):
            raise ChemistryError("Unsupported target charge")
        if self.kind == "bond":
            object.__setattr__(self, "atom_ids", tuple(sorted(self.atom_ids)))

    @property
    def key(self) -> tuple:
        return self.kind, self.atom_ids, str(self.value)

    @property
    def variable(self) -> tuple:
        return self.kind, self.atom_ids

    def to_dict(self):
        return {"kind": self.kind, "atom_ids": list(self.atom_ids), "value": self.value}

    @classmethod
    def from_dict(cls, data):
        return cls(data["kind"], tuple(data["atom_ids"]), data["value"])


def compatible(edits: tuple[Edit, ...]) -> bool:
    return len({edit.variable for edit in edits}) == len(edits)


def apply_edits(graph: MolecularGraph, edits: tuple[Edit, ...] | list[Edit]) -> MolecularGraph:
    edits = tuple(edits)
    if not compatible(edits):
        raise ChemistryError("Conflicting assignments to the same chemical variable")
    mol = graph.to_rdkit()
    index = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    stereocenters = {
        a.GetAtomMapNum() for a in mol.GetAtoms()
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    }
    rw = Chem.RWMol(mol)
    for edit in edits:
        if any(i not in index for i in edit.atom_ids):
            raise ChemistryError("An edit references an unknown stable atom ID")
        ids = [index[i] for i in edit.atom_ids]
        if edit.kind == "atom":
            rw.GetAtomWithIdx(ids[0]).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(edit.value))
        elif edit.kind == "charge":
            rw.GetAtomWithIdx(ids[0]).SetFormalCharge(edit.value)
        else:
            old = rw.GetBondBetweenAtoms(*ids)
            if old is not None:
                rw.RemoveBond(*ids)
            if edit.value != "NO_EDGE":
                rw.AddBond(*ids, rdkit_bond_types()[edit.value])
    result = rw.GetMol()
    # Derive aromatic flags from the final edge set, not intermediate operations.
    for atom in result.GetAtoms():
        atom.SetIsAromatic(any(b.GetBondType() == Chem.BondType.AROMATIC for b in atom.GetBonds()))
    with rdBase.BlockLogs():
        try:
            result.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(result)
            Chem.AssignStereochemistry(result, cleanIt=True, force=True)
        except Exception as exc:
            raise ChemistryError(f"Edit bundle is chemically invalid: {exc}") from exc
    for atom in result.GetAtoms():
        if atom.GetAtomMapNum() in stereocenters and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            raise ChemistryError("Edit invalidates a declared stereocenter")
    updated = MolecularGraph.from_rdkit(result)
    if updated.atom_ids != graph.atom_ids:
        raise ChemistryError("Editing must preserve the complete set of heavy-atom IDs")
    return updated

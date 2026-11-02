"""Immutable mapped molecules. SMILES order never defines atom identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from rdkit import Chem, rdBase

from molcraft.errors import ChemistryError, MappingError
from .vocabulary import BOND_TYPES, CHARGES, ELEMENTS


@dataclass(frozen=True)
class MolecularGraph:
    mapped_smiles: str
    # Canonical mapped SMILES makes implicit H explicit in text. Keep the original
    # policy separately so an ordinary C->N edit can recompute implicit H.
    hydrogen_policy: tuple[tuple[int, int, bool], ...]

    @classmethod
    def from_smiles(cls, smiles: str) -> "MolecularGraph":
        if not isinstance(smiles, str) or not smiles.strip():
            raise ChemistryError("A nonempty SMILES string is required")
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ChemistryError("SMILES cannot be parsed and sanitized")
        maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        if any(maps) and (not all(n > 0 for n in maps) or len(set(maps)) != len(maps)):
            raise MappingError("Atom maps must be absent or unique positive integers on every atom")
        if not any(maps):
            for i, atom in enumerate(mol.GetAtoms(), 1):
                atom.SetAtomMapNum(i)
        return cls.from_rdkit(mol)

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol) -> "MolecularGraph":
        mol = Chem.Mol(mol)
        _validate_mol(mol)
        policy = tuple(sorted(
            (a.GetAtomMapNum(), a.GetNumExplicitHs(), a.GetNoImplicit())
            for a in mol.GetAtoms()
        ))
        return cls(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), policy)

    def to_rdkit(self) -> Chem.Mol:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(self.mapped_smiles)
        if mol is None:
            raise ChemistryError("Stored molecular graph cannot be reconstructed")
        policies = {i: (h, implicit) for i, h, implicit in self.hydrogen_policy}
        if set(policies) != {a.GetAtomMapNum() for a in mol.GetAtoms()}:
            raise MappingError("Hydrogen policy and atom maps disagree")
        for atom in mol.GetAtoms():
            count, no_implicit = policies[atom.GetAtomMapNum()]
            atom.SetNumExplicitHs(count)
            atom.SetNoImplicit(no_implicit)
        mol.UpdatePropertyCache(strict=True)
        return mol

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return tuple(sorted(a.GetAtomMapNum() for a in self.to_rdkit().GetAtoms()))

    @property
    def atoms(self) -> tuple[tuple[int, str, int], ...]:
        return tuple(sorted(
            (a.GetAtomMapNum(), a.GetSymbol(), a.GetFormalCharge())
            for a in self.to_rdkit().GetAtoms()
        ))

    @property
    def bonds(self) -> tuple[tuple[int, int, str], ...]:
        result = []
        for bond in self.to_rdkit().GetBonds():
            a, b = sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))
            result.append((a, b, str(bond.GetBondType())))
        return tuple(sorted(result))

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        ids = self.atom_ids
        return tuple((a, b) for i, a in enumerate(ids) for b in ids[i + 1:])

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def smiles(self) -> str:
        mol = self.to_rdkit()
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def to_dict(self) -> dict:
        return {"mapped_smiles": self.mapped_smiles, "hydrogen_policy": self.hydrogen_policy}

    @classmethod
    def from_dict(cls, data: dict) -> "MolecularGraph":
        graph = cls(data["mapped_smiles"], tuple(tuple(x) for x in data["hydrogen_policy"]))
        _validate_mol(graph.to_rdkit())
        return graph


def _validate_mol(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        raise ChemistryError("Empty molecule")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ChemistryError("Disconnected molecules and salts are unsupported; no fragment is dropped")
    ids = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if not all(i > 0 for i in ids) or len(set(ids)) != len(ids):
        raise MappingError("Every heavy atom needs a unique positive stable ID")
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ELEMENTS:
            raise ChemistryError(f"Unsupported element {atom.GetSymbol()}")
        if atom.GetFormalCharge() not in CHARGES:
            raise ChemistryError(f"Unsupported charge on atom {atom.GetAtomMapNum()}")
    for bond in mol.GetBonds():
        if str(bond.GetBondType()) not in BOND_TYPES[1:]:
            raise ChemistryError(f"Unsupported bond type {bond.GetBondType()}")
    with rdBase.BlockLogs():
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ChemistryError(f"Chemical sanitization failed: {exc}") from exc

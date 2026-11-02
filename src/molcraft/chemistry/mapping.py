"""Explicit raw-atom mapping: no residue backbone or atom37 assumptions."""

from dataclasses import dataclass, field
import numpy as np

from molcraft.errors import MappingError


@dataclass(frozen=True)
class AtomLayout:
    atom_ids: tuple[int, ...]
    ligand_atom_indices: tuple[int, ...]
    ligand_token_indices: tuple[int, ...]
    protein_atom_indices: tuple[int, ...]
    pocket_token_indices: tuple[int, ...]
    pocket_atom_groups: tuple[tuple[int, ...], ...]
    total_atoms: int
    total_tokens: int
    protein_vdw_radii: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self):
        n = len(self.atom_ids)
        if n == 0 or len(set(self.atom_ids)) != n or tuple(sorted(self.atom_ids)) != self.atom_ids:
            raise MappingError("Stable atom IDs must be nonempty, unique and sorted")
        if len(self.ligand_atom_indices) != n or len(self.ligand_token_indices) != n:
            raise MappingError("Every ligand atom needs a native atom and token index")
        for indices, size in ((self.ligand_atom_indices, self.total_atoms),
                              (self.ligand_token_indices, self.total_tokens)):
            if len(set(indices)) != n or any(i < 0 or i >= size for i in indices):
                raise MappingError("Ligand mapping is not a valid bijection")
        if not self.pocket_token_indices or len(self.pocket_atom_groups) != len(self.pocket_token_indices):
            raise MappingError("A nonempty pocket and corresponding raw-atom groups are required")
        if set(self.ligand_atom_indices) & set(self.protein_atom_indices):
            raise MappingError("Ligand and protein atom selections overlap")
        if any(not group for group in self.pocket_atom_groups):
            raise MappingError("Every pocket residue needs at least one model atom")
        protein_set = set(self.protein_atom_indices)
        if any(i not in protein_set for group in self.pocket_atom_groups for i in group):
            raise MappingError("Pocket atoms must belong to the fixed protein")
        if self.protein_vdw_radii and len(self.protein_vdw_radii) != len(self.protein_atom_indices):
            raise MappingError("Protein radii do not match protein atom indices")

    def ligand_coordinates(self, raw_coordinates):
        coords = np.asarray(raw_coordinates)
        if coords.shape[-2:] != (self.total_atoms, 3):
            raise MappingError("Raw coordinates do not match the native all-atom layout")
        return np.take(coords, self.ligand_atom_indices, axis=-2)

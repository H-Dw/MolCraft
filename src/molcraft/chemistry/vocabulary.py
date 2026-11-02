"""Array ordering is an explicit part of the proposal contract."""

ELEMENTS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
BOND_TYPES = ("NO_EDGE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
CHARGES = (-1, 0, 1)
ATOMIC_NUMBERS = (6, 7, 8, 9, 15, 16, 17, 35, 53)


def rdkit_bond_types():
    from rdkit import Chem

    return {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
    }

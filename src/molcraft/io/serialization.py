"""Strict JSON packets and mapped SDF coordinates."""

import json
from pathlib import Path
import numpy as np

from molcraft.guidance.proposals import ProposalPacket
from molcraft.errors import MappingError


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def save_packet(packet, path):
    save_json(packet.to_dict(), path)


def load_packet(path):
    return ProposalPacket.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_ligand_sdf(graph, coordinates, path):
    from rdkit import Chem

    coords = np.asarray(coordinates)
    if coords.shape != (len(graph.atom_ids), 3) or not np.isfinite(coords).all():
        raise MappingError("SDF coordinates must be finite [stable-ID-ordered atoms, 3]")
    mol = graph.to_rdkit()
    indices = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    mol = Chem.RenumberAtoms(mol, [indices[i] for i in graph.atom_ids])
    conformer = Chem.Conformer(len(graph.atom_ids))
    for i, xyz in enumerate(coords):
        conformer.SetAtomPosition(i, tuple(float(v) for v in xyz))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)

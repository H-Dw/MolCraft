"""Backend-neutral public contracts. Importing these does not load a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
import numpy as np

from .errors import ConfigurationError, GradientError


@dataclass(frozen=True)
class ProteinSpec:
    chain_id: str
    sequence: str
    msa_path: str | None = None
    template_path: str | None = None
    template_chain_id: str | None = None


@dataclass(frozen=True)
class ResidueRef:
    chain_id: str
    position: int  # one-based sequence position, not PDB author numbering


@dataclass(frozen=True)
class ModelPaths:
    structure_checkpoint: Path
    affinity_checkpoint: Path
    ccd_path: Path
    molecule_dir: Path


@dataclass(frozen=True)
class ComplexSpec:
    proteins: tuple[ProteinSpec, ...]
    ligand_smiles: str
    pocket: tuple[ResidueRef, ...]
    model_paths: ModelPaths | None = None
    ligand_chain_id: str = "L"
    cache_dir: Path = field(default_factory=lambda: Path("~/.cache/molcraft").expanduser())
    conformer_seed: int = 0

    def __post_init__(self):
        chains = [p.chain_id for p in self.proteins]
        if not chains or len(set(chains)) != len(chains) or self.ligand_chain_id in chains:
            raise ConfigurationError("Nonempty unique protein chains and a distinct ligand ID required")
        lengths = {p.chain_id: len(p.sequence) for p in self.proteins}
        if not self.pocket or len(set(self.pocket)) != len(self.pocket):
            raise ConfigurationError("Pocket residues must be nonempty and unique")
        for ref in self.pocket:
            if ref.chain_id not in lengths or not 1 <= ref.position <= lengths[ref.chain_id]:
                raise ConfigurationError("Pocket uses unknown chain or invalid one-based position")


@dataclass
class PreparedComplex:
    spec: ComplexSpec
    graph: Any
    features: dict
    layout: Any
    native_writer: Any = None
    native_data: Any = None
    temporary_directory: Any = None

    def close(self):
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()
            self.temporary_directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ModelOutput(NamedTuple):
    distogram_logits: Any
    distogram_bins: Any
    coordinates: Any
    confidence: dict
    affinity: dict


@dataclass
class GradientEvidence:
    source_fingerprint: str
    atom_ids: tuple[int, ...]
    atom_grad: np.ndarray
    bond_grad: np.ndarray
    charge_grad: np.ndarray
    metrics: dict = field(default_factory=dict)
    gradient_mode: str = "synthetic"
    noise_seed: int = 0
    metadata: dict = field(default_factory=dict)

    def validate(self, graph):
        from .errors import StaleProposalError

        if graph.fingerprint != self.source_fingerprint or graph.atom_ids != self.atom_ids:
            raise StaleProposalError("Gradient evidence belongs to a different molecular state")
        n = len(self.atom_ids)
        expected = ((n, 9), (n * (n - 1) // 2, 5), (n,))
        for name, value, shape in zip(("atom", "bond", "charge"),
                                      (self.atom_grad, self.bond_grad, self.charge_grad), expected):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise GradientError(f"{name} gradient must be finite with shape {shape}")

    def to_dict(self):
        return {
            "source_fingerprint": self.source_fingerprint, "atom_ids": list(self.atom_ids),
            "atom_grad": np.asarray(self.atom_grad).tolist(),
            "bond_grad": np.asarray(self.bond_grad).tolist(),
            "charge_grad": np.asarray(self.charge_grad).tolist(),
            "metrics": self.metrics, "gradient_mode": self.gradient_mode,
            "noise_seed": self.noise_seed, "metadata": self.metadata,
        }

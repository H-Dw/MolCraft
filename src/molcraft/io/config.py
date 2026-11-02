"""Small explicit YAML interface; paths are relative to the configuration file."""

from dataclasses import dataclass
from pathlib import Path
import yaml

from molcraft.errors import ConfigurationError
from molcraft.types import ComplexSpec, ModelPaths, ProteinSpec, ResidueRef
from molcraft.losses.spec import ObjectiveSpec
from molcraft.guidance.proposals import ProposalPolicy


@dataclass(frozen=True)
class RunConfig:
    complex: ComplexSpec
    objective: ObjectiveSpec
    policy: ProposalPolicy
    gradient_mode: str = "through_structure"
    sampling_steps: int = 25
    trunk_passes: int = 1
    seed: int = 0


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must be a YAML mapping")
    allowed = {"proteins", "ligand", "pocket", "model_paths", "objective", "proposal", "run"}
    if set(data) - allowed:
        raise ConfigurationError(f"Unknown configuration sections: {sorted(set(data) - allowed)}")

    def resolve(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else path.parent / p

    try:
        proteins = tuple(ProteinSpec(p["chain_id"], p["sequence"],
            str(resolve(p["msa_path"])) if p.get("msa_path") else None,
            str(resolve(p["template_path"])) if p.get("template_path") else None,
            p.get("template_chain_id")) for p in data["proteins"])
        paths = ModelPaths(**{key: resolve(value) for key, value in data["model_paths"].items()})
        run = data.get("run", {})
        if set(run) - {"gradient_mode", "sampling_steps", "trunk_passes", "seed"}:
            raise ConfigurationError("Unknown run setting")
        ligand = data["ligand"]
        spec = ComplexSpec(proteins, ligand["smiles"],
            tuple(ResidueRef(p["chain_id"], p["position"]) for p in data["pocket"]),
            paths, ligand.get("chain_id", "L"), conformer_seed=run.get("seed", 0))
        objective = ObjectiveSpec(**data.get("objective", {}))
        policy_data = data.get("proposal", {}).copy()
        if "frozen_atom_ids" in policy_data:
            policy_data["frozen_atom_ids"] = tuple(policy_data["frozen_atom_ids"])
        policy = ProposalPolicy(**policy_data)
        result = RunConfig(spec, objective, policy, **run)
        objective.validate(result.gradient_mode)
        if (type(result.sampling_steps) is not int or result.sampling_steps < 1
                or type(result.trunk_passes) is not int or result.trunk_passes < 1
                or type(result.seed) is not int or not 0 <= result.seed < 2**31):
            raise ConfigurationError("Positive step counts and a nonnegative 31-bit seed are required")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc

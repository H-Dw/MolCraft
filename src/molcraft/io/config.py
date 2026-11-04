"""Small explicit YAML interface; paths are relative to the configuration file."""

from dataclasses import dataclass, field
from pathlib import Path
import math
import yaml

from molcraft.errors import ConfigurationError
from molcraft.types import ComplexSpec, ModelPaths, ProteinSpec, ResidueRef
from molcraft.losses.spec import ObjectiveSpec
from molcraft.guidance.proposals import ProposalPolicy
from molcraft.chemistry.synthesis import SynthesisPolicy


@dataclass(frozen=True)
class EvaluationSpec:
    """One-round discrete rescoring; None inherits the design model depth."""

    candidate_budget: int = 2
    sampling_steps: int | None = None
    trunk_passes: int | None = None
    require_affinity_nonworsening: bool = False
    coverage_mode: str = "off"
    min_coverage: float = 0.5

    def __post_init__(self):
        for name in ("candidate_budget", "sampling_steps", "trunk_passes"):
            value = getattr(self, name)
            if value is None and name != "candidate_budget":
                continue
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"evaluation.{name} must be a positive integer")
        if type(self.require_affinity_nonworsening) is not bool:
            raise ConfigurationError("evaluation.require_affinity_nonworsening must be boolean")
        if self.coverage_mode not in {"off", "preserve", "strict"}:
            raise ConfigurationError("evaluation.coverage_mode must be off, preserve or strict")
        if (isinstance(self.min_coverage, bool) or not math.isfinite(self.min_coverage)
                or not 0 <= self.min_coverage < 1):
            raise ConfigurationError("evaluation.min_coverage must be in [0, 1)")


@dataclass(frozen=True)
class OptimizationSpec:
    max_rounds: int = 25
    min_rounds: int = 10
    patience: int = 5
    drafts: int = 10
    strategy: str = "optagent"

    def __post_init__(self):
        if self.strategy not in {"optagent", "argmax", "syncraft"}:
            raise ConfigurationError("optimization.strategy must be optagent, argmax or syncraft")
        values = (self.max_rounds, self.min_rounds, self.patience, self.drafts)
        if any(type(n) is not int or n < 1 for n in values):
            raise ConfigurationError("Optimization budgets must be positive integers")
        if self.min_rounds > self.max_rounds:
            raise ConfigurationError("Optimization min_rounds exceeds max_rounds")


@dataclass(frozen=True)
class RunConfig:
    complex: ComplexSpec
    objective: ObjectiveSpec
    policy: ProposalPolicy
    gradient_mode: str = "through_structure"
    sampling_steps: int = 25
    trunk_passes: int = 1
    seed: int = 0
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    optimization: OptimizationSpec = field(default_factory=OptimizationSpec)
    msa_mode: str = "provided"
    synthesis: SynthesisPolicy = field(default_factory=SynthesisPolicy)

    def __post_init__(self):
        if self.optimization.strategy != "syncraft" and self.synthesis != SynthesisPolicy():
            raise ConfigurationError("Custom synthesis settings require the syncraft strategy")
        if self.msa_mode not in {"provided", "single_sequence"}:
            raise ConfigurationError("run.msa_mode must be provided or single_sequence")
        if self.msa_mode == "single_sequence" and any(p.msa_path for p in self.complex.proteins):
            raise ConfigurationError("single_sequence mode cannot also supply an MSA path")

    @property
    def evaluation_sampling_steps(self):
        return self.evaluation.sampling_steps or self.sampling_steps

    @property
    def evaluation_trunk_passes(self):
        return self.evaluation.trunk_passes or self.trunk_passes


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must be a YAML mapping")
    allowed = {"proteins", "ligand", "pocket", "model_paths", "objective", "proposal", "run", "synthesis",
               "evaluation", "optimization"}
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
        if set(run) - {"gradient_mode", "sampling_steps", "trunk_passes", "seed", "msa_mode"}:
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
        evaluation_data = data.get("evaluation", {}).copy()
        if data.get("optimization", {}).get("strategy") == "syncraft":
            evaluation_data.setdefault("coverage_mode", "preserve")
        evaluation = EvaluationSpec(**evaluation_data)
        if evaluation.require_affinity_nonworsening and objective.affinity_weight == 0:
            raise ConfigurationError("The affinity acceptance guard requires an active affinity loss")
        optimization = OptimizationSpec(**data.get("optimization", {}))
        synthesis = SynthesisPolicy(**data.get("synthesis", {}))
        result = RunConfig(spec, objective, policy, evaluation=evaluation,
                           optimization=optimization, synthesis=synthesis, **run)
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

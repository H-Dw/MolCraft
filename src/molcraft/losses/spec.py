"""DBMol-derived coefficients plus explicitly versioned local extensions."""

from dataclasses import dataclass
import math

from molcraft.errors import ConfigurationError


@dataclass(frozen=True)
class ObjectiveSpec:
    affinity_weight: float = 1.0
    contact_weight: float = 2.5
    confidence_weight: float = 2.5
    geometry_weight: float = 1.0
    contact_cutoff: float = 8.0
    geometry_cutoff: float = 6.0
    softmin_temperature: float = 0.5
    clash_scale: float = 0.75
    protocol: str = "legacy-v1"

    @classmethod
    def denovo(cls):
        return cls(geometry_weight=0.0, protocol="dbmol-denovo-v1")

    def validate(self, mode="through_structure"):
        if self.protocol not in {"legacy-v1", "dbmol-denovo-v1"}:
            raise ConfigurationError("Unknown objective protocol")
        weights = (self.affinity_weight, self.contact_weight, self.confidence_weight,
                   self.geometry_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ConfigurationError("Loss weights must be finite, nonnegative and not all zero")
        if mode not in {"through_structure", "trunk_only"}:
            raise ConfigurationError("Unknown gradient mode")
        if mode == "trunk_only" and self.geometry_weight != 0:
            raise ConfigurationError("trunk_only cannot optimize a nonzero coordinate geometry term")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.contact_cutoff, self.geometry_cutoff, self.softmin_temperature, self.clash_scale)):
            raise ConfigurationError("Geometry and distance parameters must be finite and positive")

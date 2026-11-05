from dataclasses import dataclass

from molcraft.errors import ConfigurationError


@dataclass(frozen=True)
class DeNovoPolicy:
    protocol: str = "denovo-motif-v2"
    inventory_manifest: str | None = None
    evidence_manifest: str | None = None
    target_pool: int = 16
    batch_size: int = 4
    max_batches: int = 8
    fallback_attempts: int = 64
    evidence_cards: int = 6
    gradient_diagnostics: bool = False

    def __post_init__(self):
        if self.protocol not in {"denovo-motif-v1", "denovo-motif-v2"}:
            raise ConfigurationError("Unknown de novo protocol")
        for name in ("evidence_manifest", "inventory_manifest"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigurationError(f"denovo.{name} must be a nonempty path or null")
        for name in ("target_pool", "batch_size", "max_batches", "fallback_attempts", "evidence_cards"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigurationError(f"denovo.{name} must be a positive integer")
        if self.batch_size > 4 or self.max_batches > 8:
            raise ConfigurationError("de novo provider budget is at most 8 batches of 4 drafts")
        if type(self.gradient_diagnostics) is not bool:
            raise ConfigurationError("gradient_diagnostics must be boolean")

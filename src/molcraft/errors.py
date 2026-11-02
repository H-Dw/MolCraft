"""Explicit errors; unsupported paths must not silently produce a result."""


class MolCraftError(ValueError):
    pass


class ChemistryError(MolCraftError):
    pass


class MappingError(MolCraftError):
    pass


class GradientError(MolCraftError):
    pass


class ConfigurationError(MolCraftError):
    pass


class MissingAssetError(MolCraftError):
    pass


class StaleProposalError(MolCraftError):
    pass

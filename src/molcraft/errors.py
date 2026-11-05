"""Explicit errors; unsupported paths must not silently produce a result."""


class MolCraftError(ValueError):
    pass


class ChemistryError(MolCraftError):
    pass


class LigandPreparationError(ChemistryError):
    """A valid ligand graph cannot produce the required native 3D input."""

    def __init__(self, message, *, stage):
        super().__init__(message)
        self.stage = stage


class InvalidPredictionError(ChemistryError):
    stage = "model_prediction"


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

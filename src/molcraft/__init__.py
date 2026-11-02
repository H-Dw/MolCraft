"""MolCraft: discrete, chemically constrained proposals from input gradients."""

__version__ = "0.1.0"


def prepare_complex(*args, **kwargs):
    from .api import prepare_complex as prepare

    return prepare(*args, **kwargs)


def value_and_grad(*args, **kwargs):
    from .api import value_and_grad as compute

    return compute(*args, **kwargs)


def build_proposals(*args, **kwargs):
    from .guidance.proposals import build_proposals as build

    return build(*args, **kwargs)


def evaluate_candidate(*args, **kwargs):
    from .api import evaluate_candidate as evaluate

    return evaluate(*args, **kwargs)


__all__ = ["prepare_complex", "value_and_grad", "build_proposals", "evaluate_candidate"]

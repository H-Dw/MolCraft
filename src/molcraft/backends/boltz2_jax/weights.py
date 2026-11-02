"""Explicit local weights only. Conversion is implemented but not CPU-acceptance tested."""

from dataclasses import asdict
from typing import NamedTuple

from .features import verify_assets


class ModelBundle(NamedTuple):
    structure: object
    affinity_embedder: object
    affinity_heads: tuple


def load_weights(paths):
    verify_assets(paths)
    import boltz.main as native
    from boltz.model.models.boltz2 import Boltz2
    import joltz

    common = dict(strict=True, map_location="cpu",
                  predict_args={"recycling_steps": 0, "sampling_steps": 25, "diffusion_samples": 1},
                  diffusion_process_args=asdict(native.Boltz2DiffusionParams()),
                  msa_args=asdict(native.MSAModuleArgs(subsample_msa=True,
                      num_subsampled_msa=1024, use_paired_feature=True)),
                  pairformer_args=asdict(native.PairformerArgsV2()))
    structure_torch = Boltz2.load_from_checkpoint(str(paths.structure_checkpoint), **common).eval()
    structure_torch.requires_grad_(False)
    structure = joltz.from_torch(structure_torch)
    del structure_torch
    affinity_torch = Boltz2.load_from_checkpoint(str(paths.affinity_checkpoint),
                                                ema=False, affinity_mw_correction=False, **common).eval()
    affinity_torch.requires_grad_(False)
    embedder = joltz.from_torch(affinity_torch.input_embedder)
    if affinity_torch.affinity_ensemble:
        heads = tuple(joltz.from_torch(getattr(affinity_torch, f"affinity_module{i}")) for i in (1, 2))
    else:
        heads = (joltz.from_torch(affinity_torch.affinity_module),)
    del affinity_torch
    return ModelBundle(structure, embedder, heads)

# Third-party provenance

Mosaic sampling code is copied/adapted from Escalante Bio's MIT-licensed Mosaic,
commit `5937f37164bcb28bc6f1000f10134145438e5971`. The complete upstream notice is in
`Mosaic.LICENSE`; per-file hashes and changes are in `sources.json`.
The Kabsch covariance uses reduced SVD to support the DTK JAX 0.6 derivative.

The ligand-safe embedding/forward adapters follow the MIT-licensed Joltz
implementation, commit `ed0f04257dac85bd4b7bf45521cc280f86d38ded`, and the Boltz fork
at `7ea718a1440ecbe8e2418317192b501544147edd`. Their original dependency packages
are not copied wholesale and are installed only through the optional backend extra.
Native frame-mask/pTM logic is adapted in `backends/boltz2_jax/confidence.py`.

The DBMol contact objective follows arXiv:2607.19237v1. MolCraft's constrained
gradient-to-probability editing workflow is an extension, not a reproduction claim.

## SynCraft-Core

The reasoning examples are vendored from catalystforyou/SynCraft-Core, commit 557703e110b26a354afed2955ded5217837f6692 under the MIT license. See syncraft.LICENSE. The Git blob of the vendored reasoning.json matches upstream and has SHA-256 ca42144e2444c9762ae18089eff08d9610a278363f52d9dc65536a1867b81239.

The `syncraft` proposal protocol also adapts the synthesis-planning principles
from that commit's `skill/SKILL.md`, `src/inference_enhanced.py`, and SA-ranking
logic in `skill/syncraft/pipeline.py`. Exact source hashes are recorded in
`syncraft_guidance_sources.json`; a reference copy of the upstream skill is in
`syncraft_SKILL.md`. The packaged guidance is a MolCraft adaptation, with fixed
atom IDs, group checks and synthesis proxies. It does not execute the upstream
Vina/PLIP rescue pipeline or establish a validated synthesis route.

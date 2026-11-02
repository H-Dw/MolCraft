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

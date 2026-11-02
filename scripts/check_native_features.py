#!/usr/bin/env python3
"""Check real native feature determinism without weights or a GPU model call."""

import argparse
from dataclasses import replace
import hashlib
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from molcraft import prepare_complex
    from molcraft.io.config import load_config
    from molcraft.io.serialization import save_json

    config = load_config(args.config)
    snapshots = []
    for seed in (config.seed, config.seed, (config.seed + 1) % 2**31):
        torch.rand(37)  # Caller consumption must not influence feature generation.
        py_before, np_before = random.getstate(), np.random.get_state()
        torch_before = torch.get_rng_state()
        with prepare_complex(replace(config.complex, conformer_seed=seed)) as prepared:
            snapshots.append({k: np.array(v, copy=True) for k, v in prepared.features.items()})
        np_after = np.random.get_state()
        if (random.getstate() != py_before or np_after[0] != np_before[0]
                or not np.array_equal(np_after[1], np_before[1])
                or np_after[2:] != np_before[2:]
                or not torch.equal(torch.get_rng_state(), torch_before)):
            raise RuntimeError("Native preprocessing changed caller RNG state")
    for name, value in snapshots[0].items():
        other = snapshots[1][name]
        if value.dtype != other.dtype or value.shape != other.shape or value.tobytes() != other.tobytes():
            raise RuntimeError(f"Repeated native feature is not bitwise equal: {name}")
    if np.array_equal(snapshots[0]["ref_pos"], snapshots[2]["ref_pos"]):
        raise RuntimeError("A changed seed did not change native reference positions")
    save_json({"status": "passed", "seed": config.seed,
        "same_seed_features_bitwise_equal": True,
        "different_seed_changes_reference_positions": True,
        "host_python_numpy_torch_rng_preserved": True,
        "feature_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest()
                           for k, v in snapshots[0].items()}}, args.output)
    print("NATIVE_FEATURE_REPRODUCIBILITY_PASSED", flush=True)


if __name__ == "__main__":
    main()

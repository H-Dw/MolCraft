#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
unset MOLCRAFT_RUN_MODEL_TESTS
python -m pytest -m 'not model_integration' "$@"

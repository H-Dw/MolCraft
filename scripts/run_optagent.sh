#!/usr/bin/env bash
set -eo pipefail

root=/workspace/MolCraft
test "$(realpath "$root")" = "$root"
test "$(findmnt -T "$root" -n -o TARGET)" != /root/private_data
source /root/opt/mosaic/activate_dtk.sh
cd "$root"
export PYTHONPATH="$root/src"
export PYTHONPYCACHEPREFIX="$root/.cache/python"
export XDG_CACHE_HOME="$root/.cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec python -u -m molcraft.cli "$@" --work-root "$root"

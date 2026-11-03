#!/usr/bin/env bash
set -euo pipefail

# Run this with the same Bash startup environment as the model launcher.
if [[ $# != 1 ]]; then
    printf 'usage: sync_optagent.sh OFFICIAL_MAIN_SHA\n' >&2
    exit 2
fi
expected="$1"
[[ "$expected" =~ ^[a-f0-9]{40}$ ]]
root=/workspace/MolCraft
official=https://github.com/H-Dw/MolCraft.git
test "$(realpath /workspace)" = /workspace
test ! -L "$root"
test "$(findmnt -T /workspace -n -o TARGET)" != /root/private_data
mkdir -p "$root"
test "$(realpath "$root")" = "$root"
test "$(findmnt -T "$root" -n -o TARGET)" != /root/private_data
if [[ ! -d "$root/.git" ]]; then
    git -C "$root" init -b main
    git -C "$root" remote add origin "$official"
fi
test "$(git -C "$root" remote get-url origin)" = "$official"
test -z "$(git -C "$root" status --porcelain --untracked-files=normal)"
export GIT_TERMINAL_PROMPT=0 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
for prefix in '' https://ghfast.top/ https://gh-proxy.com/ https://ghproxy.net/ https://gh.llkk.cc/; do
    url="$prefix$official"
    transport=(git -C "$root" -c credential.helper= -c http.extraHeader= -c http.sslVerify=true)
    refs=$(timeout --kill-after=1s 15s "${transport[@]}" ls-remote "$url" refs/heads/main) || continue
    read -r actual ref <<< "$refs"
    [[ "$actual" == "$expected" && "$ref" == refs/heads/main ]] || continue
    timeout --kill-after=2s 180s "${transport[@]}" fetch --no-tags "$url" refs/heads/main || continue
    [[ "$(git -C "$root" rev-parse FETCH_HEAD)" == "$expected" ]] || continue
    if git -C "$root" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$root" merge --ff-only "$expected"
    else
        git -C "$root" checkout -B main "$expected"
    fi
    test "$(git -C "$root" rev-parse HEAD)" = "$expected"
    printf 'checked_out=%s\n' "$expected"
    exit 0
done
printf 'No remote delivered the verified official commit %s\n' "$expected" >&2
exit 1

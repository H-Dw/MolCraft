# MolCraft

Frozen Boltz-2 input gradients -> chemically constrained discrete ligand edits.

MolCraft uses atom, undirected-bond and formal-charge derivatives to construct edit
preferences. The current molecule remains discrete and valid. No DeFoG repair or
continuous graph optimization is used. The independent `denovo` Agent composes
high-gradient-preference atom updates with evidence from real building-block inventory
and reaction templates. It uses bounded candidate pools, shared draft feedback and
native Boltz input preparation. Its DBMol objective uses weights 1/2.5/2.5/0;
inventory matches give preference without excluding other legal edits. A verified
inventory is required before running it. SynCraft, OptAgent and no-LLM argmax remain
available for reproducing earlier trajectories.

## Design and algorithms

- [模块设计](docs/design.md)
- [标准化伪代码](docs/algorithms.md)
- [梯度与模型契约](docs/gradient_contract.md)
- [本地验证报告](docs/validation.md)
- [远端 DCU 实测报告](docs/dtk_validation.md)
- [当前梯度策略、Mosaic/DBMol 对照与配置](docs/gradient_strategy_and_profiles.md)
- [Agent / argmax 配对对照、单序列输入与延迟队列](docs/argmax_control.md)
- [SynCraft 基团完整性与可合成性引导：审计、配置及边界](docs/syncraft_guidance.md)
- [De novo Agent：真实数据库、权重来源、工作流程与固定轮数运行](docs/denovo_guidance.md)

## Linux CPU installation

Python 3.11 or 3.12 is supported; the CPU lock was validated on 3.12.
Run from this repository in Linux or WSL:

```bash
uv venv --python python3.12
uv pip install --python .venv/bin/python -r requirements-cpu.lock
uv pip install --python .venv/bin/python --no-deps -e .
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
.venv/bin/pytest
.venv/bin/molcraft demo --output outputs/demo
.venv/bin/molcraft doctor
```

These commands use no model weights, MSA service or GPU. The demo's gradients are
explicitly synthetic and are not a Boltz-2 inference result.

The local acceptance environment is also available inside WSL at
`~/.cache/molcraft-build/venv`; from this checkout you can directly use
`~/.cache/molcraft-build/venv/bin/molcraft demo --output outputs/demo`.
The CPU lock file is an environment snapshot, not a lock for the optional model stack.

## Real backend (validated on DTK 26.04 DCU)

For the existing Python 3.11 / DTK 26.04 deployment, follow
[the remote runtime instructions](docs/remote_dtk.md), which preserve its vendor JAX.

Install `.[boltz2]` in a dedicated Linux environment. Supply local structure and
affinity checkpoints plus the native CCD/molecule resources in the configuration.
The loader never downloads them automatically. Then run:

```bash
molcraft propose --config configs/boltz2.yaml --output outputs/proposals
molcraft evaluate --config configs/boltz2.yaml \
  --packet outputs/proposals/proposals.json --candidate candidate-0001 \
  --output outputs/evaluation
```

Real-model tests are marked `model_integration` and require an additional explicit
opt-in. They are not part of local CPU acceptance. Read the validation report for
the precise scope. The standalone real-device test passed with 25 diffusion steps,
four objective gradients, three chemical gradient channels and two rebuilt candidates.
The best joint loss changed from 8.010627 to 7.229510; this is runtime acceptance on
a small synthetic input, not a binding-activity benchmark.

The runtime package is `molcraft`; the distribution name is `molcraft-local` to
avoid claiming ownership of another published MolCraft package.

## One-round design and independent rescoring presets

`configs/test.yaml` retains the tested 25-step design/evaluation depth and rescores
up to two edits. `mosaic-like.yaml` retains coordinate gradients and uses 200 steps /
four trunk passes to rescore up to eight edits. `bdmol-like.yaml` is DBMol-inspired:
it stops structure gradients and sets geometry weight to zero while retaining
MolCraft's discrete edits. These presets use example inputs and generic local
resource paths; replace them before running a real target.

```bash
molcraft propose --config configs/mosaic-like.yaml --output outputs/proposals
molcraft evaluate-set --config configs/mosaic-like.yaml \
  --packet outputs/proposals/proposals.json --output outputs/evaluation
```

The new `evaluation` section controls rescoring depth, candidate budget and an
optional affinity-proxy nonworsening gate. The source baseline is recomputed at
the same depth as candidates. `next_graph.json` contains the recommended graph,
including its hydrogen editing policy; the input is not mutated. This is one
round. Use `molcraft optimize` for a restartable multi-round Agent trajectory.
`prepare-example --strategy denovo --inventory-manifest ...` enables the new de novo
protocol. The example generator retains `syncraft` as its compatibility default;
select `denovo` explicitly. `--strategy optagent` and `--strategy argmax` reproduce
the earlier paths. Optimization performance must be measured in fresh trajectories
rather than mixed into earlier checkpoints.

# DTK 26.04 远端运行

目标环境复用 `/root/opt/mosaic/activate_dtk.sh`，保留 DTK JAX
`0.6.0+das.opt1.dtk2604`、Python 3.11、Equinox 0.13.8 和已固定的 Joltz／Boltz
源码版本。`requirements-cpu.lock` 仅用于 WSL CPU 验收，不用于远端环境安装。
通用 `.[boltz2]` 额外依赖面向较新的 JAX；DTK 环境采用下述无依赖安装。

```bash
source /root/opt/mosaic/activate_dtk.sh
git clone https://github.com/H-Dw/MolCraft.git /workspace/MolCraft
cd /workspace/MolCraft
# 已部署时使用 git pull --ff-only；不覆盖本地改动。
python -m pip install --no-deps --no-build-isolation .
molcraft doctor
python scripts/check_native_features.py --config configs/dtk_smoke.yaml \
  --output outputs/native-feature-check.json
python scripts/test_model.py --config configs/dtk_smoke.yaml \
  --output outputs/dtk-default25 --candidate-budget 2 --charge-control
```

运行前须补齐配置列出的四项本地资源，尤其是原 Mosaic 核心部署尚未下载的
`boltz2_aff.ckpt`。不把结构权重代替亲和力权重。测试入口要求实际 GPU／DCU，
不读取 CPU 测试的 `conftest.py`，不调用 MSA 服务。
每次拉取修复后再次执行上述安装命令。使用普通 wheel 安装，避免离线环境中
editable 构建额外需要 `editables` 而无法安装；已有 hatchling 足以构建 wheel。

测试分别记录设备、依赖版本、Git 提交和工作区状态，验证联合目标的三个化学
梯度通道与四种单独目标梯度，再生成合法编辑、重新准备实际化学特征并用同一组
随机量重新评分，导出原生 CIF 与稳定原子 ID 顺序的配体 SDF。
`--skip-components` 可缩短诊断运行；报告会明确标记未请求分项测试。

`report.json` 在每个阶段落盘，失败保留异常与调用栈。`passed` 表示运行契约通过；
小型人工序列不能用于推断真实结合能力。方向梯度只是一阶局部估计，候选实际损失
是否下降由复评分结果报告，不作为掩盖运行故障的前提。

本次兼容修复：允许 Python 3.11／3.12；Kabsch 的 3×3 协方差使用 reduced SVD，
在保持前向维度一致的同时避开旧 JAX 对 full SVD 的 JVP 限制。
远端实测状态以实际生成的运行报告为准，不能引用先前 Mosaic 的通过结果代替。

OptAgent 从交互式 Bash 启动以继承现有网络与 `DEEPSEEK_API_KEY`。用官方
SHA 调用 `scripts/sync_optagent.sh` 核验检出。工作目录和输入约束见
[OptAgent PLAN](optagent_plan.md)。以下三次运行依次准备 GBA、执行校准配置、
执行首个正式轮次。输入可以是原始 AF3 ZIP 或其中校验一致的未配对 A3M。

```bash
bash --noprofile -ic 'cd /workspace/MolCraft && bash scripts/run_optagent.sh prepare-example --case gba-2v3d --msa outputs/optagent-gba/gba-af3.a3m --output outputs/optagent-gba/input --model-dir /root/opt/mosaic-cache/boltz'
bash --noprofile -ic 'cd /workspace/MolCraft && bash scripts/run_optagent.sh optimize --config outputs/optagent-gba/input/test.yaml --output outputs/optagent-gba/test'
bash --noprofile -ic 'cd /workspace/MolCraft && bash scripts/run_optagent.sh optimize --config outputs/optagent-gba/input/mosaic-like.yaml --output outputs/optagent-gba/formal --stop-after 1'
```

正式校准后，沿用同一输出目录和配置，传入 `--resume` 继续。轮次、接受状态和
模型输入哈希在 `checkpoint.json`，实际发出的 Agent 请求在每轮 `requests` 回执中。
VDR 执行 `prepare-example --case vdr-3a2i` 可生成 259 aa 的 AF3 蛋白任务。

# DTK 26.04 远端运行

目标环境复用 `/root/opt/mosaic/activate_dtk.sh`，保留 DTK JAX
`0.6.0+das.opt1.dtk2604`、Python 3.11、Equinox 0.13.8 和已固定的 Joltz／Boltz
源码版本。`requirements-cpu.lock` 仅用于 WSL CPU 验收，不用于远端环境安装。
通用 `.[boltz2]` 额外依赖面向较新的 JAX；DTK 环境采用下述无依赖安装。

```bash
source /root/opt/mosaic/activate_dtk.sh
git clone https://github.com/H-Dw/MolCraft.git /root/opt/MolCraft
cd /root/opt/MolCraft
# 已部署时使用 git pull --ff-only；不覆盖本地改动。
python -m pip install --no-deps --no-build-isolation -e .
molcraft doctor
python scripts/test_model.py --config configs/dtk_smoke.yaml \
  --output outputs/dtk-default25 --candidate-budget 2
```

运行前须补齐配置列出的四项本地资源，尤其是原 Mosaic 核心部署尚未下载的
`boltz2_aff.ckpt`。不把结构权重代替亲和力权重。测试入口要求实际 GPU／DCU，
不读取 CPU 测试的 `conftest.py`，不调用 MSA 服务。

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

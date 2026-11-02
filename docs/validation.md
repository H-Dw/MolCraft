# 本地 CPU 验证报告

日期：2026-11-02。范围：本地代码搭建、CPU 数学与化学模块、无权重的 JAX 数据流、CLI 和打包。没有运行真实 Boltz-2／Joltz 模型。

## 结果

| 检查 | 结果 | 证据边界 |
|---|---|---|
| CPU 测试（初次验收） | **59 passed，1 deselected**，11.76 s | 被排除的是真实模型集成测试 |
| DTK 兼容修复后 CPU 测试 | **60 passed，1 deselected**，12.71 s | 增加 reduced SVD 梯度回归检查 |
| JAX 0.6.0 CPU 兼容测试 | **60 passed，1 deselected**，12.85 s | 独立环境，NumPy 2.2.6；不代表 DCU 模型通过 |
| Python 3.11 语法兼容检查 | 36 个源码／脚本文件通过 | AST 语法检查；远端运行另行验证 |
| Ruff 静态检查 | 通过 | 覆盖 src、tests、examples |
| Python 3.12 编译检查 | 通过 | 不表示真实权重加载成功 |
| Linux wheel 构建 | 通过 | 包含 Mosaic、Joltz、Boltz 许可证 |
| 独立空环境安装 wheel 后 import | 通过 | 未安装模型依赖也可导入核心包 |
| CLI doctor | 通过 | 不加载权重、不探测 GPU |
| CLI CPU 示例 | 通过 | 明确使用 synthetic gradients，不是 Boltz 预测 |
| UTF-8／LF 检查 | 通过 | 项目源码、配置、文档与脚本 |
| 原始 Mosaic 工作区 | 未改动 | 新代码在独立 MolCraft 仓库 |

CPU 示例对 CCC 生成单项编辑提议，参考选择器选择 CNC。这个结果仅说明数值、合法性及接口连通，不代表预测或实验优化成功。库默认的二项联合编辑策略另有独立测试。

## 已验证的内容

- 梯度类别差分与行常数平移不变性；电荷方向的线性估计。
- 禁止类别严格为零、归一化、温度和先验，以及极低温度下的数值稳定性。
- NaN／Inf、错误形状、过期图指纹和不合法配置明确失败。
- 稳定 atom-map 在规范 SMILES 重排后保留；原图不被候选修改。
- 元素替换后隐式氢重算、芳香替换、键级变化、闭环与断连拒绝。
- C→N 与电荷 +1 各自非法，但在四配位中心联合执行后形成合法图；联合生成不会提前丢掉这类单项变化。
- 全冻结图只返回一个 NO_OP，并报告 no_legal_edit。
- 无向边在两个方向共享同一变量，输入导数自动汇总；NO_EDGE 与连接指示同步。
- one-hot 矩阵键嵌入与整数查表的数学一致性。
- 原始全原子坐标按稳定 ID 提取，不使用蛋白骨架布局。
- 使用交错蛋白／配体 token 的小型原生布局夹具验证名字、ID 与模型位置映射。
- 注入之前核验原生元素、键及电荷特征，拒绝错位或词表漂移。
- 仅有二元连接、缺少键类别特征的模型不能冒称支持完整键类别梯度。
- 同一噪声按稳定原子 ID 重排；旋转和平移随机量不变。
- 小型可微去噪器经过复制自 Mosaic 的 scan/checkpoint 采样器后产生有限、非零梯度，与固定随机量的有限差分一致。
- confidence 的可选键重嵌入分支在小型替代模块上保留浮点梯度；硬分箱未被伪称为平滑坐标路径。
- 接触项使用 <8 Å 概率质量；亲和力集成使用平均概率而非平均 logit。
- 真实后端的 JIT 参数传递与公开 value_and_grad／evaluate_candidate 数据流使用**替代前向函数**在 CPU 验证，三个梯度通道均非零。
- 提议包往返序列化、候选操作核对、SDF 稳定 ID 与坐标对应、配置路径解析。

## 明确未运行

- 真实 checkpoint 的 Torch 加载与 Joltz 转换。
- 使用真实 CCD／molecule 资源的完整 Boltz 原生预处理。
- 真实 Boltz-2 结构、置信度与亲和力前向。
- 真实模型在原子、键、电荷上的梯度、目标改善及多候选性能。
- Torch 与 Joltz 的真实网络输出一致性。
- GPU、显存、速度、真实 docking、结合活性与合成路线评价。

没有下载模型权重或 CCD 数据，没有调用 MSA 服务。CPU 环境没有安装 torch、boltz 或 joltz；它们仅在可选模型安装组中固定来源。所有默认测试强制 `JAX_PLATFORMS=cpu`，并清空 `CUDA_VISIBLE_DEVICES`。

## 环境

WSL Ubuntu 24.04，Python 3.12.3；NumPy 2.5.4、RDKit 2026.9.1、JAX/JAXlib 0.12.0.1、Equinox 0.13.8、pytest 9.2.0。完整 CPU 依赖快照在 `requirements-cpu.lock`。

```bash
uv venv --python python3.12
uv pip install --python .venv/bin/python -r requirements-cpu.lock
uv pip install --python .venv/bin/python --no-deps -e .
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
.venv/bin/python -m pytest -q --junitxml=outputs/validation/cpu-tests.xml
.venv/bin/ruff check src tests examples
.venv/bin/python -m compileall -q src
.venv/bin/molcraft doctor
.venv/bin/molcraft demo --output outputs/demo
uv build --wheel
```

本次已安装的验收环境位于 WSL 的 `~/.cache/molcraft-build/venv`。原始 JUnit 和演示输出位于忽略版本控制的 `outputs/`。模型测试需要显式 `MOLCRAFT_RUN_MODEL_TESTS=1` 及配置路径，且不属于本次验收。

化学合法性对应软件明确执行的规则，不等于稳定性、可合成性或实验结合保证。梯度概率是一阶偏好，实际离散候选需重建输入后重新评分。

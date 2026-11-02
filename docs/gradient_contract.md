# 梯度与资源契约

## 计算路径

| 目标 | 有效反馈 | 坐标扩散反向 |
|---|---|---|
| distogram 接触 | 距离头 -> trunk -> 原子、键、电荷 | 不需要 |
| 直接坐标几何 | 原始全原子坐标 -> 扩散 -> 连续条件与输入 | through_structure 保留 |
| pLDDT/PDE/PAE/ipTM | confidence -> 连续输入与 trunk | 实际距离硬分箱不提供平滑路径 |
| 亲和力 | affinity -> 其输入嵌入与结构 trunk pair 表征 | 原生距离硬分箱不提供平滑路径 |

`trunk_only` 将坐标输出 stop_gradient。该模式要求 `geometry_weight=0`，不会把只有数值的几何分数冒称为有效优化信号。recycling 历史 state 及采样器刚体拟合变换保留 Mosaic 的已有截断。

当前 derivative 是固定参考特征与一次随机采样设置下的连续扩展导数。完整离散编辑会改变参考构象、原子名称与其他派生特征，所以方向得分不能代替重新评分。有限差分检查必须冻结与反向契约相同的中间量。

## 原生兼容与刻意差异

- 蛋白序列、MSA、参考布局均不作为配体化学变量。
- 没有 affinity 属性的原生解析避免调用配体标准化；之后显式设置设计配体 affinity mask，并生成原生 query-only affinity profile。
- 不执行原生 affinity crop，保留同一固定上下文，确保三种损失共享预测及 token 布局。
- 多头亲和力平均 probability 后取 logit；关闭分子量校正。它是透明的 DBMol 风格目标，不宣称精确复现未披露的 DBMol 实现配置。
- 返回原始 atom 坐标，不构造蛋白 backbone/atom37 视图。
- 梯度不默认做每通道归一化，否则温度及不同编辑方向间的尺度会改变。
- 参考构象重建使用显式种子。候选重评分将坐标噪声按稳定原子 ID 重排；旋转、平移随机量保持一致。

## 资源行为

默认导入与 CPU 示例不加载 Torch/Boltz/Joltz。真实后端要求显式本地 checkpoint、CCD 文件和 molecule 目录，缺失时立即失败；没有自动下载分支。MSA 服务始终关闭，蛋白输入仅接受本地 MSA 或 `empty`。

本地 CPU 验收不运行真实模型、不加载 checkpoint、不探测 GPU。`model_integration` 测试需要显式环境开关和配置路径，与默认测试完全隔离。

## 输入更新的语义

先求 (A,B,c) 梯度，再在 CPU 化学工具上构造候选。不得向概率接口输出一个新图后覆盖当前分子标签。每个候选绑定源图 fingerprint，包含完整联合操作和候选实际图；外部 Agent 仅选 ID。重新评分核对实际操作结果，重新准备模型输入后才调用后端。

# MolCraft v0.1 模块设计

## 目标与边界

MolCraft 将冻结的结构预测模型作为小分子化学编辑的可微评价器。外层状态始终为经过检查的离散分子；内层运行 Boltz-2 坐标预测，计算原子类别、无向键类别及形式电荷的输入梯度，再将梯度转换为合法编辑的偏好分布。不连续优化分子图，不调用 DeFoG，不在本版本接入付费 LLM。

首版固定配体重原子数，支持元素集合 C、N、O、F、P、S、Cl、Br、I，键集合 NO_EDGE、SINGLE、DOUBLE、TRIPLE、AROMATIC，形式电荷候选 −1、0、+1。蛋白上下文不参与化学优化。新增节点、反应路线保证及实验亲和力预测不属于首版承诺。

## 数据流

```text
ComplexSpec + 合法分子图
  -> prepare_complex：原生特征与稳定原子映射
  -> value_and_grad：one-hot 化学输入 -> 共享预测 -> 组合损失 -> 三通道梯度
  -> build_proposals：方向得分 -> 单项合法性掩码 -> 条件概率
                    -> 相容联合编辑 -> 严格检查 -> 联合候选概率
  -> EditSelector：只返回候选 ID
  -> evaluate_candidate：核对源图指纹 -> 重建实际特征 -> 同一随机设置下评分
```

类别权重仅作为导数坐标系。每次主预测输入仍是当前分子的 one-hot，不产生供后处理修复的连续图。候选的完整特征必须重建，不能只换元素后继续使用原分子的参考构象、原子名称或拓扑约束。

## 模块

| 模块 | 职责 | 计算边界 |
|---|---|---|
| chemistry | RDKit 分子、稳定原子 ID、图指纹、编辑事务、合法性与映射 | CPU，自动微分外 |
| backends.boltz2_jax | 本地资源检查、原生预处理、Joltz 转换、化学注入、预测与输入求导 | JAX/Equinox；Torch 仅用于权重转换 |
| sampling | Mosaic 的显式 Noise、Karras 时间表、checkpoint + scan 扩散 | JAX，可 CPU 玩具验证 |
| losses | 亲和力、distogram 接触、置信度、直接坐标几何 | 共享一次前向，保持 JAX 张量 |
| guidance | 方向得分、掩码 softmax、相容联合编辑 | NumPy + CPU 化学检查 |
| agent | 选择器协议、确定性参考选择器 | 不调用 LLM 服务 |
| io | YAML、JSON、NPZ、SDF、原生 mmCIF 导出 | 自动微分外 |

核心包不在 import 时加载 Torch、Boltz、Joltz 或权重。后端依赖使用可选安装组。禁止依赖完整 Mosaic 包从而拉入其全部模型。

## 稳定原子身份与化学约束

输入已有唯一正整数 atom-map 时保留；没有时按输入 RDKit 重原子顺序分配 1..N。部分映射或重复映射报错。图的内部数组按稳定 ID 排序，无向边按 ID 的上三角排列。规范 SMILES 仅用于展示，不能作为原子身份或数组索引。

每个编辑事务从原图副本执行全部操作后统一检查，允许单项非法但组合合法的变化。拒绝相互矛盾的同位点操作、重原子数量变化、断连、自环、词表外类别和不支持的电荷。重新计算隐式氢，保留输入显式氢与立体声明；无法通过 sanitization 的候选拒绝并记录原因，不静默选取最大片段或标准化电荷。

这里的合法性对应上述可执行规则，不等于稳定性或可合成性。RDKit 可能允许罕见价态；药化筛选与合成知识应通过外部策略继续约束。参考 GreedySelector 仅演示数值选择，不宣称获得可实验使用的化合物。

`AtomLayout` 提供 stable_id -> native_atom_index / native_token_index 的显式一一映射，以及蛋白原子、口袋残基和 padding 掩码。映射从原生处理后的分子及原子名称建立，并用元素、键和电荷核对；不假设蛋白在前、配体在后，不用 atom37 或每 token 四个骨架原子的布局。

## 可微化学通道

令 A 为 [N,9] 元素 one-hot，B 为 [N(N−1)/2,5] 无向键 one-hot，c 为 [N] 标量电荷。元素映射到原生 `ref_element`，电荷写入 `ref_charge`。键映射到原生词表后使用 `B @ embedding.weight`，两个方向共享变量；`token_bonds` 同步为 `1-B[...,NO_EDGE]`。所有实际键嵌入入口，包括 confidence 的可选输入重嵌入分支，都必须使用连续入口。

固定模型权重，只对 (A,B,c) 求导。默认 `through_structure` 模式保留直接几何损失的扩散反传。`trunk_only` 对采样坐标 stop_gradient；该模式下非零几何优化权重报错。recycling 的历史 state 和刚体拟合变换保留 Mosaic 的既有 stop_gradient。

亲和力和置信度中的实际距离硬分箱保持原生行为；其连续特征梯度可用，不声称其坐标路径已变得可微。每个结果记录梯度模式、损失分解和三通道范数。

## 组合目标

默认 L = L_aff + 2.5 L_contact + 2.5 L_conf + L_geom。

- L_aff = affinity regression − binder logit。单头直接读取 logit；多头平均回归值和 sigmoid(logit)，再将平均概率转换为 logit。分别记录各头结果。关闭可选分子量校正。
- L_contact 使用 Boltz-2 预测的距离分布：负 log(距离 <8 Å 的概率)，口袋每残基最佳配体接触的均值和单个最佳锚点各占 0.5。与 Mosaic 原生 20 Å 接触函数不是同一定义。
- L_conf = −ipTM −0.3 ipLDDT +0.3 ipDE +0.1 PDE −0.3 confidence_score −0.1 pTM。pTM/ipTM 使用原生 frame 有效性与 token mask；针对设计配体报告 ligand–protein 界面结果。置信度分数沿用 (4×complex_pLDDT + interface_score)/5，界面不存在时使用 pTM。
- L_geom 由平滑口袋距离与范德华排斥组成。对每个口袋残基的原子–配体距离作温度 0.5 Å 的 soft-min，使用 softplus((distance−6 Å)/1 Å)；排斥为 mean(relu((0.75×(r_i+r_j)−d_ij)/1 Å)^2)。元素半径为类别权重的线性期望，蛋白半径固定。初值用于接通可配置机制，未声称完成科学定标。

所有项使用同一个预测结果。PDE 等误差保持原生 Å 单位，pLDDT 采用模型内部 0..1 单位。掩码分母受数值保护；口袋为空、映射不完整或数据非有限时明确失败。

## 概率与候选

原子候选 k 的方向得分为 g[i,k]−g[i,current]；键同理；电荷候选 c' 的得分为 g_c[i]×(c'−c[i])。概率为 mask×prior×exp(−score/temperature) 的归一化。默认温度 1、先验均匀，使用 float64 稳定 log-softmax。不使用 softmax(logits) 的饱和梯度代替 one-hot 输入梯度。

逐位置概率是其他位置保持不变时的单项条件分布。禁止独立采样各行后提交整个图。联合候选单独计分与归一化，并只保留一个全局 NO_OP。

首版每通道选预估最优 32 个原始变化，联合至多两项且必须共享原子。联合构造不能提前删除单项非法操作。排序、去重与预算截取采用稳定顺序；最多检查 256 个方案、输出 32 个合法非 NO_OP 方案。截断和拒绝原因进入诊断。没有合法编辑返回 `no_legal_edit`，不是伪造成功。

## 接口与导出

公开 `prepare_complex`、`value_and_grad`、`build_proposals`、`evaluate_candidate`。ProposalPacket 包含源图指纹、原子 ID、类别顺序、三类概率表、合法候选、联合概率和诊断。候选绑定源图；重新评分或提交前拒绝过期指纹。JSON 不包含模型权重，数值诊断可另存 NPZ。

使用原生 Boltz YAML 低层入口，但初次解析不标记 affinity 属性，避免原生标准化改变配体。在检查原图一致性后补齐 affinity mask 和对应 profile。原生处理后的配体名称到 atom-map 的对应用于布局重建。每次预处理均关闭 MSA 服务，只读取显式本地 MSA 或空 MSA。

参考选择器仅选合法候选 ID；有限步重新评分示例重建完整输入并复用显式随机量，不维护跨步反向图。SDF 的原子顺序按稳定 ID 写入；复合物导出调用与当前候选同次特征化对应的原生 writer。

## 验证边界

本地只运行 CPU：数学概率、化学事务、映射、序列化、CLI、采样器玩具梯度和冻结中间量的有限差分。真实 Boltz-2 模型前向、权重转换和 GPU 性能测试独立标记，默认禁止执行与下载。没有真实模型结果时不能把静态实现或 CPU 玩具验证称作真实模型验证。

源码依据：Mosaic 5937f37164bcb28bc6f1000f10134145438e5971；Joltz ed0f04257dac85bd4b7bf45521cc280f86d38ded；Boltz fork 7ea718a1440ecbe8e2418317192b501544147edd；DBMol arXiv:2607.19237v1。梯度到概率与合法联合编辑是 MolCraft 扩展，不是 DBMol 原论文中的生成算法。

# MolCraft 标准化算法

`▷` 后为中文逻辑注释。`proposal` 为尚未提交的候选；`probe` 为不改变当前图的临时评价。所有预算都有限；失败必须有明确状态与原因。本文件描述外层离散图编辑，不能把坐标扩散时间误认为化学图生成时间。

## Algorithm 1 — MolCraft 配体梯度与提议生成

**Require:** 冻结结构模型 Fθ，合法分子 G，固定蛋白上下文 E，组合目标 W，梯度模式 m，随机种子 ξ，候选策略 P。

**Ensure:** 与当前图严格对应的梯度证据及合法编辑概率包；输入契约不满足时返回明确失败。

```text
 1  VerifyContracts(Fθ, G, E, W, m)                         ▷ 核对词表、权重、口袋、目标与梯度路径
 2  B, I ← PrepareComplex(G, E)                            ▷ 构建原生特征及稳定原子到模型位置的映射
 3  X ← OneHotChemistry(G, I)                              ▷ 以当前原子、无向键与电荷作为可求导输入
 4  Ξ ← DrawSamplingNoise(ξ, B.layout)                      ▷ 固定一次结构预测使用的全部随机量
 5  Y ← PredictWithChemicalInputs(Fθ, B, X, Ξ, m)           ▷ 共享前向产生距离分布、原始坐标和评分
 6  L, metrics ← EvaluateObjective(Y, X, W)                 ▷ 在同一结果上计算组合目标与分解指标
 7  g ← DifferentiateInputs(L, X)                          ▷ 仅对三个配体通道求导，不更新模型权重
 8  VerifyGradientEvidence(g, I, m)                        ▷ 检查形状、有限性及原子 ID 是否一致
 9  Q ← BuildLegalProposalPacket(G, g, I, P)               ▷ 调用算法 2 构造受约束离散提议
10  return Q, GradientEvidence(g, metrics, I, Ξ, m)         ▷ 返回证据，不提交连续图或跨编辑计算图
```

## Algorithm 2 — 梯度到合法离散编辑概率

**Require:** 当前合法图 G，三通道梯度 g，温度 τ>0，正先验 ρ，化学约束和候选预算 P。

**Ensure:** 禁止变化概率为零，条件分布与联合分布分别归一化；没有合法编辑时返回 no_legal_edit。

```text
 1  D_A[i,k] ← g_A[i,k] − g_A[i,current_i]                  ▷ 估计元素替换方向的损失变化
 2  D_B[e,k] ← g_B[e,k] − g_B[e,current_e]                  ▷ 估计每条无向边的类别变化
 3  D_c[i,k] ← g_c[i] × (charge_k − current_charge_i)       ▷ 用连续电荷导数估计离散电荷候选
 4  M ← CheckStandaloneEdits(G, D_A, D_B, D_c, P)           ▷ 在原图副本检查单项变化，保留当前类别
 5  q ← StableMaskedSoftmax(−D/τ + log ρ, M)               ▷ 分别产生原子、键和电荷条件概率
 6  U ← SelectRawChangesByChannel(D, P)                    ▷ 组合前保留单独不合法但可能联合合法的变化
 7  C ← CompatibleSharedAtomBundles(U, max_size=2, P)      ▷ 构造无矛盾且共享原子的有限联合候选
 8  A ← {NO_OP}                                           ▷ 全局不修改动作只计算一次
 9  for a in OrderedBudgeted(C, P) do                      ▷ 以稳定顺序检查预算内候选
10      proposal ← ApplyTransaction(G, a)                  ▷ 整组执行，不把中间非法图提交为真实状态
11      if StrictValidate(proposal, P) then                ▷ 检查价态、芳香性、连接、立体和固定原子数
12          s_a ← ⟨g_A,δA_a⟩ + ⟨g_B,δB_a⟩ + ⟨g_c,δc_a⟩   ▷ 聚合联合方向的一阶得分
13          A ← A ∪ {(a, proposal, s_a)}                   ▷ 收集合格候选及其实际图
14      else record rejection reason                      ▷ 非法提议只记诊断，不调用 DeFoG 修复
15      end if                                            ▷ 完成当前候选的合法性判断
16  end for                                               ▷ 不无限重试，不独立采样逐原子类别
17  A ← KeepBestLegal(A, P); π ← StableSoftmax(−A.scores/τ) ▷ 联合候选单独归一化并保留 NO_OP
18  return Packet(Fingerprint(G), atom_ids, q, A, π)        ▷ 将源图、类别顺序和候选绑定后交给选择器
```

## Algorithm 3 — 单个候选重新评分

**Require:** PreparedComplex，绑定当前图的 EditCandidate a，原预测随机量 Ξ，原组合目标 W。

**Ensure:** 实际离散候选的评分和诊断；不修改源状态，不用预测的一阶得分代替真实评分。

```text
 1  VerifySourceFingerprint(a, current_graph)               ▷ 拒绝针对旧图生成的过期候选
 2  G' ← ApplyTransaction(current_graph, a.operations)      ▷ 按稳定 ID 在副本执行整组操作
 3  StrictValidate(G')                                    ▷ 验证最终图，不静默标准化、断片或改变电荷
 4  B', I' ← PrepareComplex(G', fixed_context)              ▷ 重建参考构象与全部相关原生特征
 5  VerifySameFixedLayout(I, I')                            ▷ 确认重原子身份、数量与噪声位置可对应
 6  probe ← PredictAndScore(Fθ, B', Ξ, W)                   ▷ 在一致随机设置下计算实际候选分数
 7  return CandidateEvaluation(G', probe, diagnostics)      ▷ 只返回评价，不自动推进优化状态
```

## 模块语义

- **PrepareComplex**：解析配体、固定原子 ID，准备本地蛋白特征，核验原生预处理没有改变图；构建原始原子坐标的显式索引。
- **PredictWithChemicalInputs**：向 ref_element、ref_charge 及连续键嵌入注入配体变量，同时同步连接指示；默认保留直接几何项的扩散反传。
- **DifferentiateInputs**：在当前 one-hot 点对三个输入通道求导；不对模型权重求导，不从饱和序列 logits 读取替代导数。
- **CheckStandaloneEdits**：逐位置概率对应只改变当前位置；原图保持不变，失败候选不会进入输出分布。
- **ApplyTransaction**：先完成整组相容操作再统一 sanitization；局部派生隐式氢重算，显式氢与立体声明保留。
- **StableMaskedSoftmax**：禁用类别严格为零，当前类别始终保留；NaN、Inf、非正温度或不合法输入明确失败。
- **CompatibleSharedAtomBundles**：最多两个操作，允许元素＋键、元素＋电荷等配合；不把同一变量的矛盾赋值组合。
- **EditSelector**：只选择已有候选 ID。首版 GreedySelector 为可执行参考接口，不代表 LLM 已接入或分数已经改善。

概率是温度与先验控制的一阶偏好，不是校准过的化学后验或实验成功概率。实际效果需按算法 3 重新评分。

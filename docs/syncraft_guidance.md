# SynCraft 基团与可合成性引导

当前协议为 **`syncraft-group-v2`**。候选池扩充、基团目录软偏好、概率前半方向提示、
草案反馈和覆盖率验收均采用新语义；不能恢复 v1 的轨迹或请求回执。

## 原 Agent 的审计结论

原 `optagent` 做了三跳邻域、七类 SMARTS 基团/环的上下文补全、RDKit 价态和连通性、自由基、冻结原子及已声明立体信息检查。
这些机制支持局部联合编辑，但不足以执行用户要求的“从高偏好原子中提出可合成基团”。

| 项目 | 旧路径 | 新 `syncraft` 路径 |
|---|---|---|
| 功能团完整性 | 在邻域内展示部分基团；超过区域上限时可能保留未补全邻域 | 补全至固定点，整组超过上限则放弃该区域；禁止静默截断 |
| 原子优先级 | 根据梯度排序，草案间可轮换到全部锚点 | 从最高梯度/概率偏好的最多8个原子出发；允许补全邻近支撑原子 |
| 提案结构 | operations、短 rationale、case_ids | 另需合成障碍、基团变换、目标基团原子、被替换原基团及可行性依据 |
| SynCraft 案例 | 5个相似案例，推理截取600字符 | 同一3272例语料，输入仅带最多2张300字符案例卡和来源ID，不发送不能执行的增删原子脚本 |
| 可合成性 | 无 SA 评分、无结构警报、无构象预检 | 保留SA预算、结构警报和有界ETKDG预检；基团目录只影响排序 |
| 共振表示 | 可被当作新候选计分 | 拒绝与源图等价的共振表示改写 |
| 复评候选选择 | 额外调用一次LLM选择ID | 目录匹配候选优先，随后补充其他合法候选；层内按梯度与SA排序，无额外select调用 |
| 候选生成 | 固定10个独立草案，重复后池可能不足8 | 默认争取16个不同有效候选，2个草案一批，最多32个草案槽位，再选8个复评 |
| 草案反馈 | 仅有历史SMILES排除项 | 同轮分批共享反馈，跨轮读取带真实指标的草案记录；程序检查完整组合历史 |

先前两条 Agent 轨迹中，33/33轮 LLM select 的候选集合与其候选池的固定概率Top-8相同。
本次将 LLM 用在基团提案与简短合成规划，数值排序由程序执行。
这是一项新的提案协议，不能将其结果与旧轨迹拼接为同一次实验。

## 与 SynCraft 的关系

已核对官方仓库 `catalystforyou/SynCraft-Core` 的
`557703e110b26a354afed2955ded5217837f6692`（本次检索的main版本）。
引导来自其 [skill/SKILL.md](https://github.com/catalystforyou/SynCraft-Core/blob/557703e110b26a354afed2955ded5217837f6692/skill/SKILL.md)、
`src/inference_enhanced.py` 的合成障碍诊断/最小编辑提示、实际golden edit案例，以及skill pipeline的SA排序思路。
来源文件和哈希见 `third_party/syncraft_guidance_sources.json`，MIT声明保留在 `third_party/syncraft.LICENSE`。

上游完整skill适用于已生成分子的可合成性救援，包含Vina/PLIP和重新对接。
当前模块复用它的化学规划与SA代理思想，适配MolCraft固定重原子数、梯度引导和Boltz复评；
没有执行上游完整救援流程、Vina/PLIP、逆合成路线搜索或库存核验。
LLM只能写简短设计依据，不得把案例类比写成已经验证的合成路线。

## 新提案的执行流程

1. 根据原子/键/电荷的下降方向和单项条件偏好对锚点排序，最多保留8个焦点原子。
   每个提案至少触及一个焦点；为保证完整基团，可包含邻近、非焦点的补偿编辑。
   概率仍是一阶偏好，不是经校准的“可合成概率”。
2. 扩展三跳邻域，补全相交的官能团与环直到不再变化。超过12重原子则丢弃整个区域。
   重叠的普通烷基三原子片段只用于产物识别，不作为完整性单元反复扩展，因此不会把15/31碳链吞成一个大区域。
3. LLM声明目标基团及完整原子ID。程序对实际执行后的产物重新匹配SMARTS/环结构。
   实际匹配能完整覆盖编辑原子和被替换原基团者进入目录优先层，不依赖LLM是否正确命名。
   目录无法覆盖实际产物时进入其他合法候选层，不凭目录拒绝分子；错误/未知声明保存警告，
   不把声明当作已匹配证据，也不降低程序已经确认的目录匹配。
   目录外变换可使用 `target_motifs: []`；错误原子ID、重复ID或非法JSON结构仍会被拒绝。
   保留原有价态、闭壳层、立体、冻结范围、最多8操作/6受影响原子和固定重原子ID检查。
4. 拒绝纯共振改写；检查新增的带电/多价卤素、非磷酰型磷、不常见杂原子三键、过氧键和三元杂环等保守结构警报。
   这些是本模式的设计限制，不是断言所有带警报结构都不可合成；已有未修复警报会在结果中保留记录。
5. 计算RDKit Contrib SA_Score，并执行ETKDGv3构象预检。
   默认SA≤6、相对源图增加≤2。参数是本项目的首版可调整预算，并非SynCraft论文给出的通用阈值。
   不强制每步SA下降，以免低SA的简单碳链无法引入功能团。
6. 对通过者计算 `selection_score = predicted_change + 0.25 × sa_change`。
   原始 `predicted_change` 保留真实梯度内积；SA调整分数单独记录。
   先按目录优先层、再按该分数排序，最多选8个做同深度Boltz复评。
   联合 `probability` 是该排序的序数偏好，不是原始梯度概率或结合概率；三通道方向表保留原始条件偏好。
   最终要求原有总loss、affinity_loss和所配置的覆盖率条件通过。

基团目录包括酰胺、酯、脲、氨基甲酸酯、磺酰胺、腈、胺、醚、醇、硫醚、羰基、烯烃、普通烷基/芳基卤化物和常见小环。
目录匹配证明结构身份，SA与构象检查是代理/预检，均不能证明实验可合成性。
保存的 `synthesis_assessment` 明确标记 `retrosynthesis: not_run`、`experimentally_synthesizable: not_established`。
ETKDG预检也不代替后续原生Boltz输入准备和结构复评。

## 候选池、输入和反馈记录

生产入口 `optimize` 和 `propose` 传入复评预算 K，目标池大小为
`max(optimization.drafts, ceil(K × synthesis.pool_multiplier))`，默认 K=8 时目标16。
每批结束后共享候选及拒绝反馈，至达到目标或耗尽 `synthesis.max_drafts`；每槽最多请求两次。
记录 `pool.target / unique_valid / shortfall / stop_reason`，不足时如实报告，不重复填充或额外增加Boltz预算。
低层API省略 `candidate_budget` 时，仍使用 `samples` 作为有限草案数。

方向提示在**每个原子/键/电荷条件行内**，排除原状态和零概率项，保留正概率变化方向中
`ceil(n/2)` 个概率最高项，平局按词表顺序稳定处理；不是累计概率达到50%，也不是跨所有行混排。
数值仅在序列化时取整，排序使用未取整概率。未展示的方向仍可组成合法联合编辑，完整图与完整校验继续保留。
输入保留最多2个短案例、精简的motif匹配规则、局部距离，以及不超过12条/5000字符的相关历史摘要。

每个轨迹额外保存 `draft_history.json` 和可阅读的 `draft_history.md`：

- 生成后先记录草案、操作组合、来源分子、目录匹配、SA和待复评状态。
- 复评后按同一轮次更新实际loss及变化、预测affinity、binder P、ipTM、覆盖率、接受/拒绝原因。
  未复评项明确没有测量指标，不能假设其改善。
- 保留每次失败请求的结构化草案和已返回用量；解析失败也记录finish_reason及可用的reasoning/cache用量。
- 下一轮读取与当前来源、局部原子相关的摘要。完整历史用于程序去重，操作顺序不同不视为新组合。
  来源分子改变后，同一原子操作可能产生新化学结构，因此仅把同来源的组合视为严格重复，其他来源作为类比。
- 恢复运行时不把当前轮的部分记录加入自己的初始提示，按已保存回执重建批次反馈；消息哈希不一致拒绝复用。

## 小分子设计中的覆盖率

本仓库 `bcov` 指配置口袋残基中至少有一个原子距配体不超过3.5 Å的比例。
`affinity_pred_value < 0 且 bcov > 0.5` 是项目成功判据，不是普遍的药化有效性标准。
紧凑小分子或片段可能只需覆盖少数关键残基；逐轮强制扩大覆盖范围可能破坏较好的局部结合姿态，
而起点是直链时从第一轮就硬要求成功门槛又可能阻断搜索。

v2 的syncraft YAML默认 `coverage_mode: preserve`：源图覆盖率已经严格超过门槛时，
拒绝退回门槛及以下的候选；源图尚未达标时允许探索，仍需loss和affinity_loss条件通过。
它不要求所有达标候选的覆盖率持续增加。`strict` 要求所有接受候选严格超过门槛，
适用于必须保持指定口袋占据的成熟系列；`off` 关闭该接受约束，适合另行定义口袋/目标的对照或片段任务。
阈值可按任务配置。启用时缺失bcov会报错，不静默跳过。

覆盖率通过不等于亲和力成功。复合 `affinity_loss` 包括二分类binder项，原始affinity预测仍可能变差；
v2会把实际复评中同时满足原始affinity<0和覆盖率门槛的最佳候选另存为 `best_successful.json`，
包括图、来源轮次和指标，即使它不是最低总loss的继续优化点。

## 开启方式与兼容性

新生成的用例默认显式写入 `optimization.strategy: syncraft`：

```bash
bash scripts/run_optagent.sh prepare-example --case vdr-3a2i \
  --msa-mode single_sequence --strategy syncraft \
  --model-dir /root/opt/mosaic-cache/boltz \
  --output outputs/vdr-syncraft/input
bash scripts/run_optagent.sh optimize \
  --config outputs/vdr-syncraft/input/mosaic-like.yaml \
  --output outputs/vdr-syncraft/trajectory
```

GBA继续使用已经核验的同序列MSA；VDR在上例中显式使用查询单序列。
可以在已有输入配置的副本中切换strategy并加入以下可选参数，使用新的输出目录：

```yaml
optimization:
  strategy: syncraft
  max_rounds: 25
  min_rounds: 10
  patience: 5
  drafts: 10
evaluation:
  candidate_budget: 8
  coverage_mode: preserve
  min_coverage: 0.5
synthesis:
  protocol_version: syncraft-group-v2
  pool_multiplier: 2.0
  max_drafts: 32
  draft_batch_size: 2
  focus_atoms: 8
  region_atoms: 12
  max_sa_score: 6.0
  max_sa_increase: 2.0
  sa_weight: 0.25
  check_conformer: true
  conformer_seed: 0
  conformer_iterations: 200
```

`molcraft propose` 在syncraft配置下也调用新Agent；随后 `evaluate-set` 采用相同的目录优先排序并核验上下文。
所有短理由、完整基团计划、真实SA值、警报与预检结果、案例ID和联合排序分数随 proposals.json 保存。
请求回执继续采用有界重试与可恢复日志，筛选阶段不新增LLM请求。

无strategy的旧配置和显式 `optagent` 继续使用旧Agent，`argmax` 继续完全不调用LLM。
两者的旧输入身份哈希保持不变；合成参数仅在syncraft策略中生效，其他策略设置非默认合成参数会报错，避免误以为对照组执行了相同筛选。
新协议将合成/覆盖率参数、版本号、引导文件、案例语料及Agent/组装/记录/请求模块源码哈希纳入身份，拒绝混用旧检查点。
显式v1配置会报协议不兼容；将配置副本升级为v2并使用新输出目录，不覆盖历史结果。
本版本更改合成协议语义时必须增加协议版本，避免在恢复的轨迹中改变候选定义。

## 验证范围

测试覆盖真实基团匹配、完整酰胺替换、区域溢出、15/31碳链的局部窗口、焦点与原子覆盖、SA限制、
ETKDG成功/失败、历史P≡C与带电S≡C问题、带电卤素、过氧键，以及GBA式共振改写。
同时检查新策略配置/工厂、提案与复评CLI路径、确定性SA排序、日志重放不重复请求、旧身份兼容及模式隔离。
v2另覆盖16个不同候选选8个、重复风暴有界停止、普通卤代物与目录外合法提案后备、目录层优先、
概率前半提示、同轮/跨轮反馈、实际指标写回、恢复后的反馈一致性、覆盖率分阶段验收和成功档案。
测试使用本地真实RDKit检查和模拟LLM/模型返回，不将它们表述为新一轮真实设计performance。

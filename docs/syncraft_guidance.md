# SynCraft 基团与可合成性引导

## 原 Agent 的审计结论

原 `optagent` 做了三跳邻域、七类 SMARTS 基团/环的上下文补全、RDKit 价态和连通性、自由基、冻结原子及已声明立体信息检查。
这些机制支持局部联合编辑，但不足以执行用户要求的“从高偏好原子中提出可合成基团”。

| 项目 | 旧路径 | 新 `syncraft` 路径 |
|---|---|---|
| 功能团完整性 | 在邻域内展示部分基团；超过区域上限时可能保留未补全邻域 | 补全至固定点，整组超过上限则放弃该区域；禁止静默截断 |
| 原子优先级 | 根据梯度排序，草案间可轮换到全部锚点 | 从最高梯度/概率偏好的最多8个原子出发；允许补全邻近支撑原子 |
| 提案结构 | operations、短 rationale、case_ids | 另需合成障碍、基团变换、目标基团原子、被替换原基团及可行性依据 |
| SynCraft 案例 | 5个相似案例，推理截取600字符，无实际编辑序列 | 同一3272例语料，提供原映射源图、编辑序列及最多2400字符说明，明确案例原子ID独立 |
| 可合成性 | 无 SA 评分、无结构警报、无构象预检 | 必须通过基团匹配、SA预算、新增结构警报和有界ETKDG预检 |
| 共振表示 | 可被当作新候选计分 | 拒绝与源图等价的共振表示改写 |
| 复评候选选择 | 额外调用一次LLM选择ID | 按梯度变化与SA变化联合排序，无额外select调用 |

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
3. LLM同时声明目标基团及完整原子ID。程序对实际执行后的产物重新匹配SMARTS/环结构。
   每个被编辑原子必须属于声明的真实产物基团；被破坏的原官能团/环必须按完整ID显式声明，且其所有原子由产物基团覆盖。
   保留原有价态、闭壳层、立体、冻结范围、最多8操作/6受影响原子和固定重原子ID检查。
4. 拒绝纯共振改写；检查新增的带电/多价卤素、非磷酰型磷、不常见杂原子三键、过氧键和三元杂环等保守结构警报。
   这些是本模式的设计限制，不是断言所有带警报结构都不可合成；已有未修复警报会在结果中保留记录。
5. 计算RDKit Contrib SA_Score，并执行ETKDGv3构象预检。
   默认SA≤6、相对源图增加≤2。参数是本项目的首版可调整预算，并非SynCraft论文给出的通用阈值。
   不强制每步SA下降，以免低SA的简单碳链无法引入功能团。
6. 对通过者计算 `selection_score = predicted_change + 0.25 × sa_change`，再形成联合候选偏好。
   原始 `predicted_change` 保留真实梯度内积；SA调整分数单独记录。
   程序选最多8个候选做同深度Boltz复评，最终仍要求原有总loss和affinity_loss接受条件通过。

基团目录包括酰胺、酯、脲、氨基甲酸酯、磺酰胺、腈、胺、醚、醇、硫醚、羰基、烯烃和常见小环。
目录匹配证明结构身份，SA与构象检查是代理/预检，均不能证明实验可合成性。
保存的 `synthesis_assessment` 明确标记 `retrosynthesis: not_run`、`experimentally_synthesizable: not_established`。
ETKDG预检也不代替后续原生Boltz输入准备和结构复评。

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
synthesis:
  protocol_version: syncraft-group-v1
  focus_atoms: 8
  region_atoms: 12
  max_sa_score: 6.0
  max_sa_increase: 2.0
  sa_weight: 0.25
  check_conformer: true
  conformer_seed: 0
  conformer_iterations: 200
```

`molcraft propose` 在syncraft配置下也调用新Agent；随后 `evaluate-set` 采用相同的合成调整排序并核验上下文。
所有短理由、完整基团计划、真实SA值、警报与预检结果、案例ID和联合排序分数随 proposals.json 保存。
请求回执继续采用有界重试与可恢复日志，筛选阶段不新增LLM请求。

无strategy的旧配置和显式 `optagent` 继续使用旧Agent，`argmax` 继续完全不调用LLM。
两者的旧输入身份哈希保持不变；合成参数仅在syncraft策略中生效，其他策略设置非默认合成参数会报错，避免误以为对照组执行了相同筛选。
新协议将合成参数、版本号、引导文件和案例语料哈希纳入身份，拒绝混用旧检查点。
本版本更改合成协议语义时必须增加协议版本，避免在恢复的轨迹中改变候选定义。

## 验证范围

测试覆盖真实基团匹配、完整酰胺替换、区域溢出、15/31碳链的局部窗口、焦点与原子覆盖、SA限制、
ETKDG成功/失败、历史P≡C与带电S≡C问题、带电卤素、过氧键，以及GBA式共振改写。
同时检查新策略配置/工厂、提案与复评CLI路径、确定性SA排序、日志重放不重复请求、旧身份兼容及模式隔离。
测试使用本地真实RDKit检查和模拟LLM/模型返回，不将它们表述为新一轮真实设计performance。

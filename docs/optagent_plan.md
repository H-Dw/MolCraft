# MolCraft OptAgent 实施计划

范围限定为 GBA／2V3D 与 VDR／3A2I 两个 DBMol 从头设计用例。使用 AF3 Server 返回的蛋白 MSA 作为 MolCraft 输入。正式配置为 `mosaic-like`，外层优化上限为 25 轮。远端任务根目录为 `/workspace/MolCraft`。

## 1. 锁定序列、口袋与 MSA

以用户提供的 `SynCoGen_folds_2026_11_03_11_18.zip` 为 GBA MSA 的原始来源。归档 SHA-256 为 `586489d641e46c854e8eb2583e56030158ceb40313c869b40f73b82cb7bac7b1`。构建体定义以同批交付的 FASTA、上传 JSON、区域说明和每靶点 `residue_map.csv` 为准。

| 输入 | GBA／2V3D | VDR／3A2I |
| --- | --- | --- |
| 蛋白链 | A | A |
| 保留的 UniProt 区域 | P04062 40–536 | P11473 118–164、216–427 |
| 最终查询长度 | 497 aa，保留沉积序列中的 R534H 差异 | 259 aa，保留 H305F 和 165–215 内部删除，去除 N 端 GSHM |
| 实验中有坐标的目标残基 | 493 | 253 |
| 参考配体与初始图 | NBV，15 重原子，起点为 15 碳直链 | TEJ，31 重原子，起点为 31 碳直链 |
| 3.5 Å 口袋在最终查询中的位置 | D127、F128、W179、N234、E235、F246、E340、W381、N396 | Y26、S69、R106、S107、S110、F137、H229 |
| AF3 Server MSA 状态 | 未配对 A3M 已核验，含查询共 15,656 条 | 十靶点归档无此任务，等待单独返回同序列的 MSA |

GBA 使用归档成员 `syncogen_2v3d_gba1_497aa_target/msas/fold_syncogen_2v3d_gba1_497aa_target_unpaired_msa_chains_a.a3m`。归档中的 job request、A3M 首条查询及 `03_2V3D_GBA1/target.fasta` 已核验逐字符一致。未配对 A3M 的 SHA-256 为 `f03b893a320d64ad34dc35943410af530461c1ea96129be3d4c2f99b160ba7f7`。单蛋白链特征使用未配对 MSA，配对文件保留在原始归档中。

VDR 的查询使用已整理的 `syncogen_structure_audit_20261103/sequences/individual/3A2I_target_region.fasta`。据此生成独立的 259 aa 蛋白 AF3 Server 任务，设置 `count: 1`、`useStructureTemplate: false` 并让服务端对最终序列搜索。现有 `AF3_Server_UPLOAD_VDR_3A2I_PROTEIN_ONLY.json` 使用 263 aa 带标签构建体，不能作为该 MSA 的上传输入。VDR 收到同序列的未配对 A3M 后，执行与 GBA 相同的导入核验。GBA 的验收可以先行。

查询序列采用去除 FASTA 标题及换行后的 ASCII 内容计算哈希。

| 查询 | SHA-256 |
| --- | --- |
| GBA 497 aa | `974fdac6caedba83154e723459a849159558c56a936a9344239dd5c933e5d1a3` |
| VDR 259 aa | `95dd56408e9c0665b5ca170e7656bdd291843ce3243b352318d67e2bfd3a0a5f` |

数据准备与导入要求如下。

1. 固化最终 FASTA、原始 PDB 版本、配体实例、源 `label_seq_id`、作者编号和裁剪后序列位置。用晶体配体重原子到蛋白重原子的 3.5 Å 邻近关系核对口袋。裁剪遵循构建体映射，保留查询中缺失坐标的残基位置。晶体配体信息保存在参考清单中，生成配置只传递口袋索引和初始图的重原子数。[GBA 结构](https://www.rcsb.org/structure/2V3D) · [VDR 结构](https://www.rcsb.org/structure/3A2I)
2. 在任务数据目录导入原始未配对 A3M，核验来源、首条查询与序列哈希。逐行检查忽略小写插入后的对齐宽度。保留查询和原始顺序下最多 1,023 条不同的非查询有效记录，保留入选记录的小写插入，生成 Boltz 实际读取的 A3M。记录原始深度、筛选后深度和两个文件的哈希。
3. `test` 和 `mosaic-like` 的真实靶点运行都要求有效的 AF3 来源 MSA。输入与最终 FASTA 不一致，或经解析仅剩查询时，将对应靶点置为待补数据。CPU 验收使用本地夹具。运行时从 `msa_path` 读取缓存，不触发 MSA 服务。AF3 蛋白预测坐标保留为来源记录，MolCraft 使用自身的 Boltz-2 前向预测复合物。

## 2. 外层优化轮次与预算

DBMol 对每个目标做 50 次连续分子梯度更新，SGD 学习率为 2.0，随后从 \(t_0=0.8\) 进行 200 步 DeFoG 图去噪。基准设置产生 2,048 个图候选，再随机取 100 个合法分子评估。论文观察到 Boltz2 Success 早期增长后趋于饱和，过度优化会降低投影后的有效率。[DBMol §4.2、§4.4 与附录 C](https://arxiv.org/html/2607.19237v1#A3)

MolCraft 每轮在当前合法分子上求输入梯度，构建多个离散候选，并用 Boltz-2 复评后决定是否接受。轮次预算同时决定梯度次数、API 次数和复评开销。25 轮是基于该工作量和碳链起点设定的首版预算，两个靶点使用同一规则。5 轮保留为轨迹观察点。

| 参数 | `test` | `mosaic-like` |
| --- | ---: | ---: |
| 外层优化上限 | 1 | 25 |
| 允许收敛退出前的最少轮数 | 1 | 10 |
| 连续无接受编辑的退出阈值 | 不适用 | 5 轮 |
| 每轮独立 Agent 草案 | 2 | 10 |
| 每轮候选复评上限，另含当前分子基线 | 2 | 8 |
| 梯度阶段 Boltz-2 坐标扩散步数／trunk passes | 25／1 | 25／1 |
| 复评阶段 Boltz-2 坐标扩散步数／trunk passes | 25／1 | 200／4 |
| 亲和力不恶化约束 | 关闭 | 开启 |

一次正式轨迹上限为 250 个 Agent 草案、200 次候选复评及 25 次基线复评。第 5、10、15、20、25 轮保存实际到达的轨迹快照，记录有效且唯一候选数、接受次数、目标改善、运行时间和 API 用量。第 10 轮起，如果连续 5 轮没有接受编辑则停止。目标、模型、MSA 和随机量相同且分子图未变时可以复用梯度，同时轮换锚点并排除重复提案。

如果第 21–25 轮仍有接受的改进，将 50 轮作为单独标注的长程对照预算。25 轮正式结果与长程结果分别保存，保持每轮候选与复评设置一致。这里的 200 步用于 MolCraft 的 Boltz-2 坐标扩散，DBMol 的同名步数用于 DeFoG 图去噪。

## 3. OptAgent 与候选接受

- `OptAgent.propose()` 接收当前合法图、原子／键／电荷方向分数、稳定 ID、冻结区域、邻域化学和口袋上下文。沿用固定版本 SynCraft 的 3,272 条案例，用半径 2 的 2,048 位 Morgan 指纹检索 5 个案例。案例提供可追溯的编辑思路。[SynCraft 固定版本](https://github.com/catalystforyou/SynCraft-Core/tree/557703e110b26a354afed2955ded5217837f6692)
- 每个草案最多 8 项编辑和 6 个受影响原子。两例采用三跳邻域，补全相交的功能团与环，区域上限 12 原子。执行元素、无向键和形式电荷的联合编辑，保持重原子数与稳定 ID。程序检查价态、芳香性、连通性、自由基、既有立体信息与冻结范围。规范异构 SMILES 用于化学去重。
- 程序根据合法最终图的实际特征差分计算 \(\widehat{\Delta L}=\langle g_A,\Delta A\rangle+\langle g_B,\Delta B\rangle+g_c^\top\Delta c\)。在合法候选与保留当前图的选项上，用 \(q\propto\exp(-\widehat{\Delta L}/\tau)\) 归一化，\(\tau=1\)。`OptAgent.select()` 只返回已验证的候选 ID。
- 通过 `evaluate_packet()` 对 Agent 选择的 ID 复评。基线与候选使用相同深度和按稳定 ID 耦合的随机量。继续使用完整结构梯度以及亲和力、接触、置信度、几何四项目标，权重为 1、2.5、2.5、1。正式配置在实际联合损失下降且亲和力损失不恶化时提交新图，随后重新准备特征并求导。
- 使用 DeepSeek `deepseek-flash` 和 `DEEPSEEK_API_KEY`，启用 `thinking` 与 `reasoning_effort=high`。独立请求并发 2、每次上限 16,384 tokens、超时 600 秒。每个草案最多 2 次调用，ID 选择最多另调用 2 次。JSON 和编辑包都经本地校验，日志记录简短理由、案例 ID 和调用状态。[DeepSeek 接口](https://api-docs.deepseek.com/api/create-chat-completion/)

## 4. 参考分数和评估

下面的配体级分数来自 DBMol 论文，格式为 Boltz-2 Affinity／ipTM。蓝色展示候选经过 SynFormer 处理，单独记录其来源与处理状态。[DBMol PDF 第 5 页、图 2](https://arxiv.org/pdf/2607.19237v1#page=5)

| 靶点 | 晶体参考配体 | 三个展示候选 |
| --- | --- | --- |
| GBA | 0.58／0.99 | −0.31／0.98，−1.07／0.97，−0.24／0.96 |
| VDR | 0.22／0.99 | −1.52／0.98，−1.06／0.99，−1.23／0.96 |

MolCraft 报告实际输入的序列、MSA、初始图与每轮接受状态，分别输出原始 `affinity_pred_value`、binder probability、`ligand_iptm`、优化分项和联合损失。用预测复合物重原子坐标计算 BCov、Dist 与 Boltz2 Success 诊断。论文 Affinity 与 MolCraft 的 `affinity_loss` 分列。论文表 1 的集合均值保存在论文参考表中，流程通过以真实合法编辑及复评为准。[DBMol 表 1 和附录 C.7](https://arxiv.org/html/2607.19237v1#A3.SS7)

## 5. 部署和验收顺序

1. 在本地完成模块设计、带 Require／Ensure 和中文行注释的算法、实现与 CPU 验证。测试覆盖序列和 MSA 错配、原始与裁剪编号映射、碳链及六元环闭合、候选合法性、同一上下文去重、外层退出规则、重试和恢复。
2. 远端统一从 `/workspace/MolCraft` 运行。原始归档、MSA 派生文件、临时文件、缓存、轨迹和结果均放在该目录内。检查真实路径及挂载，隔离 private 盘。复用现有 DTK 26.04 环境和已核实的非 private 只读 Boltz-2 资产。GitHub 发布源码、文档和数据来源清单，服务返回的 MSA 随任务数据存放。
3. 用交互式 Bash 加载 `~/.bashrc`，让 Git 和 DeepSeek 继承现有环境。启动前核验带鉴权的 `GET /models` 返回 HTTP 200，且列表包含 `deepseek-flash`。只记录状态、模型可用性和 Git 提交。用 GitHub 官方地址核验发布 SHA，远端优先从官方地址检出。官方连接异常时依序拼接完整仓库 URL 到 `https://ghfast.top/`、`https://gh-proxy.com/`、`https://ghproxy.net/`、`https://gh.llkk.cc/`。镜像使用匿名连接，分支引用和实际拉取的提交均须匹配官方 SHA，随后快进工作树。`origin` 使用官方地址。
4. 先为 GBA 的真实 DTK `test` 和首个正式轮次预留两小时校准。`test` 使用已核验的 497 aa 查询和 A3M，记录至少一个 Agent 提出的合法非空编辑的结构复评结果。正式首轮验证 `200／4` 的峰值显存、编译和运行时间，并计入 25 轮。按每轮最多 8 个候选估算剩余窗口并加 25% 时间余量。VDR 在 259 aa AF3 MSA 核验后走同一流程。报告记录设备、提交号、输入哈希、完成轮数和分数。遇到可恢复故障时在本地修复、发布并在远端复测。

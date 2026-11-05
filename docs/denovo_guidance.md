# De novo Agent：实现与运行契约

本策略对应 2026-11-05 批准的修订计划。显式设置 `optimization.strategy: denovo`；
原 `syncraft`、`optagent`、`argmax` 及旧配置继续保留。新策略是固定重原子数的
分子图设计，不要求早期碳链已经具有药效团，不把目录命中或 SA 阈值当作拒绝条件。

## 科学目标与来源

默认目标为 `L_affinity + 2.5 L_contact + 2.5 L_confidence`，协议
`dbmol-denovo-v1`。保留完整结构反传；不会随权重配方改变梯度路径。

| 参数 | 生效值 | 来源／含义 |
|---|---|---|
| affinity | 预测值 − binding logit，外权重 1 | DBMol 附录 C.1 |
| contact | 8 Å 内距离概率；口袋平均与最优锚定各 0.5；外权重 2.5 | DBMol |
| confidence | −iPTM −0.3 ipLDDT +0.3 ipDE +0.1 PDE −0.3 confidence score −0.1 pTM；外权重 2.5 | DBMol |
| geometry | 外权重 0；仍输出真实 pocket/clash 诊断 | MolCraft 本地项，未标定 |
| geometry 旧诊断参数 | 6 Å、温度 0.5、碰撞半径缩放 0.75；内部两项等权 | 本地工程初值 |
| confidence score | `(4 × complex_pLDDT + interface_score)/5` | Boltz 公式；界面项采用本地 ligand–protein iPTM |
| ipLDDT 聚合 | ligand/interface/noninterface 为 20/10/1；8 Å 定义界面 | Boltz/Joltz 原生实现 |
| 设计／复评深度 | 25/1 与 200/4（扩散步／trunk passes） | Mosaic 计算流程；不是 loss 系数 |
| 方向温度／类别先验 | 默认 1／均匀，可配置 | 本地工程设定，未经概率校准 |
| 候选资源 | 16 个候选、4 草案/批、最多 8 批、每请求 2 次尝试、64 次程序补充、6 张证据卡 | 工程预算，非化学定律 |

来源：[DBMol 附录](https://arxiv.org/html/2607.19237v1#A3.SS1)、
[Mosaic 固定版本](https://github.com/escalante-bio/mosaic/tree/5937f37164bcb28bc6f1000f10134145438e5971)、
[Boltz 固定版本](https://github.com/escalante-bio/boltz/tree/7ea718a1440ecbe8e2418317192b501544147edd)、
[Joltz 固定版本](https://github.com/nboyd/joltz/tree/ed0f04257dac85bd4b7bf45521cc280f86d38ded)。
这些是文献支持的起始值，尚未在 MolCraft 上标定。BindCraft 的 PAE 经过 /31，
ProteoCraft 的 0.25 是归一化梯度混合系数，均不搬入本项目的 geometry 权重。
参考：[BindCraft 默认配置](https://github.com/martinpacesa/BindCraft/blob/main/settings_advanced/default_4stage_multimer.json)、
[ProteoCraft 复评源码](https://github.com/smiles724/ProteoCraft_v1/blob/3b84026e134dab751bb1fa071cdda3fbdc8d33c4/proteocraft/hallucination/esmfold2_one_step.py)。

量纲保持原定义：iPTM/pTM/pLDDT/confidence score 为 0–1；PDE/ipDE 为 Å，
没有新加归一化。ligand_iPTM 使用配体—蛋白掩码；原生 ipDE 使用全部跨链对，
原生 ipLDDT 对全部非聚合物及跨链界面加权。在当前单蛋白链、单配体输入中二者
对应目标界面；推广到多蛋白链/多配体时必须另审计这些掩码，不能宣称已经适配。

`weight_audit.json` 保存各参数来源，`effective_config.json` 保存实际配置，梯度文件
保存模型读取的 sigma/noise/step 参数。`denovo.gradient_diagnostics: true` 额外进行
最多四次同随机量反传，记录各项原始值、加权贡献、梯度范数和余弦冲突；默认关闭，
避免把诊断计算算成免费的设计预算。geometry 的 soft-min 随原子对数量变化、碰撞均值
可能稀释局部冲突，这些局限仍存在，因此它暂不参与新基线的排序和接受。

## 真实数据库依赖

[SynCraft 论文 §2.1](https://arxiv.org/html/2512.20333v1#S2.SS1) 的合成评估是
SimpRetro＋USPTO 模板＋Enamine 终止库存。eMolecules 约 2300 万条商业分子用于
构造 synthesis-cliff 案例，不作为本版运行时基团库存。公开
[SynCraft skill pipeline](https://github.com/catalystforyou/SynCraft-Core/blob/557703e110b26a354afed2955ded5217837f6692/skill/syncraft/pipeline.py)
不等同于运行完整逆合成搜索。

获取清单是 [`configs/denovo_resources.json`](../configs/denovo_resources.json)。它不是
可运行索引，也不包含虚构的库存 SHA-256。Enamine 官方 2026-10 US Stock 计划条数
292,887；下载需登录，当前仍是正式运行前依赖。不能用案例产物或 SimpRetro 示例的
`emolecules.txt` 替代。SynCraft 原实验精确库存快照未公开，本接入只声明同类真实库存。

公开模板已固定为 SimpRetro `5abbdd12d23bec3b0c524a3f5e59d160beac00f3`，51,720 个字符串，
SHA-256 `f3a0546b1cd50c463a1df2b665ffe8d14e021f5babfc13c749601f469518a00f`。
没有频次元数据，不能声称复现论文的“至少出现两次”筛选。模板按逆合成方向读取，
左侧为产品；匹配只提供产品模式/连接依据，不证明反应物在库或完整路线存在。

获得官方文件后，在独立位置保存原件、下载日期及供应商版本，构建不可覆写索引：

```bash
molcraft build-inventory \
  --stock data/inventory/sources/enamine_us_202610.sdf.gz \
  --output data/inventory/enamine_us_202610.sqlite \
  --release '2026-10 US Stock' \
  --source-url https://enamine.net/building-blocks/building-blocks-catalog \
  --expected-raw-count 292887 \
  --templates data/inventory/sources/SimpRetro_templates.json \
  --template-source-url https://github.com/catalystforyou/SimpRetro/blob/5abbdd12d23bec3b0c524a3f5e59d160beac00f3/SimpRetro_templates.json
molcraft check-inventory --manifest data/inventory/enamine_us_202610.sqlite.manifest.json
```

如果已有可信的下载校验值，同时提供 `--expected-sha256`。若官网版本或原始条数变化，
先更新获取清单，不忽略不一致。支持 SDF、SDF.gz、SMI/SMILES；不接收案例 JSON。
索引保留商品编号、源行号、规范结构、完整局部 SMARTS、原始/有效/去重/无效计数、
RDKit 版本、原文件及索引 SHA-256。保留电荷、氢和立体化学，不自动中和或脱盐；
不支持的多组分记录有计数。库存文件位于 Git 忽略目录，不随代码发布。

基团由环、杂原子及邻接原子起始，闭合环系和共轭/多键单元，连接点保留直接相邻原子
的元素、电荷、氢数、度数、芳香性及键型。核心还约束立体化学；核心包含已声明手性中心时，
保守地保留完整库存结构，防止切断后相似连接点交换造成对映体误匹配。它是库存中的局部环境，
不是“这个分子已能买到”。检索使用高概率方向和不同位置的元素可用性，最多读取每类
512 条相关记录、返回少量卡片；这是有界近似检索，未命中不等于数据库不存在该基团。

三种证据分别命名：`inventory_fragment`、`reaction_template_support`、完整路线验证。
本版最后一项始终 `not_performed`。仅模板命中不获得“库存完整基团”的优先级。
最终产物实际匹配库存核心，且所有编辑位置都被覆盖、高概率方向条件成立，才能获得
目录优先；其他合法候选继续参与选择。目录缺失则明确报未就绪，绝不回退案例库。

## 每轮流程

1. 对当前离散分子做梯度反传。每行只将概率排名前半的非当前方向写入提示；完整表
   保存在 `proposals.json`。组合编辑不会因某一个单项暂时价态非法而被方向表提前屏蔽。
2. 从有梯度偏好的位置展开化学上下文，闭合环与共轭结构并包含连接边界。窗口只读，
   全部分子图和可编辑 ID 同时给出；不再用 6/8/12 个原子作为化学拒绝条件。Agent
   可以请求下一批扩充方向上下文。每批共享近期草案结果及与当前源图/局部有关的实测摘要。
3. 每次请求最多四个草案，最多八批，每个请求最多两次 API 尝试；解析成功后的重复
   或化学失败只进入反馈，不触发无限重试。空值、缺字段、截断响应保留已知 token usage。
4. 单项和组合使用同一个完整图编译器，原子性应用全部操作；检查连接、闭壳层、价态、
   芳香性、氢、映射、冻结位置和既有立体化学。拒绝纯共振表示改写。随后使用实际 Boltz
   原生准备流程检查构象/特征，并将准备结果保留到本轮复评，避免再次生成不同输入。
5. 目标是 16 个去重且原生准备成功的候选。LLM 不足时，最多尝试 64 个梯度排序的
   单项/双项补充。记录 `llm`、`program_fallback`、`carried_unmeasured` 来源。不能保证
   任意小分子都有 16 个合法未测候选；不足时如实输出 shortfall，不重复填满。
6. 在高概率更新中先选有完整库存局部结构证据的候选，按编辑中心和基团证据类型分散，
   再补其余合法候选。组内按联合梯度预测变化排序。没有 `0.25×ΔSA` 或 `0.85/0.15`。
   SA 和 alerts 只输出诊断。选择顺序与梯度 softmax 概率分别保存，不再把名次概率叫方向概率。
7. 最多八次候选 Boltz 复评；源分子另作同深度基线。同一随机量跟随稳定原子 ID 重映射。
   候选构象失败或非有限模型输出占用候选尝试预算并隔离；映射/配置损坏仍报告错误。
8. 仅总 loss 改善才接受。affinity<0 且 coverage>0.5 的成功结果单独归档，即使它没有
   获得最低总 loss。关闭覆盖率接受 gate 不再关闭成功归档。覆盖率可能随总 loss 优化下降，
   最终应同时检查 `best_successful.json` 和最终结构。

`draft_feedback.json/.md` 每批更新，`draft_history.json/.md` 每轮追加指标。
已生成但未复评的草案只在同一源图和协议下接续；已复评候选不会再进入后续轮次预算。
重启恢复按源图、权重、数据库内容、代码、依赖版本及随机量校验。
协议变化会拒绝旧缓存；新权重必须重新计算梯度和评分，不直接横比两配方的 loss。

## 正式 GBA/VDR 运行

复用先前两条轨迹的蛋白/MSA/初始配体/随机种子，生成新配置目录：

```bash
python scripts/prepare_denovo_run.py \
  --gba-config outputs/syncraft-v2-fixed-rounds-20261105/configs/GBA.yaml \
  --vdr-config outputs/syncraft-v2-fixed-rounds-20261105/configs/VDR.yaml \
  --inventory-manifest data/inventory/enamine_us_202610.sqlite.manifest.json \
  --output outputs/denovo-20261105/configs
```

生成配置固定 GBA 17 轮、VDR 16 轮，`min_rounds=max_rounds`，每轮最多 8 次复评。
`prepare-example --strategy denovo --inventory-manifest ...` 也可以生成新输入，但配对比较时
应优先复用旧输入。库存校验通过后使用既有 DCU 启动环境和顺序队列；两个新输出目录
分别运行 GBA 然后 VDR。未取得库存时只准备配置，不启动正式轨迹。

核查输出：亲和力、binder 概率、contact/confidence 原始项和加权贡献、各 confidence
子指标、覆盖率、geometry_pocket/clash、数据库匹配、池缺口、API/程序贡献、每批 usage。
CPU 测试包含公式/梯度有限差分、组合价态、连接环境、对映体、目录无命中、原生准备
复用、空响应、去重、池不足及跨轮重启。测试库存明确为合成夹具，不是 Enamine 数据，
这些测试不代表 Boltz 新策略已获真实性能提升；正式设计效果仍须真实轨迹验证。

# Agent / argmax 配对对照与显存门控

本轮对照固定为 GBA/2V3D 和 VDR/3A2I。GBA 使用已有 497 aa 输入与核验后的 AF3 MSA；
VDR 使用 259 aa、H305F、内部删除 165–215 的构建体，按照用户要求不使用同源 MSA 或模板。
VDR 的 Agent 和 argmax 共用相同单序列输入。两靶点的 MSA 条件不同，比较应在靶点内部进行。

## 可选运行路径

配置 `optimization.strategy: optagent`（默认）使用 DeepSeek 提案与选择；
配置 `optimization.strategy: argmax` 使用确定性的合法编辑枚举和概率 Top-k，不初始化或请求 LLM。
原子、键与电荷三类编辑复用现有合法候选生成器，保留单项不合法、联合合法的编辑。
排序键为概率降序、预测损失变化升序、稳定候选 ID；剔除 noop 和同一源分子上已复评的化学结构。
noop 作为单独基线复评。候选不足时不重复填充。

正式对照每轮最多复评 8 个候选，25 轮上限，10 轮后连续 5 轮无接受编辑时停止。
梯度 25 步/1 pass，复评 200 步/4 passes，seed=0，亲和力不恶化保护开启。
初始图分别为 15 与 31 碳直链；argmax 不从 Agent 的最终分子继续。
既有枚举器最多组合两项编辑，Agent 可提出更大的编辑包，因此该实验衡量整个 Agent 路径，
不能把结果单独归因于末端选择。记录实际候选复评数与运行时间，Top-8 是相同上限而非保证每轮填满。

## 单序列输入

`run.msa_mode: provided` 是旧默认，优化 CLI 仍要求核验过的本地 MSA 及来源文件。
`run.msa_mode: single_sequence` 明确关闭同源 MSA，禁止同时提供 msa_path。
后端传递 `msa: empty` 且 `use_msa_server=False`，保留模型需要的查询序列行，检查其深度为 1。

```bash
bash scripts/run_optagent.sh prepare-example --case vdr-3a2i \
  --msa-mode single_sequence --output outputs/vdr-control/input \
  --model-dir /root/opt/mosaic-cache/boltz
```

随后按需要在生成的配置中设置 strategy，并使用不同输出目录调用 optimize。
`runtime.json`、`feature_profile.json`、`checkpoint.json` 和逐轮文件记录策略、MSA 条件和实际输入。
默认 OptAgent/provided 的身份哈希保持旧格式兼容；不同策略或 MSA 条件禁止共用检查点。

## 服务器队列

`scripts/run_control_queue.py` 是标准库独立调度器，可复制到 outputs 下运行而不修改活跃任务源码。
它读取任务清单：work_root、queue_dir、not_before_epoch、pci_bus、memory_threshold、poll_seconds、
required_commit、cpu_python、gba_source_config、model_dir 和四条 jobs（id、case、strategy、msa_mode、config、output、existing）。
清单不保存 SSH 密码或 API 密钥；Agent 继承服务器既有环境，argmax 不需要 API 密钥。

首次到期时间为北京时间 2026-11-04 07:33，保存绝对时刻，重启不重新开始六小时倒计时。
只检查分配给该实例的 PCI 设备，不使用主机其它卡的显存平均值。
已用/总显存必须严格小于 10%；等于 10%、读数失败或原 Agent 尚活跃时，每 1800 秒重新检查。
到期以前不拉取、不加载模型、不生成新任务。

原 GBA Agent 已完成则复用结果；若其进程已退出但轨迹未完成，先恢复该轨迹。
随后依次运行 GBA argmax、VDR Agent、VDR argmax。每条启动前重新检查显存。
保持全队列与轨迹锁，并通过 PID 和进程起始时间识别幸存子进程，避免断线或调度器重启造成重复运行。

首个门控通过后查询 GitHub 官方 main，复用 sync_optagent.sh 快进到该 SHA，确认包含 required_commit。
在实际检出的版本使用独立的官方 CPU JAX 环境执行测试后固定该批次提交；同步失败等待 30 分钟，不运行旧代码。
运行中不改变 checkout；工作树脏或后续 SHA 变化时拒绝启动。模型环境保持原 DTK 版本。
任务失败独立记录并继续后续任务，绝不自动改成另一策略。失败重试和恢复由检查点及请求回执约束。

queue_state.json 与日志保留定时状态，comparison.json/CSV 汇总全部四条轨迹。
这些数据包含完成/失败状态、版本、输入哈希、接受编辑数、实际复评数、分数和结构输出引用。
仅比较一个随机种子的计算轨迹，不代表实验活性或统计显著性。
调度器脱离 SSH 与本地客户端运行；服务器实例重启后需要重新启动同一调度器以恢复。

服务器 CPU 的 float32 小型矩阵乘法探针出现不正确数值，导致四项原有特征映射测试失败；
官方 JAX 0.6.2 CPU 环境也能复现。使用同一环境的 x64 CPU 验收后，106 项测试全部通过。
官方 CPU 验收环境独立放在队列目录的 cpu-env，清除 LD_LIBRARY_PATH，设置 JAX_ENABLE_X64=1。
不更换实际 DCU 生成所用的 DTK JAX，也不改变 DCU 精度或跳过测试。
远端原生 VDR 特征准备仍已在原模型环境核验。

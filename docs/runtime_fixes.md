# 原生模型接口的联调修复

本次远端使用已有 DTK 26.04 环境，所有代码修复均先在本地完成，再提交到
`H-Dw/MolCraft`，远端通过 `git pull --ff-only` 获取并重新安装。没有替换系统
DTK、已有 JAX 插件、全局 PyTorch 或 Mosaic 源码。

| 接口 | 故障及原因 | 实现与回归检查 |
|---|---|---|
| Python／JAX | 原安装声明仅允许 Python 3.12；旧 JAX 不支持 full SVD 的 JVP | 允许 3.11／3.12；3×3 Kabsch 使用 reduced SVD，验证导数有限 |
| 离线安装 | editable 安装需要环境中没有的 `editables` | 已有 hatchling 直接构建普通 wheel，使用 `--no-deps --no-build-isolation` 安装 |
| 元素 | 原生碳元素位为 6，适配器误用了 5，严格一致性检查拒绝输入 | 元素映射直接使用原子序数；蛋白范德华半径读取也使用该序数 |
| 键 | 原生单键为 2，适配器误用了 1 | 原生 token 键编码为化学键 ID 加一；无键仍为 0；类别嵌入与连接指示同步 |
| 原生 CIF | 写出器查找 `processed/complex.npz`，实际文件在 `processed/structures/` | 更正目录；传入 `[samples, atoms, 3]` 坐标与 `[records, atoms]` 掩码，写出器自行增加记录维度 |
| 参考构象 | 含磷环状候选在 UFF 构建时发生 `RingInfo not initialized` | 在私有分子副本上初始化化学缓存；UFF 参数检查之后重建环信息；去氢后再次初始化缓存 |
| 原生随机增强 | featurizer 的参考构象增强使用 PyTorch RNG，NumPy 种子未能控制，跨进程源分数波动 | 用隔离的 CPU PyTorch、Python、NumPy 随机上下文覆盖导入及原生特征准备；结束恢复调用方状态，不触碰 GPU RNG |

元素与键修复对应固定 Boltz 提交
`7ea718a1440ecbe8e2418317192b501544147edd`，不是通用词表的零基索引约定。
例如本项目的元素顺序为 C、N、O、F、P、S、Cl、Br、I，原生列分别为
6、7、8、9、15、16、17、35、53；五种键类别对应原生列 0、2、3、4、5。
保留完整原生嵌入矩阵；没有重新编号或重新训练其参数。

参考构象修复保留固定重原子数、atom-map、元素、电荷、键和原子名称。回归用例
覆盖原生风格的缺失缓存，并模拟 UFF 类型判断后清空环缓存的边界。所有操作发生
在分子副本上；原始分子不变。重新初始化之后仍执行图一致性检查。这里不吞掉
化学错误，不移除导致故障的元素，也不跳过含环候选。

`scripts/check_native_features.py` 使用真实原生资源检查重复种子的逐字节特征一致性、
不同种子的参考坐标差异及三套调用方 RNG 状态恢复，不加载 checkpoint 或运行模型。
真实模型脚本另外重建源分子并验证其特征和分数重放，避免把未控制的原生随机增强
混入候选优化效果。旧的未控制 RNG 运行仅作为诊断记录，不作为最终测评数值。

## DTK 的 CPU 路径边界

当前厂商 JAX `0.6.0+das.opt1.dtk2604` 的 CPU 路径存在独立的矩阵乘法问题：
`eye(5) @ arange(15).reshape(5,3)` 返回错误数据。同一环境的 DCU 路径返回正确
结果，本地官方 JAX 0.6.0 的 CPU 路径也正确。这个最小复现不依赖 MolCraft。

因此远端模块检查显式启用当前 DCU；本地 CPU 检查使用官方 JAX 的独立环境。
不能把远端厂商 JAX 的 CPU 结果当作本项目的有效 CPU 验收，更不能用这些结果
生成梯度证据。真实模型脚本也会在检测不到 GPU／DCU 时直接失败。

```bash
source /root/opt/mosaic/activate_dtk.sh
cd /root/opt/MolCraft
MOLCRAFT_RUN_MODEL_TESTS=1 python3.11 -m pytest -q -m 'not model_integration'
```

上述命令只检查模块；真实结构／亲和力权重的端到端检查使用独立的
`scripts/test_model.py`。模型检查的最终状态、候选实际分数与产物检查以运行报告为准。

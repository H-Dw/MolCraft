# 双 GPU 并行运行 GBA / VDR

`scripts/run_parallel_denovo.py` 同时启动并监督两条独立轨迹，各自只有一张 GPU 可见。
配置要求：`denovo`、`msa_mode: provided`、GBA 17 轮、VDR 16 轮、每轮最多 8 次候选
复评，`denovo.max_api_retries: 3`。两条任务使用独立输出目录、缓存、临时目录和请求回执。

对 DTK/ROCm，启动器在激活环境之后设置 `HIP_VISIBLE_DEVICES`，清除可能产生重复
重映射的 `ROCR_VISIBLE_DEVICES` 和 `CUDA_VISIBLE_DEVICES`。优化入口要求每个进程
恰好看到一个加速设备，运行记录保存设备数量及可见性配置。编号指容器实际可见设备，
不能从宿主机全部 DRM 设备推断；部署前须核对分配的设备节点及单设备探测结果。

```bash
source /root/opt/mosaic/activate_dtk.sh
export PYTHONPATH=/workspace/MolCraft/src
python scripts/run_parallel_denovo.py \
  --root /workspace/MolCraft \
  --gba-config outputs/denovo-dual-gpu-20261105/gba/input/mosaic-like.yaml \
  --vdr-config outputs/denovo-dual-gpu-20261105/vdr/input/mosaic-like.yaml \
  --output outputs/denovo-dual-gpu-20261105 \
  --gpus 0 1
```

后台启动时使用独立会话并将启动器日志写到任务目录。API 密钥从既有任务环境继承，
不写入配置、Git 或日志。状态在 `parallel_state.json`，每条任务有自己的 `run.log`、
`trajectory/checkpoint.json` 和 `trajectory/runtime.json`。监督器锁和进程启动标识防止
重复启动仍存活的子进程；一条任务失败不会中断另一条。首次请求之后最多三次重试，
总计四次，退避为 1/2/4 秒，认证等确定性错误不重试。

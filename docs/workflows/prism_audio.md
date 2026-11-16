# PRISM 音效组合生成

将 [PRISM](https://arxiv.org/abs/2604.18258) 的学习式条件分解迁移到
[Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) Medium Base，
研究多音效组合中的事件遗漏和时间顺序。第一版训练目前暂停。

## 方法

冻结 T5Gemma、DiT 和 SAME codec，仅训练分解器：

```text
完整文本 → T5Gemma → K 组查询条件
同一 noisy latent、同一 timestep → 冻结 DiT 分别预测各组 velocity
组合 velocity → 更新同一条生成轨迹
```

SA3 的 rectified-flow 目标为：

```text
x_t = (1 - t) * clean_latent + t * noise
target = noise - clean_latent
loss = masked_MSE(mean(component_velocity), target) + 0.1 * orthogonality
```

- 采用双向编码器的查询分解分支，默认 4 组、每组 64 tokens、6 层、宽 512、8 头。
- 论文 Eq. (4)/(5) 写求和，公开 SD1.5/Qwen 训练代码实际取均值；这里沿用均值以保持 RF 尺度。
- query 使用固定 LayerNorm，按论文附录 A.1 避免 query 的时间自适应归一化；
  KV 和 FFN 保留 timestep 调制。公开代码快照与论文对此存在差异。
- 冻结 DiT 参数，但保留经过 DiT 向分解器传播的梯度。音频/文本编码在无梯度模式下完成。
- 时间头是本项目的实验扩展，推理只使用文本、位置与时长；训练事件标注不进入生成接口。
- 图像 CLIP/PickScore 奖励没有迁移为音频指标。组合方法是否改善实际音效仍需独立评价。

参考实现：[JY-Joy/PRISM](https://github.com/JY-Joy/PRISM)，
commit `5f72ae0e46aa4b3e5c9adf57c4e076905aa273de`；
SA3 上游 commit `790793b89ddceea612be9ce42ff6830d18f6fa57`。
保留上游源码、历史和许可证。

## 入口与配置

| 路径 | 作用 |
|---|---|
| `stable_audio_3/prism/` | 分解器、组合 RF、损失、数据处理、训练与恢复 |
| `prepare_audiotime.py` | 数据审计及清单生成 |
| `train_prism.py` | 单 GPU 训练、CPU 预检、断点恢复 |
| `generate_prism.py` | 使用已训练 adapter 生成及 baseline 配对生成 |
| `score_prism_events.py` | 对独立音频事件检测结果计算指标 |
| `tests/prism/test_prism.py` | 无 GPU、无完整权重的测试 |
| `configs/prism/audiotime.json` | 第一版核心配置，约 28.23M 参数，输出到现有 v1 目录 |
| `configs/prism/audiotime_temporal.json` | 时间监督扩展，约 29.29M 参数，尚未训练 |

基础配置沿用 `runs/prism_audiotime_v1/` 输出位置，已有检查点可继续恢复。
默认 bf16、micro batch 1、累积 8、学习率 5e-5、warmup 500、预算 10,000 optimizer steps。
仅支持单 GPU；无需外部 LLM、W&B 或在线服务。

## 环境与数据

使用 `stable-audio-3` conda 环境；依赖由 `pyproject.toml` 和 `uv.lock` 管理，
安装说明见上游 README。已验证 Python 3.10、PyTorch/Torchaudio 2.7.1+cu126 和 Flash Attention 2.8.3。

- 数据：`/mnt/disk1/datasets/AudioTime`。
- 模型：`/mnt/disk1/models/stable-audio-3-medium-base`。
- 本机清单：`data/prism_audiotime/`，保留 train/validation/test JSONL 和 summary。
- 按完整 WAV 哈希分组：18,981 train / 1,019 validation / 2,000 test，未发现相同 WAV 跨划分重复。
- 音频为 16 kHz 单声道、约 1.693–10 秒，训练整段重采样到 44.1 kHz 并复制为双声道，
  不随机裁剪；填充 6 秒 headroom，loss 只计算真实音频区域。
- 最长 caption 为 108 个 T5Gemma tokens，未超过原模型窗口；实验针对组合建模而非文本截断。
- 只有 ordering/timestamp 的起止区间用于时间 BCE；duration/frequency 标量列表不当作起止时间。
  时间扩展通过 permutation-invariant matching 匹配槽位与事件语义/活动。

原始音频、模型权重、清单、检查点和运行日志不纳入 Git。换机器后准备资源并修改本地路径，
再生成清单；本机已有清单可直接使用。

```bash
conda activate stable-audio-3
cd /home/yanzhang/stable-audio-3

# 换机器或需要重建清单时执行。
CUDA_VISIBLE_DEVICES='' python prepare_audiotime.py \
  --root /mnt/disk1/datasets/AudioTime --output data/prism_audiotime

CUDA_VISIBLE_DEVICES='' python train_prism.py --preflight
CUDA_VISIBLE_DEVICES='' python train_prism.py \
  --config configs/prism/audiotime_temporal.json --preflight
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/prism -v
```

## 训练、恢复与生成

训练仍暂停；以下命令仅供显式启动。将 0 替换为空闲 GPU，勿使用多进程 torchrun。

```bash
# 从已保存的第一版检查点恢复。
CUDA_VISIBLE_DEVICES=0 python train_prism.py \
  --resume runs/prism_audiotime_v1/checkpoint-step-009500.pt

# 从头训练时省略 --resume；时间扩展需显式指定对应配置。
CUDA_VISIBLE_DEVICES=0 python train_prism.py \
  --config configs/prism/audiotime_temporal.json

CUDA_VISIBLE_DEVICES=0 python generate_prism.py \
  --checkpoint runs/prism_audiotime_v1/checkpoint-step-009500.pt \
  --prompt "An explosion is accompanied by a sharp police siren." \
  --duration 10 --steps 50 --cfg-scale 7 --paired \
  --output-dir runs/prism_audiotime_v1/example
```

checkpoint 只保存 adapter、optimizer、配置、数据/骨干元数据指纹和 RNG。
允许修改总预算、输出位置或日志频率；架构、batch size、累积次数和训练数据必须一致。
日志为运行目录下的 `train.log`、`metrics.jsonl`。默认每 500 步保存检查点、每 250 步验证固定 64 条样本。
生成入口要求训练检查点；配对生成共享 seed、Euler schedule 和 vanilla CFG，没有复刻官方默认 APG。
`--eval-manifest data/prism_audiotime/test.jsonl --limit 16` 可生成测试子集。

## 当前保存的结果与评价边界

| 项目 | 记录 |
|---|---:|
| 最后已记录训练 step | 9,650 |
| 最新完整检查点 step | 9,500 |
| 第 9,500 步验证 flow MSE | 0.217971 |
| 原骨干在同一验证输入上的 flow MSE | 0.226542 |
| 正式训练期间峰值 allocated 显存 | 约 5.56 GiB |

第一版使用 RTX A6000 48 GB，已验证完整骨干 GPU 前向、反向和检查点恢复。
全部历史检查点与原始训练记录保留在 `runs/prism_audiotime_v1/`；
9,500 步之后未保存的更新无法据此恢复。原后台停止流程曾留下子进程句柄，旧监控已移除。

需要独立音频检测/grounding 结果或人工标注来评价实际效果，不能用 flow MSE 或时间门控替代。
尚未完成事件召回、顺序准确率或主观音质对比。

检测结果 JSONL 每条为：

```json
{"uid":"test500_timestamp:syn_1","events":[{"name":"Jackhammer","start":7.6,"end":9.9}]}
```

类别名称需映射至 AudioTime 词表；UID 必须与评价清单完全一致，无事件时写空列表。
生成子集应使用对应的子集清单。一次只评分一个 mode：

```bash
python score_prism_events.py --manifest data/prism_audiotime/test.jsonl \
  --detections detections-prism.jsonl \
  --output runs/prism_audiotime_v1/event-metrics.json
```

输出事件类别召回、全部事件出现率、额外事件数、duration/frequency MAE、
ordering pair error 和 timestamp segment IoU F1；这些是诊断指标，不是官方 STEAM 复现。

# VAE RGB 微调说明

对 Wan2.1 视频 VAE 进行微调，使其更好地重建 HM3D 室内 ERP 全景图。

## 背景

Wan2.1 的 VAE 原本在自然透视视角视频上训练，对 ERP 全景图重建存在以下问题：

- **方形灯具变圆**：VAE 对高频直角细节的表达能力不足
- **两极区域质量差**：ERP 图像两极像素被拉伸，标准 VAE 不感知这一特性
- **Depth 统计差异**：深度图的值域分布与 RGB 完全不同（另有单独脚本）

## 数据格式

```
data_root/
  {area}/
    {room}/
      output/
        panorama_0000.png   ← GT 完整全景图（仅读这些）
        panorama_0001.png
        ...
```

每个房间采样一个 **4n+1 帧的 clip**：
- 保留首帧 + 尾帧
- 中间帧随机采样
- 不足 5 帧的房间丢弃

输入到 VAE 的格式：`[B, 3, 4n+1, H, W]`，与 Diffusion 训练保持一致，确保时序卷积被正确激活。

## Loss 设计

```
Total = λ_l1 × L1
      + λ_edge × EdgeLoss(Sobel)
      + lpips_weight(t) × LPIPS
      + kl_weight(t) × KL
      + λ_prox × ProxReg
```

| Loss | 作用 |
|------|------|
| **L1** | 基础像素重建 |
| **Edge (Sobel)** | 保留直角等高频边缘，直接针对方灯变圆问题 |
| **LPIPS (VGG)** | 感知质量，保纹理结构 |
| **KL** | 保持 latent 空间正则化 |
| **ProxReg** | 向预训练权重靠拢，防止灾难性遗忘 |

所有空间 loss 乘以 **ERP 纬度余弦权重**（两极降权，赤道区域权重最高）。

## 稳定性 Tricks

| Trick | 说明 |
|-------|------|
| **EMA** | 维护影子权重（存 CPU），保存时使用 EMA 权重 |
| **KL Annealing** | KL weight 从 step 500 开始线性增加，防止 posterior collapse |
| **LPIPS 预热** | LPIPS weight 从 step 200 开始线性增加，避免早期大梯度 |
| **Loss Spike 检测** | loss > 5× 滑动均值时跳过该 batch |
| **Encoder 低 LR** | Encoder LR = Decoder LR × 0.1，保护 latent space |
| **近端正则** | `‖θ - θ₀‖²` 惩罚，防止偏离预训练权重过远 |
| **GroupNorm fp32** | 归一化层强制 fp32，提升数值稳定性 |
| **梯度检查点** | 用重计算换显存，省约 40% 激活值内存 |

### 预热时间线（默认参数）

```
Step 0   → 200  : LR warmup，仅 L1 + Edge，LPIPS=0，KL=0
Step 200 → 500  : LPIPS weight 线性增加到 0.1
Step 500 → 1000 : KL weight 线性增加到 1e-6
Step 1000→ 5000 : 全部 loss 生效，LR cosine decay
```

## 快速开始

```bash
bash run_finetune_vae_rgb.sh
```

或手动运行：

```bash
python finetune_vae_rgb.py \
  --vae_path checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth \
  --data_root /root/autodl-tmp/Matrix-3D/data/dataset_train_round1 \
  --output_dir ./results/vae_rgb_finetuned \
  --height 128 --width 256 \
  --batch_size 1 --max_frames 5 \
  --lr 1e-5 --max_steps 5000 \
  --gradient_checkpointing
```

### 断点续训

```bash
python finetune_vae_rgb.py \
  ... \
  --resume_from ./results/vae_rgb_finetuned/vae_rgb_step1000.pt
```

### 只微调 Decoder（保持 latent space 不变）

适合已有 Diffusion checkpoint，不想重新训的情况：

```bash
python finetune_vae_rgb.py \
  ... \
  --decoder_only
```

## 参数说明

### 路径

| 参数 | 说明 |
|------|------|
| `--vae_path` | Wan2.1_VAE.pth 路径 |
| `--data_root` | HM3D 数据根目录 |
| `--output_dir` | checkpoint 保存目录 |

### 数据

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--height` | 128 | 图像高度 |
| `--width` | 256 | 图像宽度 |
| `--max_frames` | 5 | clip 最大帧数，超过则截断到最近的 4n+1 |

### 训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 1 | batch 大小 |
| `--lr` | 1e-5 | 最大学习率（decoder） |
| `--max_steps` | 5000 | 总训练步数 |
| `--warmup_steps` | 200 | LR warmup 步数 |
| `--max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `--mixed_precision` | bf16 | 混合精度（bf16/fp16/no） |
| `--gradient_checkpointing` | False | 开启梯度检查点 |

### Loss 权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_l1` | 1.0 | L1 loss 权重 |
| `--lambda_edge` | 0.5 | Sobel edge loss 权重 |
| `--lambda_lpips` | 0.1 | LPIPS 最终权重 |
| `--lambda_kl` | 1e-6 | KL 最终权重 |
| `--lambda_prox` | 1e-4 | 近端正则权重 |

### 预热控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lpips_warmup_start` | 200 | LPIPS 开始增加的 step |
| `--lpips_warmup_end` | 500 | LPIPS 达到最终值的 step |
| `--kl_warmup_start` | 500 | KL 开始增加的 step |
| `--kl_warmup_end` | 1000 | KL 达到最终值的 step |

### 稳定性

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ema_decay` | 0.9999 | EMA 衰减系数 |
| `--encoder_lr_ratio` | 0.1 | Encoder LR = lr × ratio |
| `--spike_threshold` | 5.0 | Loss spike 检测阈值 |

## Checkpoint 说明

每隔 `save_steps` 步保存两个文件：

```
vae_rgb_step{N}.pt       ← 完整训练状态（含优化器，可续训）
vae_rgb_ema_step{N}.pt   ← 纯 EMA 权重（推荐用于推理）
```

训练结束后额外保存：

```
vae_rgb_final_ema.pt     ← 最终 EMA 权重
```

## 使用微调后的 VAE

```python
from videox_fun.models import AutoencoderKLWan

vae = AutoencoderKLWan.from_pretrained("Wan2.1_VAE.pth")
vae.load_state_dict(torch.load("vae_rgb_final_ema.pt", map_location="cpu"))
vae.eval()
```

## 注意事项

- 微调全量 VAE（含 encoder）后，Diffusion 模型需要重新训练
- 只微调 decoder（`--decoder_only`）时，latent space 不变，已有 Diffusion checkpoint 可继续使用
- 建议先在少量数据上跑 500 步，用测试图验证重建质量后再全量训练

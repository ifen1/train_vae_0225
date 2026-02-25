"""
VAE RGB 微调脚本 v3

对 Wan VAE 进行 encoder+decoder 全量微调（或 decoder-only）
针对 HM3D ERP 全景 RGB 视频片段。

数据格式: [B, 3, 4n+1, H, W]
  - 每个房间采样一个 clip：保留首帧 + 尾帧，中间随机采样，共 4n+1 帧
  - 喂给视频 VAE，时序卷积被正确激活，与 Diffusion 训练一致

稳定性 Tricks:
  1. EMA (Exponential Moving Average)
  2. KL Annealing（防止 posterior collapse）
  3. Loss 异常检测（跳过爆炸 batch）
  4. Encoder 使用更低 LR（1/10）
  5. 权重近端正则（防止灾难性遗忘）
  6. GroupNorm / LayerNorm 保持 fp32

"""
import argparse
import gc
import glob
import os
import random
import sys
from collections import deque
from contextlib import nullcontext, contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from videox_fun.models import AutoencoderKLWan
except ImportError:
    print("Error: videox_fun not found")
    sys.exit(1)

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS loss will be skipped.")


# ============================================================
# 帧采样工具（与 real_dataset_dual.py 保持一致）
# ============================================================

def get_target_frame_count(total_frames):
    """返回最近的 4n+1 目标帧数，< 5 帧返回 -1"""
    if total_frames < 5:
        return -1
    n = (total_frames - 1) // 4
    target = 4 * n + 1
    return target if target >= 5 else -1


def select_frames(total_frames, target_frames):
    """保留首帧 + 尾帧，中间随机采样到 target_frames"""
    if total_frames == target_frames:
        return list(range(total_frames))
    middle_pool = list(range(1, total_frames - 1))
    num_middle  = target_frames - 2
    selected    = sorted(random.sample(middle_pool, num_middle))
    return [0] + selected + [total_frames - 1]


# ============================================================
# Dataset
# ============================================================

class RGBPanoramaDataset(Dataset):
    """
    HM3D GT RGB 视频片段数据集

    每个样本是一个房间的 clip：
      - 保留首帧 + 尾帧，中间随机采样，共 4n+1 帧
      - max_frames: 帧数上限，防止长序列 OOM（默认 5）
      - 返回 [3, F, H, W]，值域 [-1, 1]

    不同房间帧数不同 → 由 collate_fn_rgb 做 padding
    """
    def __init__(self, data_root, height=128, width=256, max_frames=5):
        self.height     = height
        self.width      = width
        self.max_frames = max_frames
        self.samples    = []

        for dirpath, dirnames, _ in os.walk(data_root):
            if 'output' not in dirnames:
                continue
            output_dir = os.path.join(dirpath, 'output')
            panos = sorted(glob.glob(os.path.join(output_dir, 'panorama_*.png')))
            total = len(panos)
            target = get_target_frame_count(total)
            if target == -1:
                continue
            # 超过 max_frames 时截断到最近的 4n+1
            if target > max_frames:
                target = get_target_frame_count(max_frames)
            if target == -1:
                target = 5
            self.samples.append({
                'files':         panos,
                'total_frames':  total,
                'target_frames': target,
            })

        print(f"[Dataset] Found {len(self.samples)} valid rooms "
              f"(max_frames={max_frames}, < 5 frames discarded)")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid rooms found under {data_root}")

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        """加载单张图片，返回 [3, H, W] float32 [-1, 1]"""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
        return arr.transpose(2, 0, 1)  # [3, H, W]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_indices = select_frames(sample['total_frames'], sample['target_frames'])

        frames = np.stack(
            [self._load(sample['files'][i]) for i in frame_indices],
            axis=1   # → [3, F, H, W]
        )
        return torch.from_numpy(frames)  # [3, F, H, W]


def collate_fn_rgb(batch):
    """
    batch: list of [3, F_i, H, W]，各样本帧数可能不同
    统一 padding 到 max_F，返回 [B, 3, max_F, H, W]
    """
    max_F = max(item.shape[1] for item in batch)
    padded = []
    for item in batch:
        F = item.shape[1]
        if F < max_F:
            pad = torch.zeros(3, max_F - F, item.shape[2], item.shape[3],
                              dtype=item.dtype)
            item = torch.cat([item, pad], dim=1)
        padded.append(item)
    return torch.stack(padded)  # [B, 3, max_F, H, W]


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # 存在 CPU，不占 GPU 显存
        self.shadow = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                shadow = self.shadow[name].to(param.device)
                shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                self.shadow[name] = shadow.cpu()  # 更新完立刻移回 CPU

    @contextmanager
    def average_parameters(self, model):
        store = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                store[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if name in store:
                    param.data.copy_(store[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


# ============================================================
# 稳定性工具
# ============================================================

def fix_norm_precision(model):
    """Trick 6: GroupNorm / LayerNorm 保持 fp32"""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            module.float()
            count += 1
    print(f"[Precision] {count} norm layers set to fp32")


def make_pretrained_snapshot(model):
    """Trick 5: 保存预训练权重快照（CPU）"""
    return {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compute_prox_loss(model, snapshot, device, dtype):
    """Trick 5: 近端正则 ||theta - theta_0||^2"""
    loss = torch.tensor(0.0, device=device, dtype=dtype)
    for name, param in model.named_parameters():
        if param.requires_grad and name in snapshot:
            ref  = snapshot[name].to(device=device, dtype=dtype)
            loss = loss + (param - ref).pow(2).mean()
    return loss


def get_kl_weight(step, warmup_start, warmup_end, kl_max):
    """Trick 2: KL Annealing"""
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return kl_max
    return kl_max * (step - warmup_start) / max(1, warmup_end - warmup_start)


def get_lpips_weight(step, warmup_start, warmup_end, lpips_max):
    """Trick 7: LPIPS 权重预热，避免早期大梯度破坏重建"""
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return lpips_max
    return lpips_max * (step - warmup_start) / max(1, warmup_end - warmup_start)


# ============================================================
# ERP 权重 & Loss（5D 视频版）
# ============================================================

def make_erp_weights(H, W, device, dtype):
    """
    ERP 纬度余弦权重，形状 [1, 1, 1, H, 1]，广播到 [B, C, F, H, W]
    """
    v = torch.arange(H, device=device, dtype=torch.float32)
    theta = (0.5 - (v + 0.5) / H) * torch.pi
    w = torch.cos(theta)
    w = w / w.mean()
    return w.view(1, 1, 1, H, 1).to(dtype=dtype)


def sobel_edge_map(x):
    """
    x: [B, C, F, H, W]
    返回梯度幅值 [B, C, F, H, W]
    """
    B, C, F, H, W = x.shape
    # 合并 batch/channel/frame 维度，统一做 2D Sobel
    flat = x.permute(0, 2, 1, 3, 4).reshape(B * F * C, 1, H, W)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    mag = torch.sqrt(F_.conv2d(flat, kx, padding=1) ** 2
                     + F_.conv2d(flat, ky, padding=1) ** 2 + 1e-8)

    # 还原回 [B, C, F, H, W]
    return mag.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)


# F_ 是 torch.nn.functional 的别名，避免和帧数变量 F 冲突
F_ = F


def compute_losses(pred, target, erp_weights, lpips_fn,
                   posterior, kl_weight, lpips_weight, args):
    """
    pred / target: [B, C, F, H, W]，值域 [-1, 1]
    lpips_weight:  当前 LPIPS 实际权重（预热中可能 < lambda_lpips）
    """
    # L1（纬度加权）
    loss_l1 = ((pred - target).abs() * erp_weights).mean()

    # Sobel Edge（纬度加权）
    loss_edge = ((sobel_edge_map(pred) - sobel_edge_map(target)).abs()
                 * erp_weights).mean()

    # LPIPS：reshape 成 [B*F, C, H, W] 再计算（预热期 lpips_weight=0 时跳过省显存）
    if lpips_fn is not None and lpips_weight > 0:
        B, C, Fm, H, W = pred.shape
        pred_2d   = pred.permute(0, 2, 1, 3, 4).reshape(B * Fm, C, H, W).float()
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * Fm, C, H, W).float()
        loss_lpips = lpips_fn(pred_2d, target_2d).mean()
    else:
        loss_lpips = pred.new_zeros(1).squeeze()

    # KL（Annealing 后才生效）
    if kl_weight > 0 and posterior is not None:
        loss_kl = posterior.kl().mean()
    else:
        loss_kl = pred.new_zeros(1).squeeze()

    total = (args.lambda_l1 * loss_l1
           + args.lambda_edge * loss_edge
           + lpips_weight     * loss_lpips
           + kl_weight        * loss_kl)

    return total, {
        'l1':    loss_l1.item(),
        'edge':  loss_edge.item(),
        'lpips': loss_lpips.item(),
        'kl':    loss_kl.item(),
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(vae, ema, optimizer, scheduler, step, output_dir):
    path = os.path.join(output_dir, f"vae_rgb_step{step}.pt")
    torch.save({
        'vae':       vae.state_dict(),
        'ema':       ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step':      step,
    }, path)

    ema_path = os.path.join(output_dir, f"vae_rgb_ema_step{step}.pt")
    with ema.average_parameters(vae):
        torch.save(vae.state_dict(), ema_path)

    print(f"[Step {step}] Saved: {path}  |  EMA: {ema_path}", flush=True)


def load_checkpoint(path, vae, ema, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu')
    vae.load_state_dict(ckpt['vae'])
    if 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    step = ckpt['step']
    print(f"Resumed from step {step}: {path}")
    return step


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = (torch.bfloat16 if args.mixed_precision == 'bf16' else
                    torch.float16  if args.mixed_precision == 'fp16' else
                    torch.float32)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- VAE ----
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(args.vae_path)
    vae.to(device, dtype=weight_dtype)

    if args.decoder_only:
        print("Mode: decoder-only (encoder frozen)")
        vae.encoder.requires_grad_(False)
        if hasattr(vae, 'quant_conv'):
            vae.quant_conv.requires_grad_(False)
    else:
        print("Mode: full VAE (encoder + decoder)")

    vae.train()
    fix_norm_precision(vae)  # Trick 6

    # Trick 8: 梯度检查点（用重计算换显存，省约 40% 激活值内存）
    if args.gradient_checkpointing:
        if hasattr(vae, 'enable_gradient_checkpointing'):
            vae.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled.")
        else:
            print("Warning: VAE does not support enable_gradient_checkpointing(), skipping.")

    trainable_params_all = [p for p in vae.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params_all) / 1e6:.1f}M")

    pretrained_snapshot = make_pretrained_snapshot(vae)  # Trick 5
    ema = EMA(vae, decay=args.ema_decay)                 # Trick 1
    print(f"EMA initialized (decay={args.ema_decay})")

    # ---- LPIPS ----
    lpips_fn = None
    if LPIPS_AVAILABLE and args.lambda_lpips > 0:
        lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
        lpips_fn.requires_grad_(False)
        lpips_fn.eval()
        print("LPIPS (VGG) loaded.")

    # ---- ERP 权重（5D：[1,1,1,H,1]） ----
    erp_weights = make_erp_weights(args.height, args.width, device, weight_dtype)

    # ---- 数据集 ----
    dataset = RGBPanoramaDataset(args.data_root, args.height, args.width,
                                 max_frames=args.max_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_rgb,   # 处理不同帧数的 padding
    )

    # ---- 优化器（Trick 4: encoder 低 LR） ----
    if args.decoder_only:
        param_groups = [{'params': trainable_params_all, 'lr': args.lr}]
    else:
        encoder_params = [p for n, p in vae.named_parameters()
                          if p.requires_grad and 'encoder' in n]
        other_params   = [p for n, p in vae.named_parameters()
                          if p.requires_grad and 'encoder' not in n]
        param_groups = [
            {'params': encoder_params, 'lr': args.lr * args.encoder_lr_ratio, 'name': 'encoder'},
            {'params': other_params,   'lr': args.lr,                          'name': 'decoder'},
        ]
        print(f"Encoder LR: {args.lr * args.encoder_lr_ratio:.2e}  "
              f"Decoder LR: {args.lr:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler() if args.mixed_precision == 'fp16' else None
    amp_ctx   = (torch.amp.autocast('cuda', dtype=weight_dtype)
                 if args.mixed_precision != 'no' else nullcontext())

    # ---- 断点续训 ----
    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(
            args.resume_from, vae, ema, optimizer, scheduler
        )

    # ---- Trick 3: 滑动窗口 loss 历史 ----
    loss_history = deque(maxlen=50)
    log_acc  = {'l1': 0., 'edge': 0., 'lpips': 0., 'kl': 0.,
                'prox': 0., 'total': 0.}
    skipped  = 0

    # ---- 训练循环 ----
    print("Starting training...")
    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            # batch: [B, 3, F, H, W]（collate_fn 已处理 padding）
            x = batch.to(device, dtype=weight_dtype)

            optimizer.zero_grad()

            with amp_ctx:
                posterior  = vae.encode(x).latent_dist
                z          = posterior.sample()
                decoded    = vae.decode(z).sample.clamp(-1, 1)  # [B, 3, F, H, W]

                kl_weight = get_kl_weight(
                    global_step,
                    args.kl_warmup_start, args.kl_warmup_end, args.lambda_kl
                )
                lpips_weight = get_lpips_weight(
                    global_step,
                    args.lpips_warmup_start, args.lpips_warmup_end, args.lambda_lpips
                )
                recon_loss, components = compute_losses(
                    decoded, x, erp_weights, lpips_fn,
                    posterior, kl_weight, lpips_weight, args
                )

                # Trick 5: 近端正则
                prox_loss = compute_prox_loss(
                    vae, pretrained_snapshot, device, weight_dtype
                )
                loss = recon_loss + args.lambda_prox * prox_loss

            # Trick 3: Spike 检测
            current_loss = loss.item()
            if len(loss_history) >= 20:
                avg = sum(loss_history) / len(loss_history)
                if current_loss > args.spike_threshold * avg:
                    skipped += 1
                    if skipped % 10 == 1:
                        print(f"[Step {global_step}] Spike: {current_loss:.4f} > "
                              f"{args.spike_threshold}x avg={avg:.4f}, "
                              f"skipped {skipped} total", flush=True)
                    optimizer.zero_grad()
                    continue

            loss_history.append(current_loss)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params_all, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params_all, args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            ema.update(vae)  # Trick 1

            global_step += 1

            log_acc['total'] += current_loss
            log_acc['prox']  += prox_loss.item()
            for k, v in components.items():
                log_acc[k] += v

            if global_step % args.log_steps == 0:
                n      = args.log_steps
                lr_enc = optimizer.param_groups[0]['lr']
                lr_dec = optimizer.param_groups[-1]['lr']
                print(
                    f"Step {global_step:5d}/{args.max_steps} | "
                    f"Total={log_acc['total']/n:.4f} | "
                    f"L1={log_acc['l1']/n:.4f} | "
                    f"Edge={log_acc['edge']/n:.4f} | "
                    f"LPIPS={log_acc['lpips']/n:.4f} | "
                    f"KL={log_acc['kl']/n:.2e}(w={kl_weight:.1e}) | "
                    f"LPIPS_w={lpips_weight:.1e} | "
                    f"Prox={log_acc['prox']/n:.2e} | "
                    f"LR enc={lr_enc:.2e} dec={lr_dec:.2e}",
                    flush=True
                )
                log_acc = {k: 0. for k in log_acc}

            if global_step % args.save_steps == 0:
                save_checkpoint(vae, ema, optimizer, scheduler,
                                global_step, args.output_dir)
                gc.collect()
                torch.cuda.empty_cache()

    # ---- 最终保存（EMA 权重） ----
    final_path = os.path.join(args.output_dir, "vae_rgb_final_ema.pt")
    with ema.average_parameters(vae):
        torch.save(vae.state_dict(), final_path)
    print(f"Training complete. Final EMA weights: {final_path}")
    if skipped > 0:
        print(f"Total spike-skipped batches: {skipped}")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae_path",   type=str, required=True)
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./vae_rgb_finetuned")
    parser.add_argument("--height",     type=int, default=128)
    parser.add_argument("--width",      type=int, default=256)

    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--max_frames",    type=int,   default=5,
                        help="每个 clip 最大帧数，超过则截断到最近的 4n+1")
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--max_steps",     type=int,   default=5000)
    parser.add_argument("--warmup_steps",  type=int,   default=200)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--lambda_l1",    type=float, default=1.0)
    parser.add_argument("--lambda_edge",  type=float, default=0.5)
    parser.add_argument("--lambda_lpips", type=float, default=0.1)
    parser.add_argument("--lambda_kl",    type=float, default=1e-6)
    parser.add_argument("--lambda_prox",  type=float, default=1e-4)

    parser.add_argument("--kl_warmup_start", type=int, default=500)
    parser.add_argument("--kl_warmup_end",   type=int, default=1000)

    parser.add_argument("--lpips_warmup_start", type=int, default=200,
                        help="LPIPS 权重从 0 开始增加的 step（默认 warmup 结束后）")
    parser.add_argument("--lpips_warmup_end",   type=int, default=500,
                        help="LPIPS 权重增加到 lambda_lpips 的 step")

    parser.add_argument("--ema_decay",        type=float, default=0.9999)
    parser.add_argument("--encoder_lr_ratio", type=float, default=0.1)
    parser.add_argument("--spike_threshold",  type=float, default=5.0)

    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--log_steps",    type=int, default=10)
    parser.add_argument("--save_steps",   type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="梯度检查点：用重计算换显存，省约 40%% 激活值内存")
    parser.add_argument("--decoder_only", action="store_true")
    parser.add_argument("--resume_from",  type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main()

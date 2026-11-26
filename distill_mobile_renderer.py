"""
Distill IMTalker 512x512 Teacher -> Lightweight 256x256 Mobile Student Renderer (~8 MB FP16)
on RTX 5090 for 25-60+ FPS sharp live talking heads on Google Pixel 10 (LiteRT GPU).

Usage:
  python litert_poc/distill_mobile_renderer.py --generate --train
"""
import argparse
import glob
import math
import os
import sys
import time
import wave

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_wav_16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width {sw} in {path}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if sr != 16000:
        n_out = int(round(len(x) * 16000.0 / sr))
        t_in = np.linspace(0.0, 1.0, len(x), endpoint=False)
        t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
        x = np.interp(t_out, t_in, x).astype(np.float32)
    return x


def load_portrait_tensor(path: str, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


# =============================================================================
# Lightweight FiLM-Conditioned CoordConv Mobile U-Net (~2.4M params = ~4.8 MB FP16)
# Uses BatchNorm2d during training (folded 100% into Conv2d weight/bias at export!)
# 100% pure LiteRT GPU ops: Conv2D, FullyConnected, ResizeBilinear, Concat, Add, Mul, LeakyReLU, Sigmoid
# =============================================================================
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0], device=w.device)
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    scale = gamma / torch.sqrt(var + bn.eps)
    w_folded = w * scale.view(-1, 1, 1, 1)
    b_folded = beta + (b - mean) * scale
    return w_folded.cpu().contiguous(), b_folded.cpu().contiguous()


class ConvBN(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)


class FiLMResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int = 192):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.film_gamma = nn.Linear(cond_dim, ch)
        self.film_beta = nn.Linear(cond_dim, ch)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.film_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_beta(cond).unsqueeze(-1).unsqueeze(-1)
        h = self.bn1(self.conv1(x))
        h = F.leaky_relu(h * (1.0 + gamma) + beta, 0.2, inplace=True)
        h = self.bn2(self.conv2(h))
        return F.leaky_relu(x + h, 0.2, inplace=True)


class MobileAvatarStudent(nn.Module):
    """
    Inputs:
      t_c:     [B, 32]          IMTalker 32-D motion token
      src_img: [B, 3, 256, 256] Avatar portrait in [0, 1]
    Output:
      frame:   [B, 3, 256, 256] Animated portrait in [0, 1]
    """
    def __init__(self, cond_dim: int = 192):
        super().__init__()
        self.cond_dim = cond_dim
        ys = torch.linspace(-1.0, 1.0, 256)
        xs = torch.linspace(-1.0, 1.0, 256)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("xy_grid", torch.stack([xx, yy], dim=0).unsqueeze(0))

        # Motion MLP
        self.m_fc1 = nn.Linear(32, 128)
        self.m_fc2 = nn.Linear(128, cond_dim)
        self.m_fc3 = nn.Linear(cond_dim, cond_dim)

        # Compact Encoder (16 -> 32 -> 64 -> 128 -> 192)
        self.enc0 = ConvBN(5, 16, stride=1)

        self.enc1_a = ConvBN(16, 32, stride=2)
        self.enc1_b = ConvBN(32, 32, stride=1)

        self.enc2_a = ConvBN(32, 64, stride=2)
        self.enc2_b = ConvBN(64, 64, stride=1)

        self.enc3_a = ConvBN(64, 128, stride=2)
        self.enc3_b = ConvBN(128, 128, stride=1)

        self.enc4_a = ConvBN(128, 192, stride=2)
        self.enc4_b = ConvBN(192, 192, stride=1)

        # Bottleneck at 16x16 (192ch)
        self.mid1 = FiLMResBlock(192, cond_dim)
        self.mid2 = FiLMResBlock(192, cond_dim)

        # Decoder with U-Net skip connections + FiLM conditioning at every resolution
        self.dec3_proj = ConvBN(192 + 128, 128, stride=1)
        self.dec3_res = FiLMResBlock(128, cond_dim)

        self.dec2_proj = ConvBN(128 + 64, 64, stride=1)
        self.dec2_res = FiLMResBlock(64, cond_dim)

        self.dec1_proj = ConvBN(64 + 32, 32, stride=1)
        self.dec1_res = FiLMResBlock(32, cond_dim)

        self.dec0_proj = ConvBN(32 + 16, 16, stride=1)
        self.dec0_film_gamma = nn.Linear(cond_dim, 16)
        self.dec0_film_beta = nn.Linear(cond_dim, 16)
        nn.init.zeros_(self.dec0_film_gamma.weight)
        nn.init.zeros_(self.dec0_film_gamma.bias)
        nn.init.zeros_(self.dec0_film_beta.weight)
        nn.init.zeros_(self.dec0_film_beta.bias)
        self.dec0_conv = ConvBN(16, 16, stride=1)

        # Clean smooth Sigmoid RGB head (zero dead-clamp gradient zone!)
        self.head = nn.Conv2d(16, 3, 3, stride=1, padding=1)

    def forward(self, t_c: torch.Tensor, src_img: torch.Tensor) -> torch.Tensor:
        B = src_img.shape[0]
        cond = F.leaky_relu(self.m_fc1(t_c), 0.2)
        cond = F.leaky_relu(self.m_fc2(cond), 0.2)
        cond = F.leaky_relu(self.m_fc3(cond), 0.2)

        xy = self.xy_grid.expand(B, -1, -1, -1)
        x0 = self.enc0(torch.cat([src_img, xy], dim=1))
        x1 = self.enc1_b(self.enc1_a(x0))
        x2 = self.enc2_b(self.enc2_a(x1))
        x3 = self.enc3_b(self.enc3_a(x2))
        x4 = self.enc4_b(self.enc4_a(x3))

        h = self.mid2(self.mid1(x4, cond), cond)

        h = F.interpolate(h, size=(32, 32), mode="bilinear", align_corners=False)
        h = self.dec3_res(self.dec3_proj(torch.cat([h, x3], dim=1)), cond)

        h = F.interpolate(h, size=(64, 64), mode="bilinear", align_corners=False)
        h = self.dec2_res(self.dec2_proj(torch.cat([h, x2], dim=1)), cond)

        h = F.interpolate(h, size=(128, 128), mode="bilinear", align_corners=False)
        h = self.dec1_res(self.dec1_proj(torch.cat([h, x1], dim=1)), cond)

        h = F.interpolate(h, size=(256, 256), mode="bilinear", align_corners=False)
        h = self.dec0_proj(torch.cat([h, x0], dim=1))
        g0 = self.dec0_film_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b0 = self.dec0_film_beta(cond).unsqueeze(-1).unsqueeze(-1)
        h = F.leaky_relu(h * (1.0 + g0) + b0, 0.2)
        h = self.dec0_conv(h)

        return torch.sigmoid(self.head(h))

    def export_folded_state_dict(self) -> dict:
        """Fold all BatchNorm2d layers into Conv2d weights/biases for zero-overhead LiteRT export."""
        out = {}
        # Linear layers + head
        for name in ("m_fc1", "m_fc2", "m_fc3", "dec0_film_gamma", "dec0_film_beta", "head"):
            mod = getattr(self, name)
            out[f"{name}.weight"] = mod.weight.detach().float().cpu().contiguous()
            out[f"{name}.bias"] = mod.bias.detach().float().cpu().contiguous()
        # ConvBN layers
        for name in ("enc0", "enc1_a", "enc1_b", "enc2_a", "enc2_b", "enc3_a", "enc3_b", "enc4_a", "enc4_b",
                     "dec3_proj", "dec2_proj", "dec1_proj", "dec0_proj", "dec0_conv"):
            mod = getattr(self, name)
            w, b = fold_conv_bn(mod.conv, mod.bn)
            out[f"{name}.weight"] = w
            out[f"{name}.bias"] = b
        # FiLMResBlocks
        for name in ("mid1", "mid2", "dec3_res", "dec2_res", "dec1_res"):
            blk = getattr(self, name)
            w1, b1 = fold_conv_bn(blk.conv1, blk.bn1)
            w2, b2 = fold_conv_bn(blk.conv2, blk.bn2)
            out[f"{name}.conv1.weight"] = w1
            out[f"{name}.conv1.bias"] = b1
            out[f"{name}.conv2.weight"] = w2
            out[f"{name}.conv2.bias"] = b2
            out[f"{name}.film_gamma.weight"] = blk.film_gamma.weight.detach().float().cpu().contiguous()
            out[f"{name}.film_gamma.bias"] = blk.film_gamma.bias.detach().float().cpu().contiguous()
            out[f"{name}.film_beta.weight"] = blk.film_beta.weight.detach().float().cpu().contiguous()
            out[f"{name}.film_beta.bias"] = blk.film_beta.bias.detach().float().cpu().contiguous()
        return out


@torch.no_grad()
def generate_teacher_dataset(out_dir: str, n_per_avatar: int = 3000):
    from imtalker_frontend_rtx import IMTalkerTorchFrontendRtx

    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    print("=== [Stage 1] Loading IMTalker RTX 5090 Teacher ===")
    frontend = IMTalkerTorchFrontendRtx(os.path.join(ROOT, "checkpoints"))
    device = "cuda"

    # Load blink delta [9, 32]
    blink_path = os.path.join(ROOT, "litert_poc", "avatars", "blink_delta.bin")
    blink_delta = torch.from_numpy(np.fromfile(blink_path, dtype=np.float32).reshape(9, 32)).to(device)

    # Gather all speech WAVs
    wav_paths = sorted(
        glob.glob(os.path.join(out_dir, "wavs", "*.wav"))
        + glob.glob(os.path.join(ROOT, "litert_poc", "web", "samples", "*.wav"))
    )
    print(f"Found {len(wav_paths)} speech WAVs: {[os.path.basename(p) for p in wav_paths]}")
    wavs = [load_wav_16k(p) for p in wav_paths]

    # Reserve last WAV (speech_3.wav) strictly for UNSEEN validation!
    val_wav = wavs[-1]
    train_wavs = wavs[:-1]

    avatars = [
        ("puck", os.path.join(ROOT, "litert_poc", "avatars", "puck", "portrait.png")),
        ("gretchen", os.path.join(ROOT, "litert_poc", "avatars", "gretchen", "portrait.png")),
    ]

    train_tokens, train_av_ids, train_frames = [], [], []
    val_tokens, val_av_ids, val_frames = [], [], []
    src_portraits = []

    for av_idx, (av_key, img_path) in enumerate(avatars):
        print(f"\n--- Generating teacher data for {av_key} ---")
        img256 = load_portrait_tensor(img_path, 256).to(device)
        img256_u8 = (
            img256.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        )
        src_portraits.append(img256_u8)

        f_r, g_r, t_r_fp32, m_r = frontend._get_or_encode_face(img256)

        # 1) Generate UNSEEN validation tokens from val_wav
        audio_t = torch.from_numpy(val_wav).float().unsqueeze(0).to(device)
        ms_val, _, _, _ = frontend._sample_motion_and_face(img256, audio_t, cfg_scale=2.6, nfe=6, motion_state={})
        val_tok = ms_val[0].float()  # [T_val, 32]
        # Take up to 120 frames per avatar for fast validation
        val_tok = val_tok[:120]

        # 2) Generate real speech training tokens across all train WAVs with varied CFG & pose scales
        real_tok_list = [t_r_fp32.expand(16, -1)]  # include neutral idle anchor frames
        for wi, w_arr in enumerate(train_wavs):
            for cfg in (2.2, 2.6):
                # Also shift pose phase via frame_offset so head sway covers full angles
                st = {"frame_offset": wi * 73 + (0 if cfg == 2.2 else 190)}
                a_t = torch.from_numpy(w_arr).float().unsqueeze(0).to(device)
                ms, _, _, _ = frontend._sample_motion_and_face(img256, a_t, cfg_scale=cfg, nfe=6, motion_state=st)
                real_tok_list.append(ms[0].float())

        real_tok = torch.cat(real_tok_list, dim=0)  # [N_real, 32]
        print(f"  Real FM speech tokens for {av_key}: {real_tok.shape[0]}")

        # 3) Augment with manifold coverage: empirical covariance sampling + pairwise mixes + extra blinks
        need_extra = max(200, n_per_avatar - real_tok.shape[0])
        mu = real_tok.mean(dim=0, keepdim=True)
        centered = real_tok - mu
        cov = (centered.T @ centered) / max(1, real_tok.shape[0] - 1)
        cov = cov + 1e-5 * torch.eye(32, device=device)
        L = torch.linalg.cholesky(cov)

        n_gauss = need_extra // 2
        n_mix = need_extra - n_gauss
        z = torch.randn(n_gauss, 32, device=device)
        gauss_tok = mu + (z @ L.T) * 1.15  # 1.15x covers slightly wider expressions/mouth shapes!

        idx_a = torch.randint(0, real_tok.shape[0], (n_mix,), device=device)
        idx_b = torch.randint(0, real_tok.shape[0], (n_mix,), device=device)
        alpha = torch.rand(n_mix, 1, device=device)
        mix_tok = alpha * real_tok[idx_a] + (1.0 - alpha) * real_tok[idx_b]

        all_tok = torch.cat([real_tok, gauss_tok, mix_tok], dim=0)[:n_per_avatar].clone()

        # Inject diverse blink strengths on 25% of synthetic frames so blinks are super sharp
        n_blink = n_per_avatar // 4
        b_idx = torch.randperm(n_per_avatar, device=device)[:n_blink]
        b_frame = torch.randint(0, 9, (n_blink,), device=device)
        b_scale = 0.6 + 1.0 * torch.rand(n_blink, 1, device=device)
        all_tok[b_idx] += blink_delta[b_frame] * b_scale

        # Helper to render batch of tokens -> 256x256 uint8 RGB [N, 256, 256, 3]
        def render_tokens_256(tokens_tensor: torch.Tensor) -> np.ndarray:
            N = tokens_tensor.shape[0]
            out_u8 = np.empty((N, 256, 256, 3), dtype=np.uint8)
            bs_max = 16
            for s in range(0, N, bs_max):
                e = min(s + bs_max, N)
                bs = e - s
                act = tokens_tensor[s:e]
                g_b = g_r.expand(bs, -1)
                ta_c = frontend.renderer.adapt(act, g_b)
                m_c = frontend.renderer.latent_token_decoder(ta_c)
                m_r_b = [m.expand(bs, -1, -1, -1) for m in m_r]
                f_r_b = [f.expand(bs, -1, -1, -1) for f in f_r]
                rgb512 = frontend.renderer.decode(m_c, m_r_b, f_r_b).float().clamp_(0.0, 1.0)
                rgb256 = F.interpolate(rgb512, size=(256, 256), mode="area")
                out_u8[s:e] = rgb256.mul(255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            return out_u8

        t0 = time.time()
        tr_imgs = render_tokens_256(all_tok)
        vl_imgs = render_tokens_256(val_tok)
        dt = time.time() - t0
        print(f"  Rendered {len(tr_imgs)} train + {len(vl_imgs)} val frames for {av_key} in {dt:.1f}s ({(len(tr_imgs)+len(vl_imgs))/dt:.1f} FPS on RTX 5090)")

        train_tokens.append(all_tok.cpu().numpy())
        train_av_ids.append(np.full(len(tr_imgs), av_idx, dtype=np.int64))
        train_frames.append(tr_imgs)

        val_tokens.append(val_tok.cpu().numpy())
        val_av_ids.append(np.full(len(vl_imgs), av_idx, dtype=np.int64))
        val_frames.append(vl_imgs)

    ds_path = os.path.join(out_dir, "dataset.pt")
    torch.save(
        {
            "src_portraits": np.stack(src_portraits, axis=0),  # [2, 256, 256, 3] uint8
            "train_tokens": np.concatenate(train_tokens, axis=0),
            "train_av_ids": np.concatenate(train_av_ids, axis=0),
            "train_frames": np.concatenate(train_frames, axis=0),
            "val_tokens": np.concatenate(val_tokens, axis=0),
            "val_av_ids": np.concatenate(val_av_ids, axis=0),
            "val_frames": np.concatenate(val_frames, axis=0),
        },
        ds_path,
    )
    print(f"\nSaved teacher dataset to {ds_path} in {time.time() - t_start:.1f}s!")


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kx = kx.repeat(3, 1, 1, 1)
    ky = ky.repeat(3, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=3)
    gy = F.conv2d(x, ky, padding=1, groups=3)
    return torch.cat([gx, gy], dim=1)


def train_student(out_dir: str, epochs: int = 120, batch_size: int = 32, lr: float = 1.5e-3):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ds_path = os.path.join(out_dir, "dataset.pt")
    print(f"=== [Stage 2] Loading {ds_path} into RTX 5090 VRAM ===")
    ds = torch.load(ds_path, weights_only=False)

    # Keep entire 256x256 uint8 dataset in GPU VRAM (~1.2 GB of 32 GB!) for zero DataLoader bottleneck!
    src_portraits = torch.from_numpy(ds["src_portraits"]).to(device).permute(0, 3, 1, 2).float() / 255.0  # [2, 3, 256, 256]
    tr_tok = torch.from_numpy(ds["train_tokens"]).to(device).float()
    tr_av = torch.from_numpy(ds["train_av_ids"]).to(device).long()
    tr_u8 = torch.from_numpy(ds["train_frames"]).to(device)  # [N, 256, 256, 3] uint8

    vl_tok = torch.from_numpy(ds["val_tokens"]).to(device).float()
    vl_av = torch.from_numpy(ds["val_av_ids"]).to(device).long()
    vl_u8 = torch.from_numpy(ds["val_frames"]).to(device)

    model = MobileAvatarStudent(cond_dim=192).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileAvatarStudent params: {n_params:,} ({n_params * 2 / 1e6:.2f} MB FP16)")

    # Spatial face-feature weight map: 3.5x weight on mouth/eyes/eyebrows/jaw (y in [80..210], x in [60..196])
    roi_w = torch.ones((1, 1, 256, 256), device=device)
    roi_w[:, :, 85:135, 65:190] = 3.0   # eyes & eyebrows
    roi_w[:, :, 145:210, 75:180] = 4.0  # mouth, lips, teeth, chin

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)

    N = tr_tok.shape[0]
    best_psnr = 0.0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        loss_sum = 0.0
        steps = 0
        for s in range(0, N - batch_size + 1, batch_size):
            idx = perm[s : s + batch_size]
            tok = tr_tok[idx]
            src = src_portraits[tr_av[idx]]
            tgt = tr_u8[idx].permute(0, 3, 1, 2).float() / 255.0

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(tok, src)
                diff = torch.abs(pred - tgt)
                l_pix = (diff * roi_w).mean()
                l_edge = F.l1_loss(sobel_grad(pred), sobel_grad(tgt))
                l_ms = F.l1_loss(F.avg_pool2d(pred, 2), F.avg_pool2d(tgt, 2))
                loss = l_pix + 0.15 * l_edge + 0.5 * l_ms

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += float(loss.item())
            steps += 1
        sched.step()

        if ep == 1 or ep == 5 or ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                mse_all = 0.0
                mouth_mae = 0.0
                n_val = vl_tok.shape[0]
                for s in range(0, n_val, 32):
                    e = min(s + 32, n_val)
                    src = src_portraits[vl_av[s:e]]
                    tgt = vl_u8[s:e].permute(0, 3, 1, 2).float() / 255.0
                    pred = model(vl_tok[s:e], src)
                    mse_all += float(((pred - tgt) ** 2).sum().item())
                    mouth_mae += float(torch.abs(pred[:, :, 145:210, 75:180] - tgt[:, :, 145:210, 75:180]).sum().item())
                mse_mean = mse_all / (n_val * 3 * 256 * 256)
                psnr = 10.0 * math.log10(1.0 / max(mse_mean, 1e-12))
                m_mae_255 = (mouth_mae / (n_val * 3 * 65 * 105)) * 255.0
                elapsed = time.time() - t0
                print(
                    f"Epoch {ep:3d}/{epochs} ({elapsed:5.1f}s) | loss={loss_sum/steps:.4f} | "
                    f"UNSEEN VAL PSNR={psnr:.2f} dB | MouthMAE={m_mae_255:.2f}/255 | lr={sched.get_last_lr()[0]:.1e}"
                )
                if psnr > best_psnr:
                    best_psnr = psnr
                    st_path = os.path.join(ROOT, "litert_poc", "student_renderer_256.safetensors")
                    save_file(model.export_folded_state_dict(), st_path)

    # Save visual side-by-side comparison on unseen Puck + Gretchen validation frames!
    model.eval()
    with torch.no_grad():
        # Pick 2 Puck frames + 2 Gretchen frames from unseen val speech (wide mouth frames!)
        puck_idx = [15, 55]
        gret_idx = [120 + 15, 120 + 55]
        picks = [i for i in puck_idx + gret_idx if i < vl_tok.shape[0]]
        rows = []
        for idx in picks:
            src = src_portraits[vl_av[idx : idx + 1]]
            tgt = vl_u8[idx : idx + 1].permute(0, 3, 1, 2).float() / 255.0
            pred = model(vl_tok[idx : idx + 1], src)
            pair = torch.cat([tgt[0], pred[0]], dim=2).clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            rows.append(pair)
        grid = np.concatenate(rows, axis=0)
        cmp_path = os.path.join(out_dir, "val_teacher_vs_student.png")
        Image.fromarray(grid).save(cmp_path)
        print(f"\nBest UNSEEN VAL PSNR = {best_psnr:.2f} dB | Saved comparison: {cmp_path}")

        # Also save a golden reference pair for C++ LiteRT --verify!
        ref_dir = os.path.join(ROOT, "litert_poc", "ref_student")
        os.makedirs(ref_dir, exist_ok=True)
        idx = picks[0]
        src = src_portraits[vl_av[idx : idx + 1]]
        tok = vl_tok[idx : idx + 1]
        pred = model(tok, src)
        tok.cpu().numpy().astype(np.float32).tofile(os.path.join(ref_dir, "t_c.bin"))
        src.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32).tofile(os.path.join(ref_dir, "src_img.bin"))
        pred.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32).tofile(os.path.join(ref_dir, "frame_ref.bin"))
        print(f"Saved C++ parity golden reference in {ref_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n-per-avatar", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "litert_poc", "distill_data")
    if args.generate:
        generate_teacher_dataset(out_dir, n_per_avatar=args.n_per_avatar)
    if args.train:
        train_student(out_dir, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

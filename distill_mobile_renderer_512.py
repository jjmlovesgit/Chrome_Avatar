"""
Distill IMTalker 512x512 Teacher -> Native 512x512 HD Mobile Student v2 (~10 MB FP16)
on RTX 5090 for ~28-35 FPS sharp native-HD live talking heads on Google Pixel 10
and ~100+ FPS in Desktop Chrome WebGPU.

Key v2 upgrades over v1 (256x256, 6.74 MB, pure L1):
  1. Native 512x512 output via DepthToSpace(2) (PixelShuffle) + gated src_img_512 skip
     -> 4x pixel resolution, 100% razor-sharp static hair/background/identity!
  2. PixelShuffle(2) decoder upsampling at 64->128->256->512 (zero bilinear blur!)
  3. Local IMTalker VGG19 multi-layer perceptual loss (full-frame + full-res 512px face crop)
  4. Gentle Spectral-Norm PatchGAN sharpness loss (after epoch 25 warmup)
  5. 10,000 diverse native 512x512 teacher frames + EMA shadow weights

Usage:
  python litert_poc/distill_mobile_renderer_512.py --generate --train
"""
import argparse
import copy
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
from safetensors import safe_open
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
# Local IMTalker VGG19 Perceptual Loss (loaded directly from renderer.safetensors!)
# =============================================================================
class LocalVgg19Perceptual(nn.Module):
    def __init__(self, safetensors_path: str):
        super().__init__()
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.slice2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.slice3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.slice4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Map exact keys from IMTalker's criterion_vgg.vgg.*
        mapping = [
            ("criterion_vgg.vgg.slice1.0", self.slice1[0]),
            ("criterion_vgg.vgg.slice2.2", self.slice2[0]),
            ("criterion_vgg.vgg.slice2.5", self.slice2[3]),
            ("criterion_vgg.vgg.slice3.7", self.slice3[0]),
            ("criterion_vgg.vgg.slice3.10", self.slice3[3]),
            ("criterion_vgg.vgg.slice4.12", self.slice4[0]),
            ("criterion_vgg.vgg.slice4.14", self.slice4[2]),
            ("criterion_vgg.vgg.slice4.16", self.slice4[4]),
            ("criterion_vgg.vgg.slice4.19", self.slice4[7]),
        ]
        with safe_open(safetensors_path, framework="pt") as f:
            mean = f.get_tensor("criterion_vgg.vgg.mean").float()
            std = f.get_tensor("criterion_vgg.vgg.std").float()
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            for k_pfx, mod in mapping:
                mod.weight.data.copy_(f.get_tensor(f"{k_pfx}.weight").float())
                mod.bias.data.copy_(f.get_tensor(f"{k_pfx}.bias").float())

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = 0.0
        weights = (1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0)
        for sl, w in zip((self.slice1, self.slice2, self.slice3, self.slice4), weights):
            x = sl(x)
            with torch.no_grad():
                y = sl(y)
            loss = loss + w * F.l1_loss(x, y)
        return loss


# =============================================================================
# Lightweight Spectral-Norm PatchGAN Discriminator (Training Only, 0 ms inference!)
# =============================================================================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_c: int = 3):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.b1 = nn.Sequential(sn(nn.Conv2d(in_c, 48, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True))
        self.b2 = nn.Sequential(sn(nn.Conv2d(48, 96, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True))
        self.b3 = nn.Sequential(sn(nn.Conv2d(96, 192, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True))
        self.b4 = nn.Sequential(sn(nn.Conv2d(192, 256, 4, stride=1, padding=1)), nn.LeakyReLU(0.2, inplace=True))
        self.out = sn(nn.Conv2d(256, 1, 4, stride=1, padding=1))

    def forward(self, x: torch.Tensor):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        logits = self.out(f4)
        return logits, (f1, f2, f3, f4)


# =============================================================================
# Native 512x512 HD Mobile Student v2 (~5.0M params = ~10 MB FP16)
# 100% pure LiteRT GPU ops:
#   AveragePool2D, Conv2D, Pad, DepthToSpace(2), ResizeBilinear, Concat,
#   FullyConnected, Reshape, Add, Mul, Slice, LeakyReLU, Logistic
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
    def __init__(self, ch: int, cond_dim: int = 224):
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


class MobileAvatarStudent512(nn.Module):
    """
    Inputs:
      t_c:     [B, 32]          IMTalker 32-D motion token
      src_img: [B, 3, 512, 512] Native 512x512 Avatar portrait in [0, 1]
    Output:
      frame:   [B, 3, 512, 512] Native 512x512 Animated HD portrait in [0, 1]
    """
    def __init__(self, cond_dim: int = 224):
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

        # Compact 256->16 Encoder (24 -> 48 -> 80 -> 144 -> 224)
        self.enc0 = ConvBN(5, 24, stride=1)

        self.enc1_a = ConvBN(24, 48, stride=2)
        self.enc1_b = ConvBN(48, 48, stride=1)

        self.enc2_a = ConvBN(48, 80, stride=2)
        self.enc2_b = ConvBN(80, 80, stride=1)

        self.enc3_a = ConvBN(80, 144, stride=2)
        self.enc3_b = ConvBN(144, 144, stride=1)

        self.enc4_a = ConvBN(144, 224, stride=2)
        self.enc4_b = ConvBN(224, 224, stride=1)

        # Bottleneck at 16x16 (3 FiLMResBlocks — <1 ms extra on GPU, big pose/expression boost!)
        self.mid1 = FiLMResBlock(224, cond_dim)
        self.mid2 = FiLMResBlock(224, cond_dim)
        self.mid3 = FiLMResBlock(224, cond_dim)

        # Decoder 16 -> 32 -> 64 (bilinear) -> 128 -> 256 (sharp PixelShuffle!)
        self.dec3_proj = ConvBN(224 + 144, 144, stride=1)
        self.dec3_res = FiLMResBlock(144, cond_dim)

        self.dec2_proj = ConvBN(144 + 80, 80, stride=1)
        self.dec2_res = FiLMResBlock(80, cond_dim)

        self.up1_conv = ConvBN(80, 48 * 4, stride=1)  # PixelShuffle(2) -> 48ch @ 128x128
        self.dec1_proj = ConvBN(48 + 48, 48, stride=1)
        self.dec1_res = FiLMResBlock(48, cond_dim)

        self.up0_conv = ConvBN(48, 24 * 4, stride=1)  # PixelShuffle(2) -> 24ch @ 256x256
        self.dec0_proj = ConvBN(24 + 24, 24, stride=1)
        self.dec0_res = FiLMResBlock(24, cond_dim)

        # Native 512x512 HD Refinement + Gated src_img_512 Skip Head
        self.up512_conv = ConvBN(24, 8 * 4, stride=1)  # PixelShuffle(2) -> 8ch @ 512x512
        self.hd_conv1 = ConvBN(8 + 3, 16, stride=1)
        self.hd_film_gamma = nn.Linear(cond_dim, 16)
        self.hd_film_beta = nn.Linear(cond_dim, 16)
        nn.init.zeros_(self.hd_film_gamma.weight)
        nn.init.zeros_(self.hd_film_gamma.bias)
        nn.init.zeros_(self.hd_film_beta.weight)
        nn.init.zeros_(self.hd_film_beta.bias)
        self.hd_conv2 = ConvBN(16, 16, stride=1)

        # 4-channel head: [0:3] = predicted RGB logits, [3:4] = blend alpha logits
        self.head = nn.Conv2d(16, 4, 3, stride=1, padding=1)
        nn.init.zeros_(self.head.bias)

    def forward(self, t_c: torch.Tensor, src_img: torch.Tensor) -> torch.Tensor:
        B = src_img.shape[0]
        cond = F.leaky_relu(self.m_fc1(t_c), 0.2)
        cond = F.leaky_relu(self.m_fc2(cond), 0.2)
        cond = F.leaky_relu(self.m_fc3(cond), 0.2)

        src_256 = F.avg_pool2d(src_img, kernel_size=2, stride=2)
        xy = self.xy_grid.expand(B, -1, -1, -1)
        x0 = self.enc0(torch.cat([src_256, xy], dim=1))
        x1 = self.enc1_b(self.enc1_a(x0))
        x2 = self.enc2_b(self.enc2_a(x1))
        x3 = self.enc3_b(self.enc3_a(x2))
        x4 = self.enc4_b(self.enc4_a(x3))

        h = self.mid3(self.mid2(self.mid1(x4, cond), cond), cond)

        h = F.interpolate(h, size=(32, 32), mode="bilinear", align_corners=False)
        h = self.dec3_res(self.dec3_proj(torch.cat([h, x3], dim=1)), cond)

        h = F.interpolate(h, size=(64, 64), mode="bilinear", align_corners=False)
        h = self.dec2_res(self.dec2_proj(torch.cat([h, x2], dim=1)), cond)

        h = F.pixel_shuffle(self.up1_conv(h), 2)
        h = self.dec1_res(self.dec1_proj(torch.cat([h, x1], dim=1)), cond)

        h = F.pixel_shuffle(self.up0_conv(h), 2)
        h = self.dec0_res(self.dec0_proj(torch.cat([h, x0], dim=1)), cond)

        # Native 512x512 HD refinement fused with full 512x512 source portrait
        h512 = F.pixel_shuffle(self.up512_conv(h), 2)  # [B, 8, 512, 512]
        h512 = self.hd_conv1(torch.cat([h512, src_img], dim=1))
        g_hd = self.hd_film_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b_hd = self.hd_film_beta(cond).unsqueeze(-1).unsqueeze(-1)
        h512 = F.leaky_relu(h512 * (1.0 + g_hd) + b_hd, 0.2)
        h512 = self.hd_conv2(h512)

        logits = self.head(h512)
        rgb_pred = torch.sigmoid(logits[:, :3])
        alpha = torch.sigmoid(logits[:, 3:4])
        return alpha * rgb_pred + (1.0 - alpha) * src_img

    def export_folded_state_dict(self) -> dict:
        """Fold all BatchNorm2d layers into Conv2d weights/biases for zero-overhead LiteRT export."""
        out = {}
        for name in ("m_fc1", "m_fc2", "m_fc3", "hd_film_gamma", "hd_film_beta", "head"):
            mod = getattr(self, name)
            out[f"{name}.weight"] = mod.weight.detach().float().cpu().contiguous()
            out[f"{name}.bias"] = mod.bias.detach().float().cpu().contiguous()
        for name in (
            "enc0", "enc1_a", "enc1_b", "enc2_a", "enc2_b", "enc3_a", "enc3_b", "enc4_a", "enc4_b",
            "dec3_proj", "dec2_proj", "up1_conv", "dec1_proj", "up0_conv", "dec0_proj",
            "up512_conv", "hd_conv1", "hd_conv2",
        ):
            mod = getattr(self, name)
            w, b = fold_conv_bn(mod.conv, mod.bn)
            out[f"{name}.weight"] = w
            out[f"{name}.bias"] = b
        for name in ("mid1", "mid2", "mid3", "dec3_res", "dec2_res", "dec1_res", "dec0_res"):
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
def generate_teacher_dataset_512(out_dir: str, n_per_avatar: int = 5000):
    from imtalker_frontend_rtx import IMTalkerTorchFrontendRtx

    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    print("=== [Stage 1] Loading IMTalker RTX 5090 512x512 Teacher ===")
    frontend = IMTalkerTorchFrontendRtx(os.path.join(ROOT, "checkpoints"))
    device = "cuda"

    blink_path = os.path.join(ROOT, "litert_poc", "avatars", "blink_delta.bin")
    blink_delta = torch.from_numpy(np.fromfile(blink_path, dtype=np.float32).reshape(9, 32)).to(device)

    wav_paths = sorted(
        glob.glob(os.path.join(ROOT, "litert_poc", "distill_data", "wavs", "*.wav"))
        + glob.glob(os.path.join(ROOT, "litert_poc", "web", "samples", "*.wav"))
    )
    print(f"Found {len(wav_paths)} speech WAVs: {[os.path.basename(p) for p in wav_paths]}")
    wavs = [load_wav_16k(p) for p in wav_paths]

    # Reserve speech_3.wav (or last WAV) strictly for UNSEEN validation (apples-to-apples with v1!)
    val_idx = next((i for i, p in enumerate(wav_paths) if "speech_3" in os.path.basename(p)), len(wavs) - 1)
    val_wav = wavs[val_idx]
    train_wavs = [w for i, w in enumerate(wavs) if i != val_idx]
    print(f"Unseen val WAV: {os.path.basename(wav_paths[val_idx])} | Train WAVs: {len(train_wavs)}")

    avatars = [
        ("puck", os.path.join(ROOT, "litert_poc", "avatars", "puck", "portrait.png")),
        ("gretchen", os.path.join(ROOT, "litert_poc", "avatars", "gretchen", "portrait.png")),
    ]

    train_tokens, train_av_ids, train_frames = [], [], []
    val_tokens, val_av_ids, val_frames = [], [], []
    src_portraits = []

    for av_idx, (av_key, img_path) in enumerate(avatars):
        print(f"\n--- Generating native 512x512 teacher data for {av_key} ---")
        img512 = load_portrait_tensor(img_path, 512).to(device)
        img256 = load_portrait_tensor(img_path, 256).to(device)
        img512_u8 = (
            img512.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        )
        src_portraits.append(img512_u8)

        # IMTalker's dense_feature_encoder takes 256x256 input and synthesizes native 512x512 frames!
        f_r, g_r, t_r_fp32, m_r = frontend._get_or_encode_face(img256)

        # 1) Unseen validation tokens (120 frames per avatar)
        audio_t = torch.from_numpy(val_wav).float().unsqueeze(0).to(device)
        ms_val, _, _, _ = frontend._sample_motion_and_face(img256, audio_t, cfg_scale=2.6, nfe=6, motion_state={})
        val_tok = ms_val[0].float()[:120]

        # 2) Diverse real speech training tokens across CFG scales (2.0, 2.6, 3.0) & varied pose offsets
        real_tok_list = [t_r_fp32.expand(32, -1)]
        for wi, w_arr in enumerate(train_wavs):
            for ci, cfg in enumerate((2.0, 2.6, 3.0)):
                st = {"frame_offset": wi * 83 + ci * 137}
                a_t = torch.from_numpy(w_arr).float().unsqueeze(0).to(device)
                ms, _, _, _ = frontend._sample_motion_and_face(img256, a_t, cfg_scale=cfg, nfe=6, motion_state=st)
                real_tok_list.append(ms[0].float())

        real_tok = torch.cat(real_tok_list, dim=0)
        print(f"  Real FM speech tokens for {av_key}: {real_tok.shape[0]}")

        # 3) Empirical covariance manifold + pairwise mixes + expressive blinks
        need_extra = max(400, n_per_avatar - real_tok.shape[0])
        mu = real_tok.mean(dim=0, keepdim=True)
        centered = real_tok - mu
        cov = (centered.T @ centered) / max(1, real_tok.shape[0] - 1)
        cov = cov + 1e-5 * torch.eye(32, device=device)
        L = torch.linalg.cholesky(cov)

        n_gauss = need_extra // 2
        n_mix = need_extra - n_gauss
        z = torch.randn(n_gauss, 32, device=device)
        gauss_tok = mu + (z @ L.T) * 1.18

        idx_a = torch.randint(0, real_tok.shape[0], (n_mix,), device=device)
        idx_b = torch.randint(0, real_tok.shape[0], (n_mix,), device=device)
        alpha = torch.rand(n_mix, 1, device=device)
        mix_tok = alpha * real_tok[idx_a] + (1.0 - alpha) * real_tok[idx_b]

        all_tok = torch.cat([real_tok, gauss_tok, mix_tok], dim=0)[:n_per_avatar].clone()

        # Inject diverse blink strengths on 25% of frames
        n_blink = n_per_avatar // 4
        b_idx = torch.randperm(n_per_avatar, device=device)[:n_blink]
        b_frame = torch.randint(0, 9, (n_blink,), device=device)
        b_scale = 0.5 + 1.1 * torch.rand(n_blink, 1, device=device)
        all_tok[b_idx] += blink_delta[b_frame] * b_scale

        def render_tokens_512(tokens_tensor: torch.Tensor) -> np.ndarray:
            N = tokens_tensor.shape[0]
            out_u8 = np.empty((N, 512, 512, 3), dtype=np.uint8)
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
                out_u8[s:e] = rgb512.mul(255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            return out_u8

        t0 = time.time()
        tr_imgs = render_tokens_512(all_tok)
        vl_imgs = render_tokens_512(val_tok)
        dt = time.time() - t0
        print(
            f"  Rendered {len(tr_imgs)} train + {len(vl_imgs)} val 512x512 frames for {av_key} "
            f"in {dt:.1f}s ({(len(tr_imgs) + len(vl_imgs)) / dt:.1f} FPS on RTX 5090)"
        )

        train_tokens.append(all_tok.cpu().numpy())
        train_av_ids.append(np.full(len(tr_imgs), av_idx, dtype=np.int64))
        train_frames.append(tr_imgs)

        val_tokens.append(val_tok.cpu().numpy())
        val_av_ids.append(np.full(len(vl_imgs), av_idx, dtype=np.int64))
        val_frames.append(vl_imgs)

    ds_path = os.path.join(out_dir, "dataset_512.pt")
    torch.save(
        {
            "src_portraits": torch.from_numpy(np.stack(src_portraits, axis=0)),  # [2, 512, 512, 3] uint8
            "train_tokens": torch.from_numpy(np.concatenate(train_tokens, axis=0)),
            "train_av_ids": torch.from_numpy(np.concatenate(train_av_ids, axis=0)),
            "train_frames": torch.from_numpy(np.concatenate(train_frames, axis=0)),
            "val_tokens": torch.from_numpy(np.concatenate(val_tokens, axis=0)),
            "val_av_ids": torch.from_numpy(np.concatenate(val_av_ids, axis=0)),
            "val_frames": torch.from_numpy(np.concatenate(val_frames, axis=0)),
        },
        ds_path,
        pickle_protocol=4,
    )
    size_gb = os.path.getsize(ds_path) / 1e9
    print(f"\nSaved native 512x512 teacher dataset ({size_gb:.2f} GB) to {ds_path} in {time.time() - t_start:.1f}s!")


_SOBEL_CACHE = {}


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    key = (x.device, x.dtype)
    if key not in _SOBEL_CACHE:
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        _SOBEL_CACHE[key] = (kx, ky)
    kx, ky = _SOBEL_CACHE[key]
    gx = F.conv2d(x, kx, padding=1, groups=3)
    gy = F.conv2d(x, ky, padding=1, groups=3)
    return torch.cat([gx, gy], dim=1)


@torch.no_grad()
def update_ema(ema_params, model_params, ema_bufs, model_bufs, decay: float = 0.992):
    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, model_params, alpha=1.0 - decay)
    for eb, b in zip(ema_bufs, model_bufs):
        eb.copy_(b)


def train_student_512(
    out_dir: str,
    epochs: int = 160,
    batch_size: int = 24,
    lr: float = 1.2e-3,
    gan_start_epoch: int = 25,
):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ds_path = os.path.join(out_dir, "dataset_512.pt")
    print(f"=== [Stage 2] Loading {ds_path} into RTX 5090 VRAM ===")
    ds = torch.load(ds_path, weights_only=False)

    def to_dev(x):
        return (x if isinstance(x, torch.Tensor) else torch.from_numpy(x)).to(device)

    # Keep entire 512x512 uint8 dataset (~7.9 GB of 32 GB!) directly in RTX 5090 VRAM for max throughput!
    src_portraits = to_dev(ds["src_portraits"]).permute(0, 3, 1, 2).float() / 255.0
    tr_tok = to_dev(ds["train_tokens"]).float()
    tr_av = to_dev(ds["train_av_ids"]).long()
    tr_u8 = to_dev(ds["train_frames"])

    vl_tok = to_dev(ds["val_tokens"]).float()
    vl_av = to_dev(ds["val_av_ids"]).long()
    vl_u8 = to_dev(ds["val_frames"])

    model = MobileAvatarStudent512(cond_dim=224).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_params = [p.data for p in ema_model.parameters()]
    model_params = [p.data for p in model.parameters()]
    ema_bufs = [b.data for b in ema_model.buffers()]
    model_bufs = [b.data for b in model.buffers()]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileAvatarStudent512 params: {n_params:,} ({n_params * 2 / 1e6:.2f} MB FP16)")

    # Local IMTalker VGG19 perceptual loss (zero download!)
    vgg_loss_fn = LocalVgg19Perceptual(os.path.join(ROOT, "checkpoints", "renderer.safetensors")).to(device)
    disc = PatchDiscriminator(in_c=3).to(device)

    # Spatial 512x512 face-feature weight map
    roi_w = torch.ones((1, 1, 512, 512), device=device)
    roi_w[:, :, 170:270, 130:380] = 3.0   # eyes & eyebrows
    roi_w[:, :, 290:420, 150:360] = 4.5   # mouth, lips, teeth, chin

    opt_g = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, fused=True)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr * 0.4, betas=(0.5, 0.99), weight_decay=1e-4, fused=True)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=3e-5)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-5)

    N = tr_tok.shape[0]
    best_psnr = 0.0
    best_val_vgg = 1e9
    t0 = time.time()

    def crop_face_256(img512: torch.Tensor) -> torch.Tensor:
        # Full native-512-resolution 256x256 crop covering eyes, nose, lips, teeth, chin!
        return img512[:, :, 160:416, 128:384]

    for ep in range(1, epochs + 1):
        model.train()
        disc.train()
        perm = torch.randperm(N, device=device)
        loss_g_sum, loss_vgg_sum, loss_d_sum = 0.0, 0.0, 0.0
        steps = 0
        use_gan = (ep >= gan_start_epoch)
        # Smooth ramp of GAN weight over 10 epochs after warmup
        gan_w = 0.0 if not use_gan else 0.012 * min(1.0, (ep - gan_start_epoch + 1) / 10.0)

        for s in range(0, N - batch_size + 1, batch_size):
            idx = perm[s : s + batch_size]
            tok = tr_tok[idx]
            src = src_portraits[tr_av[idx]]
            tgt = tr_u8[idx].permute(0, 3, 1, 2).float() / 255.0

            # 1) Generator step
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(tok, src)
                diff = torch.abs(pred - tgt)
                l_pix = (diff * roi_w).mean()
                l_edge = F.l1_loss(sobel_grad(pred), sobel_grad(tgt))
                pred_256 = F.avg_pool2d(pred, 2)
                tgt_256 = F.avg_pool2d(tgt, 2)
                l_ms = F.l1_loss(pred_256, tgt_256)

                # Batched VGG19 perceptual loss: full-frame 256px + native-512px 256x256 face crop in 1 pass!
                pred_face = crop_face_256(pred)
                tgt_face = crop_face_256(tgt)
                l_vgg = vgg_loss_fn(torch.cat([pred_256, pred_face], dim=0), torch.cat([tgt_256, tgt_face], dim=0))

                loss_g = l_pix + 0.18 * l_edge + 0.35 * l_ms + 0.08 * l_vgg

                if use_gan:
                    fake_logits, fake_feats = disc(pred_face)
                    with torch.no_grad():
                        _, real_feats = disc(tgt_face)
                    l_adv = -fake_logits.mean()
                    l_fm = sum(F.l1_loss(ff, rf) for ff, rf in zip(fake_feats, real_feats)) / len(fake_feats)
                    loss_g = loss_g + gan_w * l_adv + 0.08 * l_fm

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt_g.step()

            # Vectorized EMA update every 2 steps (zero Python overhead!)
            if (steps & 1) == 0:
                ema_decay = min(0.992, (ep * 8.0) / (ep * 8.0 + 8.0))
                update_ema(ema_params, model_params, ema_bufs, model_bufs, decay=ema_decay)

            # 2) PatchGAN Discriminator step (only after warmup, every 2nd step to keep G dominant)
            if use_gan and (steps % 2 == 0):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    real_logits, _ = disc(tgt_face.detach())
                    fake_logits_d, _ = disc(pred_face.detach())
                    loss_d = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits_d).mean()
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 2.0)
                opt_d.step()
                loss_d_sum += float(loss_d.item())

            loss_g_sum += float(loss_g.item())
            loss_vgg_sum += float(l_vgg.item())
            steps += 1

        sched_g.step()
        if use_gan:
            sched_d.step()

        if ep == 1 or ep == 5 or ep % 10 == 0 or ep == epochs:
            eval_net = ema_model if ep >= 10 else model
            eval_net.eval()
            with torch.no_grad():
                mse_all = 0.0
                mouth_mae = 0.0
                vgg_val = 0.0
                n_val = vl_tok.shape[0]
                for s in range(0, n_val, 24):
                    e = min(s + 24, n_val)
                    p = eval_net(vl_tok[s:e], src_portraits[vl_av[s:e]])
                    t = vl_u8[s:e].permute(0, 3, 1, 2).float() / 255.0
                    mse_all += F.mse_loss(p, t, reduction="sum").item()
                    mouth_mae += torch.abs(p[:, :, 290:420, 150:360] - t[:, :, 290:420, 150:360]).mean().item() * (e - s)
                    vgg_val += vgg_loss_fn(crop_face_256(p), crop_face_256(t)).item() * (e - s)
                mse_all /= (n_val * 3 * 512 * 512)
                psnr = -10.0 * math.log10(max(1e-10, mse_all))
                mouth_mae = (mouth_mae / n_val) * 255.0
                vgg_val /= n_val
                elapsed = (time.time() - t0) / 60.0
                print(
                    f"Epoch {ep:3d}/{epochs} ({elapsed:4.1f}m) | G={loss_g_sum/steps:.4f} | "
                    f"VGG={loss_vgg_sum/steps:.4f} | D={loss_d_sum/max(1, steps//2):.3f} | "
                    f"UNSEEN 512px PSNR={psnr:.2f} dB | MouthMAE={mouth_mae:.2f}/255 | ValVGG={vgg_val:.4f}",
                    flush=True,
                )

                # Save whenever PSNR improves (warmup) or perceptual ValVGG improves (post-GAN) or final epoch
                is_best = (psnr > best_psnr) or (ep >= gan_start_epoch and vgg_val < best_val_vgg) or (ep == epochs)
                if psnr > best_psnr:
                    best_psnr = psnr
                if ep >= gan_start_epoch and vgg_val < best_val_vgg:
                    best_val_vgg = vgg_val
                if is_best:
                    st_path = os.path.join(out_dir, "student_renderer_512.safetensors")
                    save_file(eval_net.export_folded_state_dict(), st_path)
                    # Save PyTorch golden reference for exact C++ LiteRT parity check
                    ref_dir = os.path.join(out_dir, "ref_student_512")
                    os.makedirs(ref_dir, exist_ok=True)
                    p0 = eval_net(vl_tok[:1], src_portraits[vl_av[:1]])
                    vl_tok[:1].cpu().numpy().astype(np.float32).tofile(os.path.join(ref_dir, "t_c.bin"))
                    src_portraits[vl_av[:1]].permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32).tofile(
                        os.path.join(ref_dir, "src_img.bin")
                    )
                    p0.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32).tofile(
                        os.path.join(ref_dir, "frame_ref.bin")
                    )

                if ep in (25, 80, 120, epochs):
                    # Save crisp visual comparison grid (Full 512x512 + 2x Face Crop: Teacher vs Student v2)
                    sample_ids = [10, 45, 95, 130, 165, 215]
                    cols = []
                    for sid in sample_ids:
                        if sid >= n_val:
                            continue
                        p = eval_net(vl_tok[sid : sid + 1], src_portraits[vl_av[sid : sid + 1]])[0]
                        p_u8 = (p.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                        t_u8 = vl_u8[sid].cpu().numpy()
                        # Full frame (resized 256) + native 512px 256x256 face/mouth crop!
                        t_full = np.asarray(Image.fromarray(t_u8).resize((256, 256), Image.LANCZOS))
                        p_full = np.asarray(Image.fromarray(p_u8).resize((256, 256), Image.LANCZOS))
                        t_crop = t_u8[160:416, 128:384]
                        p_crop = p_u8[160:416, 128:384]
                        cols.append(np.concatenate([t_full, p_full, t_crop, p_crop], axis=0))
                    grid = np.concatenate(cols, axis=1)
                    Image.fromarray(grid).save(os.path.join(out_dir, "val_teacher_vs_student512.png"))

    print(f"\nTraining complete! Best unseen 512px PSNR = {best_psnr:.2f} dB")
    print(f"Saved folded safetensors -> {os.path.join(out_dir, 'student_renderer_512.safetensors')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "litert_poc", "distill_data_512"))
    ap.add_argument("--n-per-avatar", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--gan-start", type=int, default=25)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    if args.generate:
        generate_teacher_dataset_512(args.out_dir, n_per_avatar=args.n_per_avatar)
    if args.train:
        train_student_512(
            args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gan_start_epoch=args.gan_start,
        )


if __name__ == "__main__":
    main()

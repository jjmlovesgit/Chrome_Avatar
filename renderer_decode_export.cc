// IMTalker IMTRenderer per-frame decoder (renderer_decode.tflite) exporter.
// Loads weights directly from checkpoints/renderer.safetensors via LiteRT's
// SafetensorLoader (stripping training-only criterion_vgg and one-time
// dense_feature_encoder / latent_token_encoder), folds all BatchNorm2d layers
// into Conv2D weights, rewrites batch=1 ModulatedConv2d with static weights,
// uses strictly rank <= 4 ops (zero Split, zero 5D/6D tensors) for 100% WebGPU
// execution, serializes renderer_decode.tflite via ModelFactory::Save, and
// optionally verifies PSNR against PyTorch golden tensors.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/utils/safetensor_loader.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/signature_runner.h"

namespace {

using litert::tensor::Add;
using litert::tensor::ArgMax;
using litert::tensor::BatchMatMul;
using litert::tensor::Concatenation;
using litert::tensor::Conv2D;
using litert::tensor::DepthToSpace;
using litert::tensor::DepthwiseConv2D;
using litert::tensor::FullyConnected;
using litert::tensor::Gather;
using litert::tensor::Gelu;
using litert::tensor::kPaddingSame;
using litert::tensor::kPaddingValid;
using litert::tensor::LeakyRelu;
using litert::tensor::Logistic;
using litert::tensor::Mean;
using litert::tensor::ModelFactory;
using litert::tensor::Mul;
using litert::tensor::OwningCpuBuffer;
using litert::tensor::Pad;
using litert::tensor::Reshape;
using litert::tensor::ResizeBilinear;
using litert::tensor::ResizeNearestNeighbor;
using litert::tensor::Rsqrt;
using litert::tensor::Slice;
using litert::tensor::Softmax;
using litert::tensor::Sub;
using litert::tensor::Tensor;
using litert::tensor::TensorHandle;
using litert::tensor::TfLiteMixinTag;
using litert::tensor::Transpose;
using litert::tensor::TransposeConv;
using litert::tensor::Type;
using litert::tensor::examples::SafetensorLoader;

using T = Tensor<TfLiteMixinTag>;

bool g_fp16_weights = false;
bool g_int8_weights = false;
bool g_fast_mobile = false;
bool g_fast_synth64 = false;
bool g_skip_swin256 = false;
bool g_skip_swin256_attn = false;
bool g_fast_swin256 = false;
bool g_fast_swin128 = false;
bool g_skip_swin128 = false;
bool g_skip_swin128_b1 = false;
bool g_skip_res256_extra = false;
bool g_out_256 = false;

T MakeConstF32(const std::string& name, std::vector<int> shape,
               const std::vector<float>& data) {
  if (g_int8_weights && data.size() >= 4096 &&
      (shape.size() == 2 || shape.size() == 4) && shape[0] > 1) {
    const int out_c = shape[0];
    const size_t inner = data.size() / static_cast<size_t>(out_c);
    std::vector<int8_t> q(data.size());
    std::vector<float> scales(out_c);
    std::vector<int64_t> zps(out_c, 0);
    for (int o = 0; o < out_c; ++o) {
      float max_abs = 0.0f;
      const size_t base = static_cast<size_t>(o) * inner;
      for (size_t j = 0; j < inner; ++j) {
        float a = std::abs(data[base + j]);
        if (a > max_abs) max_abs = a;
      }
      float s = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1e-8f;
      scales[o] = s;
      float inv_s = 1.0f / s;
      for (size_t j = 0; j < inner; ++j) {
        int v = static_cast<int>(std::lrint(data[base + j] * inv_s));
        if (v > 127) v = 127;
        else if (v < -127) v = -127;
        q[base + j] = static_cast<int8_t>(v);
      }
    }
    T raw({.name = name + "_i8",
           .type = Type::kI8,
           .shape = std::move(shape),
           .buffer = OwningCpuBuffer::Copy<Type::kI8>(q)});
    auto quant = std::make_shared<litert::tensor::PerChannelAffineQuantization>();
    quant->scales = std::move(scales);
    quant->zero_points = std::move(zps);
    quant->quantized_dimension = 0;
    raw.SetQuantization(std::move(quant));
    return litert::tensor::Dequantize(raw);
  }
  if (g_fp16_weights && data.size() >= 4096) {
    std::vector<litert::tensor::fp16_t> h(data.size());
    for (size_t i = 0; i < data.size(); ++i) h[i] = litert::tensor::fp16_t(data[i]);
    T raw({.name = name + "_fp16",
           .type = Type::kFP16,
           .shape = std::move(shape),
           .buffer = OwningCpuBuffer::Copy<Type::kFP16>(h)});
    return litert::tensor::Dequantize(raw);
  }
  return T({.name = name,
            .type = Type::kFP32,
            .shape = std::move(shape),
            .buffer = OwningCpuBuffer::Copy<Type::kFP32>(data)});
}

T MakeZeroBias(const std::string& name, int channels) {
  std::vector<float> zeros(channels, 0.0f);
  return MakeConstF32(name, {channels}, zeros);
}

absl::StatusOr<T> LoadRawF32(const SafetensorLoader& loader,
                             absl::string_view name,
                             std::vector<int> override_shape = {}) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, loader.GetTensorInfo(name));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> data,
                              loader.ReadTensorAs<float>(name));
  std::vector<int> shape;
  if (!override_shape.empty()) {
    shape = std::move(override_shape);
  } else {
    shape.reserve(info.shape.size());
    for (int64_t d : info.shape) shape.push_back(static_cast<int>(d));
  }
  return MakeConstF32(std::string(name), std::move(shape), data);
}

// PyTorch Conv2d [OutC, InC, KH, KW] -> TFLite NHWC filter [OutC, KH, KW, InC]
std::vector<float> OihwToOhwi(const std::vector<float>& src, int out_c,
                              int in_c, int kh, int kw) {
  std::vector<float> dst(src.size());
  for (int o = 0; o < out_c; ++o) {
    for (int i = 0; i < in_c; ++i) {
      for (int h = 0; h < kh; ++h) {
        for (int w = 0; w < kw; ++w) {
          size_t s_idx = (((size_t)o * in_c + i) * kh + h) * kw + w;
          size_t d_idx = (((size_t)o * kh + h) * kw + w) * in_c + i;
          dst[d_idx] = src[s_idx];
        }
      }
    }
  }
  return dst;
}

// EqualLinear(in_dim, out_dim, lr_mul=1): weight scaled by 1 / sqrt(in_dim)
absl::StatusOr<T> ApplyEqualLinear(const SafetensorLoader& loader,
                                   const std::string& pfx, T x) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& w_info,
                              loader.GetTensorInfo(pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> w_data,
                              loader.ReadTensorAs<float>(pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> b_data,
                              loader.ReadTensorAs<float>(pfx + ".bias"));
  const int out_dim = static_cast<int>(w_info.shape[0]);
  const int in_dim = static_cast<int>(w_info.shape[1]);
  const float scale = 1.0f / std::sqrt(static_cast<float>(in_dim));
  for (float& v : w_data) v *= scale;
  T w = MakeConstF32(pfx + ".w_scaled", {out_dim, in_dim}, w_data);
  T b = MakeConstF32(pfx + ".bias", {out_dim}, b_data);
  return FullyConnected(x, w, b);
}

// Standard nn.Linear
absl::StatusOr<T> ApplyLinear(const SafetensorLoader& loader,
                              const std::string& pfx, T x) {
  LRT_TENSOR_ASSIGN_OR_RETURN(T w, LoadRawF32(loader, pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b, LoadRawF32(loader, pfx + ".bias"));
  return FullyConnected(x, w, b);
}

// LayerNorm with affine weight & bias over last axis (axis = rank - 1), eps=1e-5
absl::StatusOr<T> ApplyLayerNorm(const SafetensorLoader& loader,
                                 const std::string& pfx, T x, int axis) {
  LRT_TENSOR_ASSIGN_OR_RETURN(T w, LoadRawF32(loader, pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b, LoadRawF32(loader, pfx + ".bias"));
  T mean = Mean(x, std::vector<int>{axis}, /*keep_dims=*/true);
  T diff = Sub(x, mean);
  T var = Mean(Mul(diff, diff), std::vector<int>{axis}, /*keep_dims=*/true);
  T norm = Mul(diff, Rsqrt(Add(var, 1e-5f)));
  return Add(Mul(norm, w), b);
}

// Conv2D (k=3, stride=1, SAME) with optional folded BatchNorm2d (eps=1e-5).
absl::StatusOr<T> ApplyConv3x3(const SafetensorLoader& loader,
                               const std::string& conv_w_key,
                               const std::string& bn_pfx, T x) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& w_info,
                              loader.GetTensorInfo(conv_w_key));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> w_oihw,
                              loader.ReadTensorAs<float>(conv_w_key));
  const int out_c = static_cast<int>(w_info.shape[0]);
  const int in_c = static_cast<int>(w_info.shape[1]);
  const int kh = static_cast<int>(w_info.shape[2]);
  const int kw = static_cast<int>(w_info.shape[3]);

  std::vector<float> bias(out_c, 0.0f);
  if (!bn_pfx.empty()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> gamma,
                                loader.ReadTensorAs<float>(bn_pfx + ".weight"));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> beta,
                                loader.ReadTensorAs<float>(bn_pfx + ".bias"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<float> mean,
        loader.ReadTensorAs<float>(bn_pfx + ".running_mean"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<float> var,
        loader.ReadTensorAs<float>(bn_pfx + ".running_var"));
    for (int o = 0; o < out_c; ++o) {
      float scale = gamma[o] / std::sqrt(var[o] + 1e-5f);
      bias[o] = beta[o] - mean[o] * scale;
      size_t base = (size_t)o * in_c * kh * kw;
      size_t stride = (size_t)in_c * kh * kw;
      for (size_t k = 0; k < stride; ++k) {
        w_oihw[base + k] *= scale;
      }
    }
  }

  std::vector<float> w_ohwi = OihwToOhwi(w_oihw, out_c, in_c, kh, kw);
  T w_t = MakeConstF32(conv_w_key + ".nhwc", {out_c, kh, kw, in_c}, w_ohwi);
  T b_t = MakeConstF32(conv_w_key + ".bias", {out_c}, bias);
  return Conv2D(x, w_t, b_t, /*stride_h=*/1, /*stride_w=*/1, kPaddingSame);
}

// FeatResBlock (modules.py): conv1+BN+LeakyReLU(0.01) -> conv2+BN -> +res -> LeakyReLU(0.01)
absl::StatusOr<T> ApplyFeatResBlock(const SafetensorLoader& loader,
                                    const std::string& pfx, T x) {
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T out, ApplyConv3x3(loader, pfx + ".conv1.conv.weight",
                          pfx + ".conv1.norm.norm", x));
  out = LeakyRelu(out, 0.01f);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      out, ApplyConv3x3(loader, pfx + ".conv2.conv.weight",
                        pfx + ".conv2.norm.norm", out));
  return LeakyRelu(Add(out, x), 0.01f);
}

// ConvResBlock / UpConvResBlock (modules.py)
absl::StatusOr<T> ApplyResBlock(const SafetensorLoader& loader,
                                const std::string& pfx, T x,
                                int upsample_to_res,
                                bool skip_extra_feat_res = false) {
  if (upsample_to_res > 0) {
    x = ResizeNearestNeighbor(x, {upsample_to_res, upsample_to_res},
                              /*align_corners=*/false,
                              /*half_pixel_centers=*/false);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T out,
      ApplyConv3x3(loader, pfx + ".conv1.weight", pfx + ".norm.norm", x));
  out = LeakyRelu(out, 0.01f);
  LRT_TENSOR_ASSIGN_OR_RETURN(out,
                              ApplyConv3x3(loader, pfx + ".conv2.weight", "", out));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      out, ApplyFeatResBlock(loader, pfx + ".feat_res_block1", out));
  if (!skip_extra_feat_res) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        out, ApplyFeatResBlock(loader, pfx + ".feat_res_block2", out));
  }
  return out;
}

// StyledConv (batch=1, NHWC, static weights!)
absl::StatusOr<T> ApplyStyledConv(const SafetensorLoader& loader,
                                  const std::string& pfx, T x, T style_latent,
                                  int cur_res, bool upsample) {
  // 1. Style modulation: EqualLinear(32 -> in_c)
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T style, ApplyEqualLinear(loader, pfx + ".conv.modulation", style_latent));

  // 2. Load conv weight [1, out_c, in_c, 3, 3] and pre-scale by 1/sqrt(in_c*9)
  const std::string w_key = pfx + ".conv.weight";
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& w_info, loader.GetTensorInfo(w_key));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> w_raw,
                              loader.ReadTensorAs<float>(w_key));
  const int out_c = static_cast<int>(w_info.shape[1]);
  const int in_c = static_cast<int>(w_info.shape[2]);
  const int kh = static_cast<int>(w_info.shape[3]);
  const int kw = static_cast<int>(w_info.shape[4]);
  const float w_scale = 1.0f / std::sqrt(static_cast<float>(in_c * kh * kw));
  for (float& v : w_raw) v *= w_scale;

  // 3. Algebraic demodulation: demod[o] = rsqrt(sum_i (W_sq[o,i] * style[i]^2) + 1e-8)
  std::vector<float> w_sq(out_c * in_c, 0.0f);
  for (int o = 0; o < out_c; ++o) {
    for (int i = 0; i < in_c; ++i) {
      double s = 0.0;
      size_t base = ((size_t)o * in_c + i) * kh * kw;
      for (int k = 0; k < kh * kw; ++k) {
        double v = static_cast<double>(w_raw[base + k]);
        s += v * v;
      }
      w_sq[o * in_c + i] = static_cast<float>(s);
    }
  }
  T w_sq_t = MakeConstF32(pfx + ".w_sq", {out_c, in_c}, w_sq);
  T demod = Rsqrt(Add(FullyConnected(Mul(style, style), w_sq_t), 1e-8f));

  // 4. Scale input channels by style (batch=1 exact equivalence)
  T x_mod = Mul(x, Reshape(style, {1, 1, 1, in_c}));

  // 5. Static Conv2D or Upsample TransposeConv + Blur
  std::vector<float> w_ohwi = OihwToOhwi(w_raw, out_c, in_c, kh, kw);
  T w_t = MakeConstF32(pfx + ".w_nhwc", {out_c, kh, kw, in_c}, w_ohwi);
  T zero_b = MakeZeroBias(pfx + ".zero_b", out_c);

  T y = x_mod;
  if (!upsample) {
    y = Conv2D(x_mod, w_t, zero_b, /*stride_h=*/1, /*stride_w=*/1, kPaddingSame);
  } else {
    // TransposeConv (padding=VALID, stride=2): [1, H, W, in_c] -> [1, 2H+1, 2W+1, out_c]
    int up_h = 2 * cur_res + 1;
    y = TransposeConv(w_t, x_mod, zero_b, {1, up_h, up_h, out_c}, kPaddingValid,
                      /*stride_h=*/2, /*stride_w=*/2);
    // Blur: pad=(1, 1) on H and W (p=(4-2)-(3-1)=0 -> pad0=1, pad1=1),
    // then DepthwiseConv2D with flipped 4x4 kernel: (2H+1)+2-4+1 = 2H
    T paddings({.name = pfx + ".blur_pad",
                .type = Type::kI32,
                .shape = {4, 2},
                .buffer = OwningCpuBuffer::Copy<Type::kI32>(
                    {0, 0, 1, 1, 1, 1, 0, 0})});
    y = Pad(y, paddings);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<float> bk,
        loader.ReadTensorAs<float>(pfx + ".conv.blur.kernel"));
    // DepthwiseConv2D filter: [1, 4, 4, out_c], flipped spatially per upfirdn2d_native
    std::vector<float> dw_filter(4 * 4 * out_c);
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        float kv = bk[(3 - r) * 4 + (3 - c)];
        for (int o = 0; o < out_c; ++o) {
          dw_filter[(r * 4 + c) * out_c + o] = kv;
        }
      }
    }
    T dw_t = MakeConstF32(pfx + ".blur_dw", {1, 4, 4, out_c}, dw_filter);
    y = DepthwiseConv2D(y, dw_t, zero_b, /*stride_h=*/1, /*stride_w=*/1,
                        kPaddingValid);
  }

  // 6. Scale output channels by demod + FusedLeakyReLU(bias, 0.2, sqrt(2))
  y = Mul(y, Reshape(demod, {1, 1, 1, out_c}));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> act_b_vec,
      loader.ReadTensorAs<float>(pfx + ".activate.bias"));
  T act_b = MakeConstF32(pfx + ".act_bias", {1, 1, 1, out_c}, act_b_vec);
  return Mul(LeakyRelu(Add(y, act_b), 0.2f), std::sqrt(2.0f));
}

// StandardUnifiedAttention (seq_len = N, dim = C, num_heads = 8)
// Returns {out_seq [1, N, C], attn_map [1, 8, N, N]}
absl::StatusOr<std::pair<T, T>> ApplyStandardAttn(
    const SafetensorLoader& loader, const std::string& pfx, T q_seq, T k_seq,
    T v_seq, int n_tokens, int dim, int num_heads) {
  const int head_dim = dim / num_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  LRT_TENSOR_ASSIGN_OR_RETURN(T q, ApplyLinear(loader, pfx + ".q_proj", q_seq));
  LRT_TENSOR_ASSIGN_OR_RETURN(T k, ApplyLinear(loader, pfx + ".k_proj", k_seq));
  LRT_TENSOR_ASSIGN_OR_RETURN(T v, ApplyLinear(loader, pfx + ".v_proj", v_seq));
  q = Transpose(Reshape(q, {1, n_tokens, num_heads, head_dim}), {0, 2, 1, 3});
  k = Transpose(Reshape(k, {1, n_tokens, num_heads, head_dim}), {0, 2, 1, 3});
  v = Transpose(Reshape(v, {1, n_tokens, num_heads, head_dim}), {0, 2, 1, 3});
  T scores = BatchMatMul(Mul(q, scale), k, /*adj_x=*/false, /*adj_y=*/true);
  T attn = Softmax(scores);
  T ctx = BatchMatMul(attn, v);
  ctx = Reshape(Transpose(ctx, {0, 2, 1, 3}), {1, n_tokens, dim});
  LRT_TENSOR_ASSIGN_OR_RETURN(T out, ApplyLinear(loader, pfx + ".proj", ctx));
  return std::make_pair(out, attn);
}

// UnifiedTransformerBlock (self-attention at 16x16, 32x32, 64x64)
absl::StatusOr<T> ApplyUnifiedTransformerBlock(const SafetensorLoader& loader,
                                               const std::string& pfx, T x,
                                               int res, int dim) {
  const int n = res * res;
  T seq = Reshape(x, {1, n, dim});
  T attn_delta = seq;
  if ((g_fast_mobile || g_fast_synth64) && res == 64) {
    // 16x smaller attention (32x32 = 1024 tokens instead of 64x64 = 4096 tokens)
    // while preserving the full-resolution 64x64 residual & FFN path!
    constexpr int kHalf = 32;
    constexpr int kNHalf = kHalf * kHalf;
    T x_half = ResizeBilinear(x, {kHalf, kHalf}, /*align_corners=*/false,
                              /*half_pixel_centers=*/true);
    T seq_half = Reshape(x_half, {1, kNHalf, dim});
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> qp_raw,
                                loader.ReadTensorAs<float>(pfx + ".q_pos_embedding"));
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> kp_raw,
                                loader.ReadTensorAs<float>(pfx + ".k_pos_embedding"));
    auto PoolPos2x = [&](const std::vector<float>& src) {
      std::vector<float> dst(kNHalf * dim, 0.0f);
      for (int r = 0; r < kHalf; ++r) {
        for (int c = 0; c < kHalf; ++c) {
          for (int d = 0; d < dim; ++d) {
            float s = 0.0f;
            for (int dr = 0; dr < 2; ++dr)
              for (int dc = 0; dc < 2; ++dc)
                s += src[((r * 2 + dr) * 64 + (c * 2 + dc)) * dim + d];
            dst[(r * kHalf + c) * dim + d] = s * 0.25f;
          }
        }
      }
      return dst;
    };
    T q_pos = MakeConstF32(pfx + ".q_pos_32", {1, kNHalf, dim}, PoolPos2x(qp_raw));
    T k_pos = MakeConstF32(pfx + ".k_pos_32", {1, kNHalf, dim}, PoolPos2x(kp_raw));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T q_norm, ApplyLayerNorm(loader, pfx + ".norm_q", Add(seq_half, q_pos), 2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T k_norm, ApplyLayerNorm(loader, pfx + ".norm_kv", Add(seq_half, k_pos), 2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T v_norm, ApplyLayerNorm(loader, pfx + ".norm_kv", seq_half, 2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto attn_pair, ApplyStandardAttn(loader, pfx + ".attn", q_norm, k_norm,
                                          v_norm, kNHalf, dim, /*num_heads=*/8));
    T up = ResizeBilinear(Reshape(attn_pair.first, {1, kHalf, kHalf, dim}),
                          {res, res}, /*align_corners=*/false,
                          /*half_pixel_centers=*/true);
    attn_delta = Reshape(up, {1, n, dim});
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(T q_pos,
                                LoadRawF32(loader, pfx + ".q_pos_embedding"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T k_pos,
                                LoadRawF32(loader, pfx + ".k_pos_embedding"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T q_norm, ApplyLayerNorm(loader, pfx + ".norm_q", Add(seq, q_pos), 2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T k_norm, ApplyLayerNorm(loader, pfx + ".norm_kv", Add(seq, k_pos), 2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T v_norm, ApplyLayerNorm(loader, pfx + ".norm_kv", seq, 2));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto attn_pair, ApplyStandardAttn(loader, pfx + ".attn", q_norm, k_norm,
                                          v_norm, n, dim, /*num_heads=*/8));
    attn_delta = attn_pair.first;
  }
  T h = Add(seq, attn_delta);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T h_norm, ApplyLayerNorm(loader, pfx + ".norm_ffn", h, 2));
  LRT_TENSOR_ASSIGN_OR_RETURN(T m, ApplyLinear(loader, pfx + ".mlp.0", h_norm));
  m = Gelu(m, /*approximate=*/false);
  LRT_TENSOR_ASSIGN_OR_RETURN(m, ApplyLinear(loader, pfx + ".mlp.3", m));
  return Reshape(Add(h, m), {1, res, res, dim});
}

// 4D cyclic roll on NHWC [1, H, W, C] by (shift_h, shift_w)
T CyclicRollNHWC(T x, int h, int w, int c, int shift_h, int shift_w) {
  // PyTorch torch.roll(x, shifts=(sh, sw), dims=(1, 2)):
  // Negative shift (-4) moves top 4 rows to bottom -> cut at k = 4
  // Positive shift (+4) moves bottom 4 rows to top -> cut at k = H - 4
  int kh = ((-shift_h) % h + h) % h;
  int kw = ((-shift_w) % w + w) % w;
  if (kh > 0) {
    T top = Slice(x, {0, kh, 0, 0}, {1, h - kh, w, c});
    T bot = Slice(x, {0, 0, 0, 0}, {1, kh, w, c});
    x = Concatenation({top, bot}, /*axis=*/1);
  }
  if (kw > 0) {
    T left = Slice(x, {0, 0, kw, 0}, {1, h, w - kw, c});
    T right = Slice(x, {0, 0, 0, 0}, {1, h, kw, c});
    x = Concatenation({left, right}, /*axis=*/2);
  }
  return x;
}

// Pure Rank <= 4 window_partition: [1, H, W, C] -> [nW, w*w, C]
T WindowPartition4D(T x, int h, int w, int c, int win) {
  int nh = h / win;
  int nw = w / win;
  T r = Reshape(x, {nh, win, nw, win * c});
  T tr = Transpose(r, {0, 2, 1, 3});
  return Reshape(tr, {nh * nw, win * win, c});
}

// Pure Rank <= 4 window_reverse: [nW, w*w, C] -> [1, H, W, C]
T WindowReverse4D(T x, int h, int w, int c, int win) {
  int nh = h / win;
  int nw = w / win;
  T r = Reshape(x, {nh, nw, win, win * c});
  T tr = Transpose(r, {0, 2, 1, 3});
  return Reshape(tr, {1, h, w, c});
}

// UnifiedSwinBlock (self-attention at 128x128 & 256x256, window_size=8)
// mode: 0 = exact full-res, 1 = 2x downsampled attn (4x faster, full-res MLP), 2 = skip attn (MLP only)
absl::StatusOr<T> ApplyUnifiedSwinBlock(const SafetensorLoader& loader,
                                        const std::string& pfx, T x, int res,
                                        int dim, int shift_size, int mode = 0) {
  T out = x;
  if (mode != 2) {
    constexpr int kWin = 8;
    constexpr int kWinSq = kWin * kWin;  // 64
    constexpr int kHeads = 8;
    const int head_dim = dim / kHeads;
    const int work_res = (mode == 1) ? (res / 2) : res;
    const int nw_total = (work_res / kWin) * (work_res / kWin);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int eff_shift = (mode == 1) ? 0 : shift_size;

    T x_work = (mode == 1)
                   ? ResizeBilinear(x, {work_res, work_res}, false, true)
                   : x;
    LRT_TENSOR_ASSIGN_OR_RETURN(T q_in,
                                ApplyLayerNorm(loader, pfx + ".norm_q", x_work, 3));
    LRT_TENSOR_ASSIGN_OR_RETURN(T kv_in,
                                ApplyLayerNorm(loader, pfx + ".norm_kv", x_work, 3));

    if (eff_shift > 0) {
      q_in = CyclicRollNHWC(q_in, work_res, work_res, dim, -eff_shift, -eff_shift);
      kv_in = CyclicRollNHWC(kv_in, work_res, work_res, dim, -eff_shift, -eff_shift);
    }

    T q_win = WindowPartition4D(q_in, work_res, work_res, dim, kWin);   // [nW, 64, C]
    T kv_win = WindowPartition4D(kv_in, work_res, work_res, dim, kWin);  // [nW, 64, C]

    // SwinUnifiedAttention
    const std::string apfx = pfx + ".attn";
    LRT_TENSOR_ASSIGN_OR_RETURN(T q, ApplyLinear(loader, apfx + ".q", q_win));
    LRT_TENSOR_ASSIGN_OR_RETURN(T k, ApplyLinear(loader, apfx + ".k", kv_win));
    LRT_TENSOR_ASSIGN_OR_RETURN(T v, ApplyLinear(loader, apfx + ".v", kv_win));
    q = Transpose(Reshape(q, {nw_total, kWinSq, kHeads, head_dim}), {0, 2, 1, 3});
    k = Transpose(Reshape(k, {nw_total, kWinSq, kHeads, head_dim}), {0, 2, 1, 3});
    v = Transpose(Reshape(v, {nw_total, kWinSq, kHeads, head_dim}), {0, 2, 1, 3});

    T scores = BatchMatMul(Mul(q, scale), k, /*adj_x=*/false, /*adj_y=*/true);

    // Precompute static relative_position_bias [1, 8, 64, 64]
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<float> rpb_table,
        loader.ReadTensorAs<float>(apfx + ".relative_position_bias_table"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        std::vector<int64_t> rpb_idx,
        loader.ReadTensorAs<int64_t>(apfx + ".relative_position_index"));
    std::vector<float> rpb(kHeads * kWinSq * kWinSq);
    for (int i = 0; i < kWinSq; ++i) {
      for (int j = 0; j < kWinSq; ++j) {
        int64_t idx = rpb_idx[i * kWinSq + j];
        for (int hd = 0; hd < kHeads; ++hd) {
          rpb[(hd * kWinSq + i) * kWinSq + j] = rpb_table[idx * kHeads + hd];
        }
      }
    }
    T rpb_t = MakeConstF32(apfx + ".rpb_static", {1, kHeads, kWinSq, kWinSq}, rpb);
    scores = Add(scores, rpb_t);

    if (eff_shift > 0) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          T mask, LoadRawF32(loader, pfx + ".attn_mask",
                             {nw_total, 1, kWinSq, kWinSq}));
      scores = Add(scores, mask);
    }

    T attn = Softmax(scores);
    T ctx = BatchMatMul(attn, v);
    ctx = Reshape(Transpose(ctx, {0, 2, 1, 3}), {nw_total, kWinSq, dim});
    LRT_TENSOR_ASSIGN_OR_RETURN(T attn_win,
                                ApplyLinear(loader, apfx + ".proj", ctx));

    T shifted_x = WindowReverse4D(attn_win, work_res, work_res, dim, kWin);
    T delta = (eff_shift > 0)
                  ? CyclicRollNHWC(shifted_x, work_res, work_res, dim, eff_shift, eff_shift)
                  : shifted_x;
    if (mode == 1) {
      delta = ResizeBilinear(delta, {res, res}, false, true);
    }
    out = Add(x, delta);
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      T out_norm, ApplyLayerNorm(loader, pfx + ".norm_ffn", out, 3));
  LRT_TENSOR_ASSIGN_OR_RETURN(T m, ApplyLinear(loader, pfx + ".mlp.0", out_norm));
  m = Gelu(m, /*approximate=*/false);
  LRT_TENSOR_ASSIGN_OR_RETURN(m, ApplyLinear(loader, pfx + ".mlp.2", m));
  return Add(out, m);
}

absl::Status ExportRendererDecode(const std::string& safetensors_path,
                                  const std::string& output_tflite_path) {
  std::cout << "Loading renderer safetensors: " << safetensors_path << "\n";
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(safetensors_path));

  // Graph Inputs (avatar-agnostic! Precomputed portrait cache + motion token)
  T t_c({.name = "t_c", .type = Type::kFP32, .shape = {1, 32}});
  T i_r({.name = "i_r", .type = Type::kFP32, .shape = {1, 512}});
  T ma_r_0({.name = "ma_r_0", .type = Type::kFP32, .shape = {1, 8, 8, 512}});
  T ma_r_1({.name = "ma_r_1", .type = Type::kFP32, .shape = {1, 16, 16, 512}});
  T ma_r_2({.name = "ma_r_2", .type = Type::kFP32, .shape = {1, 32, 32, 256}});
  T ma_r_3({.name = "ma_r_3", .type = Type::kFP32, .shape = {1, 64, 64, 128}});
  T f_r_0({.name = "f_r_0", .type = Type::kFP32, .shape = {1, 8, 8, 512}});
  T f_r_1({.name = "f_r_1", .type = Type::kFP32, .shape = {1, 16, 16, 512}});
  T f_r_2({.name = "f_r_2", .type = Type::kFP32, .shape = {1, 32, 32, 256}});
  T f_r_3({.name = "f_r_3", .type = Type::kFP32, .shape = {1, 64, 64, 128}});
  T f_r_4_pool(
      {.name = "f_r_4_pool", .type = Type::kFP32, .shape = {4096, 64}});
  T f_r_5_pool(
      {.name = "f_r_5_pool", .type = Type::kFP32, .shape = {4096, 32}});

  // =========================================================================
  // Stage 1: IdentidyAdaptive (adapt.*)
  // =========================================================================
  T ad = Concatenation({t_c, i_r}, /*axis=*/1);  // [1, 544]
  LRT_TENSOR_ASSIGN_OR_RETURN(ad, ApplyEqualLinear(loader, "adapt.in_layer", ad));
  for (int l = 0; l < 4; ++l) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        ad, ApplyEqualLinear(loader, absl::StrCat("adapt.linear_layers.", l), ad));
    ad = LeakyRelu(ad, 0.2f);
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T ta_c, ApplyEqualLinear(loader, "adapt.final_linear", ad));  // [1, 32]

  // =========================================================================
  // Stage 2: MotionDecoder (latent_token_decoder.*)
  // =========================================================================
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> const_nchw,
      loader.ReadTensorAs<float>("latent_token_decoder.const"));
  // [1, 32, 4, 4] NCHW -> [1, 4, 4, 32] NHWC
  std::vector<float> const_nhwc(32 * 4 * 4);
  for (int c = 0; c < 32; ++c) {
    for (int h = 0; h < 4; ++h) {
      for (int w = 0; w < 4; ++w) {
        const_nhwc[(h * 4 + w) * 32 + c] = const_nchw[(c * 4 + h) * 4 + w];
      }
    }
  }
  T m = MakeConstF32("latent_token_decoder.const_nhwc", {1, 4, 4, 32},
                     const_nhwc);

  int cur_res = 4;
  std::vector<T> ma_c;
  for (int i = 0; i < 13; ++i) {
    bool upsample = (i == 1 || i == 4 || i == 7 || i == 10);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        m, ApplyStyledConv(
               loader, absl::StrCat("latent_token_decoder.style_conv_layers.", i),
               m, ta_c, cur_res, upsample));
    if (upsample) cur_res *= 2;
    if (i == 3 || i == 6 || i == 9 || i == 12) {
      ma_c.push_back(m);
    }
  }

  // =========================================================================
  // Stage 3: IMT CrossAttention (levels 0..3) + Fast GuidedResampler (4..5)
  // =========================================================================
  std::vector<T> ma_r = {ma_r_0, ma_r_1, ma_r_2, ma_r_3};
  std::vector<T> f_r = {f_r_0, f_r_1, f_r_2, f_r_3};
  const int coarse_res[4] = {8, 16, 32, 64};
  const int coarse_dim[4] = {512, 512, 256, 128};

  std::vector<T> aligned;
  aligned.reserve(6);
  T attn_64 = t_c;  // overwritten at lvl == 3

  for (int lvl = 0; lvl < 4; ++lvl) {
    int r = coarse_res[lvl];
    int d = coarse_dim[lvl];
    if (g_fast_mobile && lvl == 3) {
      // 16x smaller IMT Level-3 attention (32x32 = 1024 instead of 64x64 = 4096)
      constexpr int kHalf = 32;
      constexpr int kNHalf = kHalf * kHalf;
      T a32 = ResizeBilinear(ma_c[3], {kHalf, kHalf}, false, true);
      T b32 = ResizeBilinear(ma_r[3], {kHalf, kHalf}, false, true);
      T c32 = ResizeBilinear(f_r[3], {kHalf, kHalf}, false, true);
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto res_pair,
          ApplyStandardAttn(loader, "imt.3.block_efc",
                            Reshape(a32, {1, kNHalf, d}),
                            Reshape(b32, {1, kNHalf, d}),
                            Reshape(c32, {1, kNHalf, d}), kNHalf, d, 8));
      aligned.push_back(ResizeBilinear(
          Reshape(res_pair.first, {1, kHalf, kHalf, d}), {64, 64}, false, true));
      attn_64 = res_pair.second;  // [1, 8, 1024, 1024]
    } else {
      int n = r * r;
      T a_seq = Reshape(ma_c[lvl], {1, n, d});
      T b_seq = Reshape(ma_r[lvl], {1, n, d});
      T c_seq = Reshape(f_r[lvl], {1, n, d});
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto res_pair,
          ApplyStandardAttn(loader, absl::StrCat("imt.", lvl, ".block_efc"), a_seq,
                            b_seq, c_seq, n, d, /*num_heads=*/8));
      aligned.push_back(Reshape(res_pair.first, {1, r, r, d}));
      if (lvl == 3) {
        attn_64 = res_pair.second;  // [1, 8, 4096, 4096]
      }
    }
  }

  // Exact algebraic GuidedResampler for levels 4 (128x128) and 5 (256x256):
  T attn_mean = Mean(attn_64, std::vector<int>{1}, /*keep_dims=*/false);
  if (g_fast_mobile) {
    T argmax_idx = Reshape(ArgMax(attn_mean, /*axis=*/2, Type::kI32), {1024});
    T p4_32 = Reshape(ResizeBilinear(Reshape(f_r_4_pool, {1, 64, 64, 64}), {32, 32}, false, true), {1024, 64});
    T p5_32 = Reshape(ResizeBilinear(Reshape(f_r_5_pool, {1, 64, 64, 32}), {32, 32}, false, true), {1024, 32});
    T g4 = Reshape(Gather(p4_32, argmax_idx, /*axis=*/0), {1, 32, 32, 64});
    aligned.push_back(ResizeBilinear(g4, {128, 128}, false, true));
    T g5 = Reshape(Gather(p5_32, argmax_idx, /*axis=*/0), {1, 32, 32, 32});
    aligned.push_back(ResizeBilinear(g5, {256, 256}, false, true));
  } else {
    T argmax_idx = Reshape(ArgMax(attn_mean, /*axis=*/2, Type::kI32), {4096});
    T g4 = Reshape(Gather(f_r_4_pool, argmax_idx, /*axis=*/0), {1, 64, 64, 64});
    aligned.push_back(ResizeNearestNeighbor(g4, {128, 128}, false, false));
    T g5 = Reshape(Gather(f_r_5_pool, argmax_idx, /*axis=*/0), {1, 64, 64, 32});
    int last_res = g_out_256 ? 128 : 256;
    aligned.push_back(ResizeNearestNeighbor(g5, {last_res, last_res}, false, false));
  }

  // =========================================================================
  // Stage 4: SynthesisNetwork (frame_decoder.*)
  // =========================================================================
  const int synth_res[5] = {16, 32, 64, 128, 256};
  const int synth_dim[5] = {512, 256, 128, 64, 32};

  T x = aligned[0];  // [1, 8, 8, 512]
  for (int i = 0; i < 5; ++i) {
    int r = synth_res[i];
    int d = synth_dim[i];
    int eff_r = (g_out_256 && i == 4) ? 128 : r;
    int up_r = (g_out_256 && i == 4) ? 0 : r;
    const bool skip_extra = (g_skip_res256_extra && r == 256);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        x, ApplyResBlock(loader,
                         absl::StrCat("frame_decoder.upconv_blocks.", i), x, up_r,
                         skip_extra));
    x = Concatenation({x, aligned[i + 1]}, /*axis=*/3);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        x, ApplyResBlock(loader, absl::StrCat("frame_decoder.resblocks.", i), x,
                         /*upsample_to_res=*/0, skip_extra));
    const std::string tb_pfx =
        absl::StrCat("frame_decoder.transformer_blocks.", i);
    if (r < 128) {
      LRT_TENSOR_ASSIGN_OR_RETURN(
          x, ApplyUnifiedTransformerBlock(loader, tb_pfx + ".blocks.0", x, r, d));
    } else if (r == 128) {
      if (!g_skip_swin128) {
        int m128 = g_fast_swin128 ? 1 : 0;
        LRT_TENSOR_ASSIGN_OR_RETURN(
            x, ApplyUnifiedSwinBlock(loader, tb_pfx + ".blocks.0", x, r, d, 0, m128));
        if (!g_skip_swin128_b1) {
          LRT_TENSOR_ASSIGN_OR_RETURN(
              x, ApplyUnifiedSwinBlock(loader, tb_pfx + ".blocks.1", x, r, d, 4, m128));
        }
      }
    } else {
      // r == 256 (or 128 when g_out_256 is true!)
      if (!g_skip_swin256) {
        int m256 = g_skip_swin256_attn ? 2 : (g_fast_swin256 ? 1 : 0);
        // When g_out_256 is set, eff_r=128 so m256=1 runs Swin at 64x64 (16x fewer windows than 256!)
        // and shift_size=0 avoids mismatched 256x256 attn_mask shapes.
        int sh1 = g_out_256 ? 0 : 4;
        LRT_TENSOR_ASSIGN_OR_RETURN(
            x, ApplyUnifiedSwinBlock(loader, tb_pfx + ".blocks.0", x, eff_r, d, 0, m256));
        LRT_TENSOR_ASSIGN_OR_RETURN(
            x, ApplyUnifiedSwinBlock(loader, tb_pfx + ".blocks.1", x, eff_r, d, sh1, m256));
      }
    }
  }

  // Final RGB Head: LeakyReLU(0.2) -> Conv2d(32->12) -> DepthToSpace(2) -> Sigmoid
  x = LeakyRelu(x, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> fw_raw,
      loader.ReadTensorAs<float>("frame_decoder.final_conv.1.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> fb_raw,
      loader.ReadTensorAs<float>("frame_decoder.final_conv.1.bias"));

  // Reorder 12 output channels from PyTorch PixelShuffle (c, r, s) to TFLite
  // DepthToSpace (r, s, c) so DepthToSpace(2) matches PixelShuffle(2) exactly.
  std::vector<float> fw_reordered(fw_raw.size());
  std::vector<float> fb_reordered(12);
  const size_t ch_stride = 32 * 3 * 3;
  for (int r = 0; r < 2; ++r) {
    for (int s = 0; s < 2; ++s) {
      for (int c = 0; c < 3; ++c) {
        int src_o = (c * 2 + r) * 2 + s;
        int dst_o = (r * 2 + s) * 3 + c;
        fb_reordered[dst_o] = fb_raw[src_o];
        std::memcpy(fw_reordered.data() + dst_o * ch_stride,
                    fw_raw.data() + src_o * ch_stride,
                    ch_stride * sizeof(float));
      }
    }
  }
  std::vector<float> fw_ohwi = OihwToOhwi(fw_reordered, 12, 32, 3, 3);
  T fw_t = MakeConstF32("frame_decoder.final_conv.1.w_dts", {12, 3, 3, 32},
                        fw_ohwi);
  T fb_t = MakeConstF32("frame_decoder.final_conv.1.b_dts", {12}, fb_reordered);

  x = Conv2D(x, fw_t, fb_t, /*stride_h=*/1, /*stride_w=*/1, kPaddingSame);
  x = DepthToSpace(x, 2);  // [1, 512, 512, 3]
  T frame = Logistic(x);
  frame.SetName("frame");

  // Serialize .tflite
  ModelFactory factory;
  std::vector<TensorHandle> sig_inputs = {t_c,    i_r,    ma_r_0,     ma_r_1,
                                          ma_r_2, ma_r_3, f_r_0,      f_r_1,
                                          f_r_2,  f_r_3,  f_r_4_pool, f_r_5_pool};
  std::vector<TensorHandle> sig_outputs = {frame};
  LRT_TENSOR_RETURN_IF_ERROR(
      factory.AddSignature(sig_inputs, sig_outputs, "serving_default"));
  LRT_TENSOR_RETURN_IF_ERROR(factory.Save(output_tflite_path));

  std::cout << "Successfully saved " << output_tflite_path << "\n";
  return absl::OkStatus();
}

bool ReadBinFloats(const std::string& path, std::vector<float>& out) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) return false;
  const std::streamsize sz = ifs.tellg();
  if (sz <= 0 || (sz % sizeof(float)) != 0) return false;
  ifs.seekg(0, std::ios::beg);
  out.resize(static_cast<size_t>(sz) / sizeof(float));
  return static_cast<bool>(ifs.read(reinterpret_cast<char*>(out.data()), sz));
}

absl::Status VerifyTflite(const std::string& tflite_path,
                          const std::string& ref_dir) {
  const char* names[12] = {"t_c",    "i_r",    "ma_r_0",     "ma_r_1",
                           "ma_r_2", "ma_r_3", "f_r_0",      "f_r_1",
                           "f_r_2",  "f_r_3",  "f_r_4_pool", "f_r_5_pool"};
  std::vector<std::vector<float>> in_bufs(12);
  for (int i = 0; i < 12; ++i) {
    if (!ReadBinFloats(ref_dir + "/" + names[i] + ".bin", in_bufs[i])) {
      return absl::NotFoundError(
          absl::StrCat("Missing input bin: ", ref_dir, "/", names[i], ".bin"));
    }
  }
  std::vector<float> frame_ref;
  if (!ReadBinFloats(ref_dir + "/frame_ref.bin", frame_ref)) {
    return absl::NotFoundError("Missing frame_ref.bin");
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
  if (!model) return absl::InternalError("Failed to load " + tflite_path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk ||
      !interpreter) {
    return absl::InternalError("InterpreterBuilder failed.");
  }
  interpreter->SetNumThreads(12);

  tflite::SignatureRunner* runner =
      interpreter->GetSignatureRunner("serving_default");
  if (!runner) return absl::InternalError("Signature serving_default missing.");
  if (runner->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("AllocateTensors failed.");
  }

  for (int i = 0; i < 12; ++i) {
    TfLiteTensor* tensor = runner->input_tensor(names[i]);
    if (!tensor || tensor->bytes != in_bufs[i].size() * sizeof(float)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input mismatch for ", names[i]));
    }
    std::memcpy(tensor->data.f, in_bufs[i].data(), tensor->bytes);
  }

  auto t0 = std::chrono::steady_clock::now();
  if (runner->Invoke() != kTfLiteOk) {
    return absl::InternalError("Invoke failed.");
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const TfLiteTensor* out = runner->output_tensor("frame");
  if (!out) {
    return absl::InvalidArgumentError("Output tensor 'frame' missing.");
  }
  const int out_h = out->dims->data[1];
  const int out_w = out->dims->data[2];
  std::vector<float> ref_cmp = frame_ref;
  if (out_h == 256 && out_w == 256 && frame_ref.size() == 512 * 512 * 3) {
    // 2x2 box-downsample 512x512 reference to 256x256 for apples-to-apples PSNR!
    ref_cmp.assign(256 * 256 * 3, 0.0f);
    for (int y = 0; y < 256; ++y) {
      for (int x = 0; x < 256; ++x) {
        for (int c = 0; c < 3; ++c) {
          float s = 0.0f;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
              s += frame_ref[((y * 2 + dy) * 512 + (x * 2 + dx)) * 3 + c];
          ref_cmp[(y * 256 + x) * 3 + c] = s * 0.25f;
        }
      }
    }
  }
  if (out->bytes != ref_cmp.size() * sizeof(float)) {
    return absl::InvalidArgumentError("Output tensor 'frame' size mismatch.");
  }

  double mse = 0.0, sum_abs = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < ref_cmp.size(); ++i) {
    double diff = static_cast<double>(out->data.f[i]) -
                  static_cast<double>(ref_cmp[i]);
    double ad = std::abs(diff);
    if (ad > max_abs) max_abs = ad;
    sum_abs += ad;
    mse += diff * diff;
  }
  mse /= static_cast<double>(ref_cmp.size());
  double mae_255 = (sum_abs / static_cast<double>(ref_cmp.size())) * 255.0;
  double max_255 = max_abs * 255.0;
  double psnr = (mse > 1e-15) ? (10.0 * std::log10(1.0 / mse)) : 99.99;

  // Save raw float output so we can inspect rendered PNG if desired
  std::ofstream ofs(ref_dir + "/frame_tflite.bin", std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(out->data.f), out->bytes);

  std::cout << absl::StrFormat(
      "VERIFY CPU (%dx%d TFLite XNNPACK): %.1f ms | PSNR=%.2f dB | "
      "MAE=%.4f/255 | max_err=%.4f/255\n",
      out_h, out_w, ms, psnr, mae_255, max_255);
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: imtalker_renderer_export <renderer.safetensors> "
                 "<renderer_decode.tflite> [--fp16-weights] [--verify <ref_dir>]\n";
    return 1;
  }
  std::string verify_dir;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fp16-weights") g_fp16_weights = true;
    else if (a == "--int8-weights") g_int8_weights = true;
    else if (a == "--fast-mobile") g_fast_mobile = true;
    else if (a == "--fast-synth64") g_fast_synth64 = true;
    else if (a == "--skip-swin256") g_skip_swin256 = true;
    else if (a == "--skip-swin256-attn") g_skip_swin256_attn = true;
    else if (a == "--fast-swin256") g_fast_swin256 = true;
    else if (a == "--fast-swin128") g_fast_swin128 = true;
    else if (a == "--skip-swin128") g_skip_swin128 = true;
    else if (a == "--skip-swin128-b1") g_skip_swin128_b1 = true;
    else if (a == "--skip-res256-extra") g_skip_res256_extra = true;
    else if (a == "--out-256") g_out_256 = true;
    else if (a == "--verify" && i + 1 < argc) verify_dir = argv[++i];
  }
  absl::Status status = ExportRendererDecode(argv[1], argv[2]);
  if (!status.ok()) {
    std::cerr << "Export failed: " << status << "\n";
    return 1;
  }
  if (!verify_dir.empty()) {
    status = VerifyTflite(argv[2], verify_dir);
    if (!status.ok()) {
      std::cerr << "Verify failed: " << status << "\n";
      return 1;
    }
  }
  return 0;
}

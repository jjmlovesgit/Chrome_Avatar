// IMTalker Wav2Vec2 (wav2vec2-base-960h) + audio_projection + pose_projection
// exporter for fixed 2.0-second windows (32,000 samples @ 16 kHz -> 50 frames).
// Loads directly from wav2vec2-base-960h/model.safetensors and
// generator.safetensors via LiteRT's SafetensorLoader (zero Python converter),
// strictly rank <= 4 ops (16-slice pos_conv_embed, zero Split) for 100% WebGPU,
// serializes wav2vec2_2s.tflite via ModelFactory::Save, and optionally verifies
// numerical parity against PyTorch golden tensors.

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
#include "absl/types/span.h"
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
using litert::tensor::BatchMatMul;
using litert::tensor::Concatenation;
using litert::tensor::Conv2D;
using litert::tensor::FullyConnected;
using litert::tensor::Gelu;
using litert::tensor::kPaddingValid;
using litert::tensor::Logistic;
using litert::tensor::Mean;
using litert::tensor::ModelFactory;
using litert::tensor::Mul;
using litert::tensor::OwningCpuBuffer;
using litert::tensor::Pad;
using litert::tensor::Reshape;
using litert::tensor::ResizeBilinear;
using litert::tensor::Rsqrt;
using litert::tensor::Slice;
using litert::tensor::Softmax;
using litert::tensor::Sub;
using litert::tensor::Tensor;
using litert::tensor::TensorHandle;
using litert::tensor::TfLiteMixinTag;
using litert::tensor::Transpose;
using litert::tensor::Type;
using litert::tensor::examples::SafetensorLoader;

using T = Tensor<TfLiteMixinTag>;

bool g_fp16_weights = false;

T MakeConstF32(const std::string& name, std::vector<int> shape,
               const std::vector<float>& data) {
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
                             absl::string_view name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, loader.GetTensorInfo(name));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> data,
                              loader.ReadTensorAs<float>(name));
  std::vector<int> shape;
  shape.reserve(info.shape.size());
  for (int64_t d : info.shape) shape.push_back(static_cast<int>(d));
  return MakeConstF32(std::string(name), std::move(shape), data);
}

// PyTorch Conv1d [OutC, InC, K] -> TFLite Conv2D NHWC filter [OutC, 1, K, InC]
std::vector<float> OikToO1ki(const std::vector<float>& src, int out_c,
                             int in_c, int k) {
  std::vector<float> dst(src.size());
  for (int o = 0; o < out_c; ++o) {
    for (int i = 0; i < in_c; ++i) {
      for (int kw = 0; kw < k; ++kw) {
        size_t s_idx = ((size_t)o * in_c + i) * k + kw;
        size_t d_idx = ((size_t)o * k + kw) * in_c + i;
        dst[d_idx] = src[s_idx];
      }
    }
  }
  return dst;
}

absl::StatusOr<T> ApplyLinear(const SafetensorLoader& loader,
                              const std::string& pfx, T x) {
  LRT_TENSOR_ASSIGN_OR_RETURN(T w, LoadRawF32(loader, pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b, LoadRawF32(loader, pfx + ".bias"));
  return FullyConnected(x, w, b);
}

// LayerNorm with affine weight & bias over axis (eps=1e-5)
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

T Silu(T x) { return Mul(x, Logistic(x)); }

absl::Status ExportWav2Vec2(const std::string& w2v_safetensors_path,
                            const std::string& gen_safetensors_path,
                            const std::string& output_tflite_path) {
  std::cout << "Loading wav2vec2 safetensors: " << w2v_safetensors_path << "\n";
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader w2v,
                              SafetensorLoader::Load(w2v_safetensors_path));
  std::cout << "Loading generator safetensors: " << gen_safetensors_path
            << "\n";
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader gen,
                              SafetensorLoader::Load(gen_safetensors_path));

  constexpr int kSamples = 32000;  // 2.0 s @ 16 kHz
  constexpr int kSeqLen = 50;      // 2.0 s @ 25 fps
  constexpr int kHidden = 768;
  constexpr int kHeads = 12;
  constexpr int kHeadDim = kHidden / kHeads;  // 64
  constexpr int kLayers = 12;

  // Graph Inputs
  T pcm_in({.name = "pcm_in", .type = Type::kFP32, .shape = {1, kSamples}});
  T pose_in(
      {.name = "pose_in", .type = Type::kFP32, .shape = {1, kSeqLen, 3}});

  // =========================================================================
  // 1. 7-layer Conv1D Feature Extractor (as NHWC Conv2D with H=1)
  // =========================================================================
  const int kernels[7] = {10, 3, 3, 3, 3, 2, 2};
  const int strides[7] = {5, 2, 2, 2, 2, 2, 2};
  const int in_ch[7] = {1, 512, 512, 512, 512, 512, 512};

  T x = Reshape(pcm_in, {1, 1, kSamples, 1});
  for (int i = 0; i < 7; ++i) {
    const std::string pfx =
        absl::StrCat("feature_extractor.conv_layers.", i);
    LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> w_raw,
                                w2v.ReadTensorAs<float>(pfx + ".conv.weight"));
    std::vector<float> w_nhwc = OikToO1ki(w_raw, 512, in_ch[i], kernels[i]);
    T w_t = MakeConstF32(pfx + ".w", {512, 1, kernels[i], in_ch[i]}, w_nhwc);
    T b_t = MakeZeroBias(pfx + ".zero_b", 512);
    x = Conv2D(x, w_t, b_t, /*stride_h=*/1, /*stride_w=*/strides[i],
               kPaddingValid);

    if (i == 0) {
      // Wav2Vec2GroupNormConvLayer: GroupNorm(512, 512) = InstanceNorm over time axis 2
      LRT_TENSOR_ASSIGN_OR_RETURN(
          T gn_w, LoadRawF32(w2v, pfx + ".layer_norm.weight"));
      LRT_TENSOR_ASSIGN_OR_RETURN(
          T gn_b, LoadRawF32(w2v, pfx + ".layer_norm.bias"));
      T mean = Mean(x, std::vector<int>{2}, /*keep_dims=*/true);
      T diff = Sub(x, mean);
      T var = Mean(Mul(diff, diff), std::vector<int>{2}, /*keep_dims=*/true);
      x = Add(Mul(Mul(diff, Rsqrt(Add(var, 1e-5f))), gn_w), gn_b);
    }
    x = Gelu(x, /*approximate=*/false);
  }
  // x is now [1, 1, 99, 512]

  // =========================================================================
  // 2. Linear Interpolation 99 -> 50 (align_corners=True)
  // =========================================================================
  x = ResizeBilinear(x, {1, kSeqLen}, /*align_corners=*/true,
                     /*half_pixel_centers=*/false);  // [1, 1, 50, 512]
  x = Reshape(x, {1, kSeqLen, 512});

  // =========================================================================
  // 3. Feature Projection: LayerNorm(512) -> Linear(512 -> 768)
  // =========================================================================
  LRT_TENSOR_ASSIGN_OR_RETURN(
      x, ApplyLayerNorm(w2v, "feature_projection.layer_norm", x, 2));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      x, ApplyLinear(w2v, "feature_projection.projection", x));  // [1, 50, 768]

  // =========================================================================
  // 4. Positional Conv Embedding (groups=16, k=128, weight-norm folded)
  //    Lowered as 16 standard groups=1 Conv2D slices for 100% WebGPU support.
  // =========================================================================
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> wg,
      w2v.ReadTensorAs<float>("encoder.pos_conv_embed.conv.weight_g"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> wv,
      w2v.ReadTensorAs<float>("encoder.pos_conv_embed.conv.weight_v"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      std::vector<float> pos_bias,
      w2v.ReadTensorAs<float>("encoder.pos_conv_embed.conv.bias"));

  // Reconstruct weight_norm along dim=2 (k=0..127): wv shape [768, 48, 128]
  constexpr int kPosK = 128;
  constexpr int kGroups = 16;
  constexpr int kGrpCh = kHidden / kGroups;  // 48
  std::vector<float> pos_w(wv.size());
  for (int kw = 0; kw < kPosK; ++kw) {
    double sum_sq = 0.0;
    for (int o = 0; o < kHidden; ++o) {
      for (int i = 0; i < kGrpCh; ++i) {
        double v = static_cast<double>(wv[((size_t)o * kGrpCh + i) * kPosK + kw]);
        sum_sq += v * v;
      }
    }
    double scale = static_cast<double>(wg[kw]) / (std::sqrt(sum_sq) + 1e-12);
    for (int o = 0; o < kHidden; ++o) {
      for (int i = 0; i < kGrpCh; ++i) {
        size_t idx = ((size_t)o * kGrpCh + i) * kPosK + kw;
        pos_w[idx] = static_cast<float>(static_cast<double>(wv[idx]) * scale);
      }
    }
  }

  // Pad time axis (64 left, 63 right) -> length 177 -> k=128 VALID -> length 50
  // (exact match to padding=64 + SamePadLayer dropping the final sample)
  T x_4d = Reshape(x, {1, 1, kSeqLen, kHidden});
  T pos_paddings({.name = "pos_conv_pad",
                  .type = Type::kI32,
                  .shape = {4, 2},
                  .buffer = OwningCpuBuffer::Copy<Type::kI32>(
                      {0, 0, 0, 0, 64, 63, 0, 0})});
  T x_pad = Pad(x_4d, pos_paddings);  // [1, 1, 177, 768]

  std::vector<T> pos_groups;
  pos_groups.reserve(kGroups);
  for (int g = 0; g < kGroups; ++g) {
    T xg = Slice(x_pad, {0, 0, 0, g * kGrpCh}, {1, 1, 177, kGrpCh});
    std::vector<float> wg_slice(kGrpCh * kGrpCh * kPosK);
    std::memcpy(wg_slice.data(),
                pos_w.data() + (size_t)g * kGrpCh * kGrpCh * kPosK,
                wg_slice.size() * sizeof(float));
    std::vector<float> wg_nhwc = OikToO1ki(wg_slice, kGrpCh, kGrpCh, kPosK);
    std::vector<float> bg_slice(pos_bias.begin() + g * kGrpCh,
                                pos_bias.begin() + (g + 1) * kGrpCh);
    T wg_t = MakeConstF32(absl::StrCat("pos_conv.g", g, ".w"),
                          {kGrpCh, 1, kPosK, kGrpCh}, wg_nhwc);
    T bg_t = MakeConstF32(absl::StrCat("pos_conv.g", g, ".b"), {kGrpCh},
                          bg_slice);
    pos_groups.push_back(
        Conv2D(xg, wg_t, bg_t, /*stride_h=*/1, /*stride_w=*/1, kPaddingValid));
  }
  T pos = Reshape(Concatenation(absl::MakeSpan(pos_groups), /*axis=*/3),
                  {1, kSeqLen, kHidden});
  pos = Gelu(pos, /*approximate=*/false);
  LRT_TENSOR_ASSIGN_OR_RETURN(
      x, ApplyLayerNorm(w2v, "encoder.layer_norm", Add(x, pos), 2));

  // =========================================================================
  // 5. 12x Wav2Vec2EncoderLayer (Post-LN, 12 heads, dim=768, ffn=3072)
  // =========================================================================
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));  // 0.125
  for (int l = 0; l < kLayers; ++l) {
    const std::string pfx = absl::StrCat("encoder.layers.", l);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T q, ApplyLinear(w2v, pfx + ".attention.q_proj", x));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T k, ApplyLinear(w2v, pfx + ".attention.k_proj", x));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T v, ApplyLinear(w2v, pfx + ".attention.v_proj", x));

    q = Transpose(Reshape(q, {1, kSeqLen, kHeads, kHeadDim}), {0, 2, 1, 3});
    k = Transpose(Reshape(k, {1, kSeqLen, kHeads, kHeadDim}), {0, 2, 1, 3});
    v = Transpose(Reshape(v, {1, kSeqLen, kHeads, kHeadDim}), {0, 2, 1, 3});

    T attn = Softmax(
        BatchMatMul(Mul(q, q_scale), k, /*adj_x=*/false, /*adj_y=*/true));
    T ctx = Reshape(Transpose(BatchMatMul(attn, v), {0, 2, 1, 3}),
                    {1, kSeqLen, kHidden});
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T attn_out, ApplyLinear(w2v, pfx + ".attention.out_proj", ctx));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        x, ApplyLayerNorm(w2v, pfx + ".layer_norm", Add(x, attn_out), 2));

    LRT_TENSOR_ASSIGN_OR_RETURN(
        T ff, ApplyLinear(w2v, pfx + ".feed_forward.intermediate_dense", x));
    ff = Gelu(ff, /*approximate=*/false);
    LRT_TENSOR_ASSIGN_OR_RETURN(
        ff, ApplyLinear(w2v, pfx + ".feed_forward.output_dense", ff));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        x, ApplyLayerNorm(w2v, pfx + ".final_layer_norm", Add(x, ff), 2));
  }

  // =========================================================================
  // 6. Baked audio_projection (768->32) & pose_projection (3->32)
  // =========================================================================
  LRT_TENSOR_ASSIGN_OR_RETURN(T a_lin, ApplyLinear(gen, "audio_projection.0", x));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T a_ln, ApplyLayerNorm(gen, "audio_projection.1", a_lin, 2));
  T a_now = Silu(a_ln);
  a_now.SetName("a_now");

  LRT_TENSOR_ASSIGN_OR_RETURN(T p_lin,
                              ApplyLinear(gen, "pose_projection.0", pose_in));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T p_ln, ApplyLayerNorm(gen, "pose_projection.1", p_lin, 2));
  T pose_emb = Silu(p_ln);
  pose_emb.SetName("pose_emb");

  // Serialize .tflite
  ModelFactory factory;
  std::vector<TensorHandle> sig_inputs = {pcm_in, pose_in};
  std::vector<TensorHandle> sig_outputs = {a_now, pose_emb};
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

void PrintStats(const char* label, const float* got,
                const std::vector<float>& ref) {
  double max_abs = 0.0, sum_abs = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double a = static_cast<double>(got[i]);
    double b = static_cast<double>(ref[i]);
    double err = std::abs(a - b);
    if (err > max_abs) max_abs = err;
    sum_abs += err;
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  double mean_abs = sum_abs / static_cast<double>(ref.size());
  double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
  std::cout << absl::StrFormat(
      "  [%s] max_abs_err=%.6f | mean_abs_err=%.6f | cos_sim=%.8f\n", label,
      max_abs, mean_abs, cos_sim);
}

absl::Status VerifyTflite(const std::string& tflite_path,
                          const std::string& ref_dir) {
  std::vector<float> pcm_ref, pose_ref, a_ref, p_ref;
  if (!ReadBinFloats(ref_dir + "/pcm_in.bin", pcm_ref) ||
      !ReadBinFloats(ref_dir + "/pose_in.bin", pose_ref) ||
      !ReadBinFloats(ref_dir + "/a_now_ref.bin", a_ref) ||
      !ReadBinFloats(ref_dir + "/pose_emb_ref.bin", p_ref)) {
    return absl::NotFoundError("Failed to read ref_wav2vec2 .bin files.");
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

  TfLiteTensor* pcm_t = runner->input_tensor("pcm_in");
  TfLiteTensor* pose_t = runner->input_tensor("pose_in");
  std::memcpy(pcm_t->data.f, pcm_ref.data(), pcm_t->bytes);
  std::memcpy(pose_t->data.f, pose_ref.data(), pose_t->bytes);

  auto t0 = std::chrono::steady_clock::now();
  if (runner->Invoke() != kTfLiteOk) return absl::InternalError("Invoke failed.");
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << absl::StrFormat("VERIFY CPU (TFLite XNNPACK): %.2f ms\n", ms);
  PrintStats("a_now", runner->output_tensor("a_now")->data.f, a_ref);
  PrintStats("pose_emb", runner->output_tensor("pose_emb")->data.f, p_ref);
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: imtalker_wav2vec2_export <wav2vec2.safetensors> "
                 "<generator.safetensors> <wav2vec2_2s.tflite> "
                 "[--fp16-weights] [--verify <ref_dir>]\n";
    return 1;
  }
  std::string verify_dir;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fp16-weights") g_fp16_weights = true;
    else if (a == "--verify" && i + 1 < argc) verify_dir = argv[++i];
  }
  absl::Status status = ExportWav2Vec2(argv[1], argv[2], argv[3]);
  if (!status.ok()) {
    std::cerr << "Export failed: " << status << "\n";
    return 1;
  }
  if (!verify_dir.empty()) {
    status = VerifyTflite(argv[3], verify_dir);
    if (!status.ok()) {
      std::cerr << "Verify failed: " << status << "\n";
      return 1;
    }
  }
  return 0;
}

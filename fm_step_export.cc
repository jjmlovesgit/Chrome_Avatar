// IMTalker Flow-Matching Transformer (FMGenerator / FMT) single-step exporter.
// Loads weights directly from checkpoints/generator.safetensors using LiteRT's
// SafetensorLoader, authors the exact 8-layer DiT graph with LiteRT Tensor API,
// serializes fm_step.tflite via ModelFactory::Save, and optionally runs CPU
// verification against PyTorch golden tensors (litert_poc/ref_data/).

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
using litert::tensor::BatchMatMul;
using litert::tensor::Concatenation;
using litert::tensor::Cos;
using litert::tensor::FullyConnected;
using litert::tensor::Gelu;
using litert::tensor::Logistic;
using litert::tensor::Mean;
using litert::tensor::ModelFactory;
using litert::tensor::Mul;
using litert::tensor::OwningCpuBuffer;
using litert::tensor::Reshape;
using litert::tensor::Rsqrt;
using litert::tensor::Sin;
using litert::tensor::Slice;
using litert::tensor::Softmax;
using litert::tensor::Split;
using litert::tensor::Sub;
using litert::tensor::Tensor;
using litert::tensor::TensorHandle;
using litert::tensor::TfLiteMixinTag;
using litert::tensor::Transpose;
using litert::tensor::Type;
using litert::tensor::examples::SafetensorLoader;

using T = Tensor<TfLiteMixinTag>;

bool g_fp16_weights = false;

// Helper: load a float32 weight tensor from SafetensorLoader into T.
absl::StatusOr<T> LoadWeight(const SafetensorLoader& loader,
                             absl::string_view name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, loader.GetTensorInfo(name));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> data,
                              loader.ReadTensorAs<float>(name));
  std::vector<int> shape;
  shape.reserve(info.shape.size());
  for (int64_t d : info.shape) {
    shape.push_back(static_cast<int>(d));
  }
  if (g_fp16_weights && data.size() >= 4096) {
    std::vector<litert::tensor::fp16_t> h(data.size());
    for (size_t i = 0; i < data.size(); ++i) h[i] = litert::tensor::fp16_t(data[i]);
    T raw({.name = std::string(name) + "_fp16",
           .type = Type::kFP16,
           .shape = std::move(shape),
           .buffer = OwningCpuBuffer::Copy<Type::kFP16>(h)});
    return litert::tensor::Dequantize(raw);
  }
  return T({.name = std::string(name),
            .type = Type::kFP32,
            .shape = std::move(shape),
            .buffer = OwningCpuBuffer::Copy<Type::kFP32>(data)});
}

// Helper: SiLU(x) = x * sigmoid(x)
T Silu(T x) { return Mul(x, Logistic(x)); }

// Helper: LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) on axis 2
// Input shape: [B, L, D]
T LayerNormNoAffine(T x) {
  T mean = Mean(x, std::vector<int>{2}, /*keep_dims=*/true);
  T diff = Sub(x, mean);
  T var = Mean(Mul(diff, diff), std::vector<int>{2}, /*keep_dims=*/true);
  T inv_std = Rsqrt(Add(var, 1e-6f));
  return Mul(diff, inv_std);
}

// Helper: framewise_modulate(x, shift, scale) = x * (1 + scale) + shift
T FramewiseModulate(T x, T shift, T scale) {
  return Add(Mul(x, Add(scale, 1.0f)), shift);
}

absl::Status ExportFmStep(const std::string& safetensors_path,
                          const std::string& output_tflite_path) {
  std::cout << "Loading safetensors: " << safetensors_path << "\n";
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(safetensors_path));

  constexpr int kBatch = 2;  // [uncond, cond] for batched CFG
  constexpr int kSeqLen = 60;  // 10 prev + 50 clip frames
  constexpr int kPrevLen = 10;
  constexpr int kClipLen = 50;
  constexpr int kDimH = 512;
  constexpr int kNumHeads = 8;
  constexpr int kHeadDim = kDimH / kNumHeads;  // 64
  constexpr int kFreqHalf = 128;               // 256 / 2
  constexpr int kDepth = 8;

  // Graph Inputs
  T t_in({.name = "t", .type = Type::kFP32, .shape = {1}});
  T x_in({.name = "x_in", .type = Type::kFP32, .shape = {kBatch, kSeqLen, 64}});
  T cond_in(
      {.name = "cond_in", .type = Type::kFP32, .shape = {kBatch, kSeqLen, 32}});
  T cfg_scale({.name = "cfg_scale", .type = Type::kFP32, .shape = {1}});

  // 1. TimestepEmbedder: sinusoidal frequencies + 2-layer SiLU MLP
  std::vector<float> freqs_vec(kFreqHalf);
  const double neg_log_max_period = -std::log(10000.0);
  for (int i = 0; i < kFreqHalf; ++i) {
    freqs_vec[i] = static_cast<float>(
        std::exp(neg_log_max_period * static_cast<double>(i) / kFreqHalf));
  }
  T freqs({.name = "t_freqs",
           .type = Type::kFP32,
           .shape = {1, kFreqHalf},
           .buffer = OwningCpuBuffer::Copy<Type::kFP32>(freqs_vec)});

  T t_args = Mul(Reshape(t_in, {1, 1}), freqs);
  T t_freq = Concatenation({Cos(t_args), Sin(t_args)}, /*axis=*/1);  // [1, 256]

  LRT_TENSOR_ASSIGN_OR_RETURN(T w_t0,
                              LoadWeight(loader, "fmt.t_embedder.mlp.0.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b_t0,
                              LoadWeight(loader, "fmt.t_embedder.mlp.0.bias"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T w_t2,
                              LoadWeight(loader, "fmt.t_embedder.mlp.2.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b_t2,
                              LoadWeight(loader, "fmt.t_embedder.mlp.2.bias"));

  T t_emb = FullyConnected(Silu(FullyConnected(t_freq, w_t0, b_t0)), w_t2, b_t2);
  t_emb = Reshape(t_emb, {1, 1, kDimH});

  // 2. Sequence & Condition Embeddings
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T w_x, LoadWeight(loader, "fmt.x_embedder.proj.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b_x,
                              LoadWeight(loader, "fmt.x_embedder.proj.bias"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T pos_embed, LoadWeight(loader, "fmt.pos_embed"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T w_c,
                              LoadWeight(loader, "fmt.c_embedder.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b_c, LoadWeight(loader, "fmt.c_embedder.bias"));

  T x = Add(FullyConnected(x_in, w_x, b_x), pos_embed);  // [2, 60, 512]
  T c = Add(FullyConnected(cond_in, w_c, b_c), t_emb);   // [2, 60, 512]

  // 3. Static Attention Bias from alignment_mask (window expansion = 5)
  // Matches enc_dec_mask(60, 60, frame_width=1, expansion=5):
  // allowed when max(0, i-5) <= j < i+6; masked (-1e4, FP16-safe) otherwise.
  std::vector<float> attn_bias_vec(kSeqLen * kSeqLen, -1e4f);
  for (int i = 0; i < kSeqLen; ++i) {
    int j_start = std::max(0, i - 5);
    int j_end = std::min(kSeqLen, i + 6);
    for (int j = j_start; j < j_end; ++j) {
      attn_bias_vec[i * kSeqLen + j] = 0.0f;
    }
  }
  T attn_bias({.name = "alignment_attn_bias",
               .type = Type::kFP32,
               .shape = {1, 1, kSeqLen, kSeqLen},
               .buffer = OwningCpuBuffer::Copy<Type::kFP32>(attn_bias_vec)});

  // 4. 8x FMTBlock (adaLN-Zero DiT blocks)
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));  // 0.125
  for (int i = 0; i < kDepth; ++i) {
    const std::string pfx = absl::StrCat("fmt.blocks.", i, ".");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T w_ada, LoadWeight(loader, pfx + "adaLN_modulation.1.weight"));
    LRT_TENSOR_ASSIGN_OR_RETURN(
        T b_ada, LoadWeight(loader, pfx + "adaLN_modulation.1.bias"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T w_qkv,
                                LoadWeight(loader, pfx + "attn.qkv.weight"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T b_qkv,
                                LoadWeight(loader, pfx + "attn.qkv.bias"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T w_proj,
                                LoadWeight(loader, pfx + "attn.proj.weight"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T b_proj,
                                LoadWeight(loader, pfx + "attn.proj.bias"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T w_fc1,
                                LoadWeight(loader, pfx + "mlp.fc1.weight"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T b_fc1,
                                LoadWeight(loader, pfx + "mlp.fc1.bias"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T w_fc2,
                                LoadWeight(loader, pfx + "mlp.fc2.weight"));
    LRT_TENSOR_ASSIGN_OR_RETURN(T b_fc2,
                                LoadWeight(loader, pfx + "mlp.fc2.bias"));

    // adaLN modulation: 6 slices of [2, 60, 512] (avoids Split op on WebGPU)
    T mod = FullyConnected(Silu(c), w_ada, b_ada);
    T shift_msa = Slice(mod, {0, 0, 0 * kDimH}, {kBatch, kSeqLen, kDimH});
    T scale_msa = Slice(mod, {0, 0, 1 * kDimH}, {kBatch, kSeqLen, kDimH});
    T gate_msa = Slice(mod, {0, 0, 2 * kDimH}, {kBatch, kSeqLen, kDimH});
    T shift_mlp = Slice(mod, {0, 0, 3 * kDimH}, {kBatch, kSeqLen, kDimH});
    T scale_mlp = Slice(mod, {0, 0, 4 * kDimH}, {kBatch, kSeqLen, kDimH});
    T gate_mlp = Slice(mod, {0, 0, 5 * kDimH}, {kBatch, kSeqLen, kDimH});

    // Self-Attention (strictly <= 4D tensors for 100% WebGPU acceleration)
    T h_attn = FramewiseModulate(LayerNormNoAffine(x), shift_msa, scale_msa);
    T qkv = FullyConnected(h_attn, w_qkv, b_qkv);  // [2, 60, 1536]
    T q_flat = Slice(qkv, {0, 0, 0 * kDimH}, {kBatch, kSeqLen, kDimH});
    T k_flat = Slice(qkv, {0, 0, 1 * kDimH}, {kBatch, kSeqLen, kDimH});
    T v_flat = Slice(qkv, {0, 0, 2 * kDimH}, {kBatch, kSeqLen, kDimH});
    T q = Transpose(Reshape(q_flat, {kBatch, kSeqLen, kNumHeads, kHeadDim}),
                    {0, 2, 1, 3});  // [2, 8, 60, 64]
    T k = Transpose(Reshape(k_flat, {kBatch, kSeqLen, kNumHeads, kHeadDim}),
                    {0, 2, 1, 3});
    T v = Transpose(Reshape(v_flat, {kBatch, kSeqLen, kNumHeads, kHeadDim}),
                    {0, 2, 1, 3});

    T scores = BatchMatMul(Mul(q, q_scale), k, /*adj_x=*/false, /*adj_y=*/true);
    scores = Add(scores, attn_bias);
    T attn_w = Softmax(scores);
    T ctx = BatchMatMul(attn_w, v);  // [2, 8, 60, 64]
    ctx = Reshape(Transpose(ctx, {0, 2, 1, 3}), {kBatch, kSeqLen, kDimH});
    T attn_out = FullyConnected(ctx, w_proj, b_proj);
    x = Add(x, Mul(gate_msa, attn_out));

    // MLP (GELU approximate="tanh")
    T h_mlp = FramewiseModulate(LayerNormNoAffine(x), shift_mlp, scale_mlp);
    T mlp_h = Gelu(FullyConnected(h_mlp, w_fc1, b_fc1), /*approximate=*/true);
    T mlp_out = FullyConnected(mlp_h, w_fc2, b_fc2);
    x = Add(x, Mul(gate_mlp, mlp_out));
  }

  // 5. Final Decoder
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T w_dec_ada, LoadWeight(loader, "fmt.decoder.adaLN_modulation.1.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T b_dec_ada, LoadWeight(loader, "fmt.decoder.adaLN_modulation.1.bias"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T w_dec_lin, LoadWeight(loader, "fmt.decoder.linear.weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      T b_dec_lin, LoadWeight(loader, "fmt.decoder.linear.bias"));

  T dec_mod = FullyConnected(Silu(c), w_dec_ada, b_dec_ada);
  T dec_shift = Slice(dec_mod, {0, 0, 0}, {kBatch, kSeqLen, kDimH});
  T dec_scale = Slice(dec_mod, {0, 0, kDimH}, {kBatch, kSeqLen, kDimH});
  T dec_h = FramewiseModulate(LayerNormNoAffine(x), dec_shift, dec_scale);
  T out_full = FullyConnected(dec_h, w_dec_lin, b_dec_lin);  // [2, 60, 32]

  // 6. CFG blend + slice generated 50 frames [:, 10:60, :]
  T uncond = Slice(out_full, {0, kPrevLen, 0}, {1, kClipLen, 32});
  T cond_out = Slice(out_full, {1, kPrevLen, 0}, {1, kClipLen, 32});
  T cfg_3d = Reshape(cfg_scale, {1, 1, 1});
  T velocity = Add(uncond, Mul(cfg_3d, Sub(cond_out, uncond)));
  velocity.SetName("velocity");

  // 7. Serialize to .tflite
  ModelFactory factory;
  std::vector<TensorHandle> sig_inputs = {t_in, x_in, cond_in, cfg_scale};
  std::vector<TensorHandle> sig_outputs = {velocity};
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
  return static_cast<bool>(
      ifs.read(reinterpret_cast<char*>(out.data()), sz));
}

absl::Status VerifyTflite(const std::string& tflite_path,
                          const std::string& ref_dir) {
  std::vector<float> t_ref, x_ref, cond_ref, cfg_ref, vel_ref;
  if (!ReadBinFloats(ref_dir + "/t.bin", t_ref) ||
      !ReadBinFloats(ref_dir + "/x_in.bin", x_ref) ||
      !ReadBinFloats(ref_dir + "/cond_in.bin", cond_ref) ||
      !ReadBinFloats(ref_dir + "/cfg_scale.bin", cfg_ref) ||
      !ReadBinFloats(ref_dir + "/vel_ref.bin", vel_ref)) {
    return absl::NotFoundError("Failed to read golden .bin files from " +
                               ref_dir);
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
  if (!model) {
    return absl::InternalError("Failed to load tflite model: " + tflite_path);
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk ||
      !interpreter) {
    return absl::InternalError("InterpreterBuilder failed.");
  }
  interpreter->SetNumThreads(8);

  tflite::SignatureRunner* runner =
      interpreter->GetSignatureRunner("serving_default");
  if (!runner) {
    return absl::InternalError("Signature 'serving_default' not found.");
  }
  if (runner->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("AllocateTensors failed.");
  }

  auto copy_in = [&](const char* name, const std::vector<float>& src) {
    TfLiteTensor* tensor = runner->input_tensor(name);
    if (!tensor || tensor->bytes != src.size() * sizeof(float)) {
      std::cerr << "Input mismatch for " << name << "\n";
      return false;
    }
    std::memcpy(tensor->data.f, src.data(), tensor->bytes);
    return true;
  };

  if (!copy_in("t", t_ref) || !copy_in("x_in", x_ref) ||
      !copy_in("cond_in", cond_ref) || !copy_in("cfg_scale", cfg_ref)) {
    return absl::InvalidArgumentError("Input tensor size mismatch.");
  }

  auto t0 = std::chrono::steady_clock::now();
  if (runner->Invoke() != kTfLiteOk) {
    return absl::InternalError("Invoke failed.");
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  const TfLiteTensor* out = runner->output_tensor("velocity");
  if (!out || out->bytes != vel_ref.size() * sizeof(float)) {
    return absl::InvalidArgumentError("Output tensor 'velocity' mismatch.");
  }

  double max_abs = 0.0, sum_abs = 0.0, dot = 0.0, n_a = 0.0, n_b = 0.0;
  for (size_t i = 0; i < vel_ref.size(); ++i) {
    double a = static_cast<double>(out->data.f[i]);
    double b = static_cast<double>(vel_ref[i]);
    double err = std::abs(a - b);
    if (err > max_abs) max_abs = err;
    sum_abs += err;
    dot += a * b;
    n_a += a * a;
    n_b += b * b;
  }
  double mean_abs = sum_abs / static_cast<double>(vel_ref.size());
  double cos_sim = dot / (std::sqrt(n_a) * std::sqrt(n_b) + 1e-12);

  std::cout << absl::StrFormat(
      "VERIFY CPU (TFLite XNNPACK): %.2f ms | max_abs_err=%.6f | "
      "mean_abs_err=%.6f | cos_sim=%.8f\n",
      ms, max_abs, mean_abs, cos_sim);
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: fm_step_export <generator.safetensors> "
                 "<fm_step.tflite> [--fp16-weights] [--verify <ref_dir>]\n";
    return 1;
  }
  std::string verify_dir;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fp16-weights") g_fp16_weights = true;
    else if (a == "--verify" && i + 1 < argc) verify_dir = argv[++i];
  }
  absl::Status status = ExportFmStep(argv[1], argv[2]);
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

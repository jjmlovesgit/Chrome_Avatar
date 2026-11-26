// Mobile 256x256 Student Talking-Head Renderer (student_renderer_256_fp16w.tflite)
// Direct C++ LiteRT Tensor API exporter from student_renderer_256.safetensors.
// Inputs:
//   t_c:     [1, 32]           IMTalker 32-D motion token
//   src_img: [1, 256, 256, 3]  Avatar portrait in [0, 1] (NHWC)
// Output:
//   frame:   [1, 256, 256, 3]  Animated portrait in [0, 1] (NHWC)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
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
using litert::tensor::Concatenation;
using litert::tensor::Conv2D;
using litert::tensor::FullyConnected;
using litert::tensor::kPaddingSame;
using litert::tensor::kPaddingValid;
using litert::tensor::LeakyRelu;
using litert::tensor::Logistic;
using litert::tensor::Maximum;
using litert::tensor::Minimum;
using litert::tensor::ModelFactory;
using litert::tensor::Mul;
using litert::tensor::OwningCpuBuffer;
using litert::tensor::Pad;
using litert::tensor::Reshape;
using litert::tensor::ResizeBilinear;
using litert::tensor::Sub;
using litert::tensor::Tensor;
using litert::tensor::TensorHandle;
using litert::tensor::TfLiteMixinTag;
using litert::tensor::Type;
using litert::tensor::examples::SafetensorLoader;

using T = Tensor<TfLiteMixinTag>;

bool g_fp16_weights = false;

T MakeConstF32(const std::string& name, std::vector<int> shape,
               const std::vector<float>& data) {
  if (g_fp16_weights && data.size() >= 1024) {
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

absl::StatusOr<T> ApplyLinear(const SafetensorLoader& loader,
                              const std::string& pfx, T x) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& w_info, loader.GetTensorInfo(pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> w_data, loader.ReadTensorAs<float>(pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> b_data, loader.ReadTensorAs<float>(pfx + ".bias"));
  std::vector<int> w_shape;
  for (int64_t d : w_info.shape) w_shape.push_back(static_cast<int>(d));
  T w = MakeConstF32(pfx + ".weight", std::move(w_shape), w_data);
  T b = MakeConstF32(pfx + ".bias", {static_cast<int>(b_data.size())}, b_data);
  return FullyConnected(x, w, b);
}

absl::StatusOr<T> ApplyConv(const SafetensorLoader& loader,
                            const std::string& pfx, T x, int stride) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& w_info, loader.GetTensorInfo(pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> w_oihw, loader.ReadTensorAs<float>(pfx + ".weight"));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> b_data, loader.ReadTensorAs<float>(pfx + ".bias"));
  const int out_c = static_cast<int>(w_info.shape[0]);
  const int in_c = static_cast<int>(w_info.shape[1]);
  const int kh = static_cast<int>(w_info.shape[2]);
  const int kw = static_cast<int>(w_info.shape[3]);
  std::vector<float> w_ohwi = OihwToOhwi(w_oihw, out_c, in_c, kh, kw);
  T w_t = MakeConstF32(pfx + ".w_nhwc", {out_c, kh, kw, in_c}, w_ohwi);
  T b_t = MakeConstF32(pfx + ".bias", {out_c}, b_data);
  if (stride == 1) {
    return Conv2D(x, w_t, b_t, /*stride_h=*/1, /*stride_w=*/1, kPaddingSame);
  }
  // Exact PyTorch Conv2d(k=3, stride=2, padding=1) parity on even H/W
  T paddings({.name = pfx + ".pad1",
              .type = Type::kI32,
              .shape = {4, 2},
              .buffer = OwningCpuBuffer::Copy<Type::kI32>({0, 0, 1, 1, 1, 1, 0, 0})});
  T xp = Pad(x, paddings);
  return Conv2D(xp, w_t, b_t, /*stride_h=*/stride, /*stride_w=*/stride, kPaddingValid);
}

absl::StatusOr<T> ApplyFiLMResBlock(const SafetensorLoader& loader,
                                    const std::string& pfx, T x, T cond,
                                    int ch) {
  LRT_TENSOR_ASSIGN_OR_RETURN(T gamma, ApplyLinear(loader, pfx + ".film_gamma", cond));
  LRT_TENSOR_ASSIGN_OR_RETURN(T beta, ApplyLinear(loader, pfx + ".film_beta", cond));
  T g4 = Reshape(Add(gamma, 1.0f), {1, 1, 1, ch});
  T b4 = Reshape(beta, {1, 1, 1, ch});
  LRT_TENSOR_ASSIGN_OR_RETURN(T h, ApplyConv(loader, pfx + ".conv1", x, 1));
  h = LeakyRelu(Add(Mul(h, g4), b4), 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyConv(loader, pfx + ".conv2", h, 1));
  return LeakyRelu(Add(x, h), 0.2f);
}

absl::Status ExportStudent(const std::string& safetensors_path,
                           const std::string& output_tflite_path) {
  std::cout << "Loading student safetensors: " << safetensors_path << "\n";
  LRT_TENSOR_ASSIGN_OR_RETURN(SafetensorLoader loader,
                              SafetensorLoader::Load(safetensors_path));

  T t_c({.name = "t_c", .type = Type::kFP32, .shape = {1, 32}});
  T src_img({.name = "src_img", .type = Type::kFP32, .shape = {1, 256, 256, 3}});

  // Motion MLP
  LRT_TENSOR_ASSIGN_OR_RETURN(T cond, ApplyLinear(loader, "m_fc1", t_c));
  cond = LeakyRelu(cond, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(cond, ApplyLinear(loader, "m_fc2", cond));
  cond = LeakyRelu(cond, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(cond, ApplyLinear(loader, "m_fc3", cond));
  cond = LeakyRelu(cond, 0.2f);

  // Static CoordConv [1, 256, 256, 2] (x, y in [-1, 1])
  std::vector<float> xy_nhwc(256 * 256 * 2);
  for (int y = 0; y < 256; ++y) {
    float yf = -1.0f + 2.0f * static_cast<float>(y) / 255.0f;
    for (int x = 0; x < 256; ++x) {
      float xf = -1.0f + 2.0f * static_cast<float>(x) / 255.0f;
      xy_nhwc[(y * 256 + x) * 2 + 0] = xf;
      xy_nhwc[(y * 256 + x) * 2 + 1] = yf;
    }
  }
  T xy_t = MakeConstF32("xy_grid_nhwc", {1, 256, 256, 2}, xy_nhwc);

  // Encoder
  T x_in = Concatenation({src_img, xy_t}, /*axis=*/3);  // [1, 256, 256, 5]
  LRT_TENSOR_ASSIGN_OR_RETURN(T x0, ApplyConv(loader, "enc0", x_in, 1));
  x0 = LeakyRelu(x0, 0.2f);

  LRT_TENSOR_ASSIGN_OR_RETURN(T x1, ApplyConv(loader, "enc1_a", x0, 2));
  x1 = LeakyRelu(x1, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(x1, ApplyConv(loader, "enc1_b", x1, 1));
  x1 = LeakyRelu(x1, 0.2f);

  LRT_TENSOR_ASSIGN_OR_RETURN(T x2, ApplyConv(loader, "enc2_a", x1, 2));
  x2 = LeakyRelu(x2, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(x2, ApplyConv(loader, "enc2_b", x2, 1));
  x2 = LeakyRelu(x2, 0.2f);

  LRT_TENSOR_ASSIGN_OR_RETURN(T x3, ApplyConv(loader, "enc3_a", x2, 2));
  x3 = LeakyRelu(x3, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(x3, ApplyConv(loader, "enc3_b", x3, 1));
  x3 = LeakyRelu(x3, 0.2f);

  LRT_TENSOR_ASSIGN_OR_RETURN(T x4, ApplyConv(loader, "enc4_a", x3, 2));
  x4 = LeakyRelu(x4, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(x4, ApplyConv(loader, "enc4_b", x4, 1));
  x4 = LeakyRelu(x4, 0.2f);

  // Bottleneck (16x16, 192ch)
  LRT_TENSOR_ASSIGN_OR_RETURN(T h, ApplyFiLMResBlock(loader, "mid1", x4, cond, 192));
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyFiLMResBlock(loader, "mid2", h, cond, 192));

  // Decoder (32 -> 64 -> 128 -> 256)
  h = ResizeBilinear(h, {32, 32}, /*align_corners=*/false, /*half_pixel_centers=*/true);
  h = Concatenation({h, x3}, /*axis=*/3);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyConv(loader, "dec3_proj", h, 1));
  h = LeakyRelu(h, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyFiLMResBlock(loader, "dec3_res", h, cond, 128));

  h = ResizeBilinear(h, {64, 64}, false, true);
  h = Concatenation({h, x2}, /*axis=*/3);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyConv(loader, "dec2_proj", h, 1));
  h = LeakyRelu(h, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyFiLMResBlock(loader, "dec2_res", h, cond, 64));

  h = ResizeBilinear(h, {128, 128}, false, true);
  h = Concatenation({h, x1}, /*axis=*/3);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyConv(loader, "dec1_proj", h, 1));
  h = LeakyRelu(h, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyFiLMResBlock(loader, "dec1_res", h, cond, 32));

  h = ResizeBilinear(h, {256, 256}, false, true);
  h = Concatenation({h, x0}, /*axis=*/3);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyConv(loader, "dec0_proj", h, 1));
  h = LeakyRelu(h, 0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(T g0, ApplyLinear(loader, "dec0_film_gamma", cond));
  LRT_TENSOR_ASSIGN_OR_RETURN(T b0, ApplyLinear(loader, "dec0_film_beta", cond));
  h = LeakyRelu(Add(Mul(h, Reshape(Add(g0, 1.0f), {1, 1, 1, 16})),
                    Reshape(b0, {1, 1, 1, 16})),
                0.2f);
  LRT_TENSOR_ASSIGN_OR_RETURN(h, ApplyConv(loader, "dec0_conv", h, 1));
  h = LeakyRelu(h, 0.2f);

  // Clean Sigmoid RGB head
  LRT_TENSOR_ASSIGN_OR_RETURN(T logits, ApplyConv(loader, "head", h, 1));
  T frame = Logistic(logits);
  frame.SetName("frame");

  ModelFactory factory;
  LRT_TENSOR_RETURN_IF_ERROR(
      factory.AddSignature({t_c, src_img}, {frame}, "serving_default"));
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
  std::vector<float> tc, src, ref;
  if (!ReadBinFloats(ref_dir + "/t_c.bin", tc) ||
      !ReadBinFloats(ref_dir + "/src_img.bin", src) ||
      !ReadBinFloats(ref_dir + "/frame_ref.bin", ref)) {
    return absl::NotFoundError("Missing reference bins in " + ref_dir);
  }
  auto model = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
  if (!model) return absl::InternalError("Failed to load " + tflite_path);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("InterpreterBuilder failed.");
  }
  interpreter->SetNumThreads(12);
  tflite::SignatureRunner* runner = interpreter->GetSignatureRunner("serving_default");
  if (!runner || runner->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("SignatureRunner setup failed.");
  }
  std::memcpy(runner->input_tensor("t_c")->data.f, tc.data(), tc.size() * sizeof(float));
  std::memcpy(runner->input_tensor("src_img")->data.f, src.data(), src.size() * sizeof(float));

  // Warmup + timed run
  runner->Invoke();
  auto t0 = std::chrono::steady_clock::now();
  if (runner->Invoke() != kTfLiteOk) return absl::InternalError("Invoke failed.");
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  const TfLiteTensor* out = runner->output_tensor("frame");
  double mse = 0.0, sum_abs = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = std::abs(static_cast<double>(out->data.f[i]) - static_cast<double>(ref[i]));
    if (d > max_abs) max_abs = d;
    sum_abs += d;
    mse += d * d;
  }
  mse /= static_cast<double>(ref.size());
  double psnr = (mse > 1e-15) ? (10.0 * std::log10(1.0 / mse)) : 99.99;
  std::cout << absl::StrFormat(
      "VERIFY STUDENT CPU (256x256): %.1f ms | PSNR=%.2f dB | MAE=%.4f/255 | max_err=%.4f/255\n",
      ms, psnr, (sum_abs / ref.size()) * 255.0, max_abs * 255.0);
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: imtalker_student_export <student.safetensors> <out.tflite> [--fp16-weights] [--verify <ref_dir>]\n";
    return 1;
  }
  std::string verify_dir;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fp16-weights") g_fp16_weights = true;
    else if (a == "--verify" && i + 1 < argc) verify_dir = argv[++i];
  }
  absl::Status status = ExportStudent(argv[1], argv[2]);
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

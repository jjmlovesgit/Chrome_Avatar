# IMTalker â†’ LiteRT Tensor API â†’ LiteRT.js (WebGPU) Build Tracker

Goal: Author IMTalker's 3 static inference graphs in C++ using Google's **LiteRT Tensor API** (loading our existing `.safetensors` directly), export `.tflite` models via `ModelFactory::Save`, verify numerical parity against PyTorch, and run real-time talking-head rendering client-side in Chrome via **`@litertjs/core` (WebGPU)**.

---

## Environment & Toolchain

- **OS:** Windows 11 (native build first; WSL2 available as fallback)
- **Compiler / Build:** Visual Studio 2022 Community (`MSVC`), `CMake 4.2.0`, `Ninja`
- **GPU:** NVIDIA RTX 5090 (32 GB)
- **LiteRT Repo Path:** `C:\Projects\LiteRT` (`https://github.com/google-ai-edge/LiteRT`)
- **Source Python Repo:** `C:\Projects\FlashIMTalkerV2`

---

## Checkpoint Inventory (Verified from `.safetensors` Headers)

| File | Disk Size (FP32) | Tensors / Params | Runtime Browser Role | Est. FP16 `.tflite` |
| --- | --- | --- | --- | --- |
| `checkpoints/generator.safetensors` | **148.22 MB** | 110 tensors (`38.86M` params, `109 F32 + 1 BOOL`) | **Graph 2:** `fm_step.tflite` (8-layer DiT Transformer, fixed `seq_len=60`) | **~74 MB** |
| `checkpoints/renderer.safetensors` | **543.09 MB** | 828 tensors (`142.32M` params, `748 F32 + 80 I64`) | **Graph 3:** `renderer_decode.tflite` (**277.84 MB** FP32 runtime subset after stripping encoders + VGG loss!) | **~139 MB** |
| `checkpoints/wav2vec2-base-960h/model.safetensors` | **360.02 MB** | 211 tensors (`94.37M` params, `211 F32`) | **Graph 1:** `wav2vec2_2s.tflite` (fixed 2.0 s @ 16 kHz â†’ `[1, 50, 32]`) | **~180 MB** |
| `driver_video/blink_delta.pt` | **2.7 KB** | `[9, 32]` F32 | Shipped as `1.15 KB` raw Float32Array in JS for speaking blinks | **~1 KB** |

### Renderer Module Breakdown (`renderer.safetensors`)
- **Kept in Browser (`renderer_decode.tflite` = 277.84 MB FP32 / ~139 MB FP16):**
  - `frame_decoder` (`SynthesisNetwork`): `197.02 MB`
  - `latent_token_decoder` (`MotionDecoder`): `66.42 MB`
  - `imt` (`CrossAttention` + `GuidedResampler`): `9.27 MB`
  - `adapt` (`IdentidyAdaptive`): `5.13 MB`
- **Eliminated from Browser (265.14 MB FP32 saved!):**
  - `dense_feature_encoder`: `133.88 MB` *(precomputed once per portrait offline â†’ shipped as avatar cache)*
  - `latent_token_encoder`: `81.87 MB` *(precomputed once per portrait offline â†’ shipped as avatar cache)*
  - `criterion_vgg`: `49.39 MB` *(training-only VGG perceptual loss)*

**Total Browser Download (FP16):** **~393 MB** (down from `1,051 MB` FP32 â€” a **2.7Ã— reduction** before INT8 quantization).

---

## Milestones & Checklist

### Milestone 0 â€” Setup & Toolchain Verification
- [x] Inspect all 3 `.safetensors` checkpoints and verify tensor keys/shapes/sizes
- [x] Create `build.md` progress tracker
- [x] Clone `google-ai-edge/LiteRT` to `C:\Projects\LiteRT`
- [x] Inspect `tensor/CMakeLists.txt` and available Tensor API ops / `ModelFactory::Save`
- [x] Dump PyTorch golden reference tensors for `FlowMatchingTransformer` (`litert_poc/ref_data/`, 94 tensors loaded with `missing=0, unexpected=0`)
- [x] Build LiteRT Tensor API (`litert_tensor_api_core` + `litert_tensor_api_tflite_backend`) natively on Windows with CMake + VS 2022 MSVC 19.44 â€” **0 errors, no WSL required!**

### Milestone 1 â€” `fm_step.tflite` (`generator.safetensors`, Proof of Concept) â€” COMPLETE âœ…
- [x] Read exact forward pass of `FlowMatchingTransformer` ([generator/FMT.py](generator/FMT.py)) and `FMGenerator.sample()` ([generator/FM.py](generator/FM.py))
- [x] Author `fm_step` in C++ ([litert_poc/fm_step_export.cc](fm_step_export.cc)) using LiteRT Tensor API + `SafetensorLoader` + built-in `tflite::Interpreter` `--verify` parity check
- [x] Build `imtalker_fm_export.exe` and serialize `litert_poc/fm_step.tflite` (`148.20 MB`) via `ModelFactory::Save` (first-try zero-error MSVC build!)
- [x] Verify numerical parity against PyTorch golden tensors: **`cos_sim = 1.00000000`**, `mean_abs_err = 0.000440`, `max_abs_err = 0.009754` (on range `[-56.86, +45.30]`, `std = 13.93`), **20.42 ms/step on native CPU (XNNPACK)**
- [x] Run 6-step Euler ODE loop in Chrome with `@litertjs/core@2.6.0` (WebGPU on NVIDIA Blackwell) â€” **100% WebGPU-accelerated (0 unsupported ops!)**:

| Backend (`fm_step.tflite`, 148.20 MB) | Compile | Parity vs PyTorch (`cos_sim` / MAE) | 6-Step ODE (2.0 s clip) | Avg / Step | Speedup |
| --- | --- | --- | --- | --- | --- |
| **Chrome WebGPU (FP32)** | **110.7 ms** | **`1.00000000`** / `0.000455` | **26.34 ms** | **4.39 ms** | **75.9Ã— realtime** |
| **Chrome WebGPU (FP16)** | **124.4 ms** | `0.99803799` / `0.621211` | **28.12 ms** | **4.69 ms** | **71.1Ã— realtime** |
| **Native Windows CPU (TFLite XNNPACK)** | â€” | **`1.00000000`** / `0.000440` | **~122.5 ms** | **20.42 ms** | **16.3Ã— realtime** |
| **Chrome WASM (CPU)** | **20.5 ms** | **`1.00000000`** / `0.000448` | **854.36 ms** | **142.39 ms** | **2.3Ã— realtime** |

### Milestone 2 â€” `renderer_decode.tflite` (`renderer.safetensors` + Avatar Cache) â€” COMPLETE âœ…
- [x] Dump offline Puck NHWC face cache (`litert_poc/ref_renderer/`, **9.17 MB total**) + PyTorch golden 512Ã—512 frame (`frame_ref.png`), and verify exact `GuidedResampler` algebraic identity (`max_abs_err = 0.000000000` on both 128Ã—128 and 256Ã—256 levels!)
- [x] Author `renderer_decode` (`batch=1`, 100% NHWC, rank â‰¤ 4) in C++ ([litert_poc/renderer_decode_export.cc](renderer_decode_export.cc)):
  - `IdentidyAdaptive` (5 `EqualLinear` layers)
  - `MotionDecoder` (13 `StyledConv`s with algebraic `demod = rsqrt(W_sq @ styleÂ² + 1e-8)` + static `Conv2D` / `TransposeConv` + `Blur` `pad=(1,1)` `DepthwiseConv2D`)
  - `IMT` (`StandardUnifiedAttention` at `8..64`, bit-exact `GuidedResampler` `ArgMax(kI32) + Gather + ResizeNearestNeighbor` at `128, 256`)
  - `SynthesisNetwork` (all `BatchNorm2d` folded into `Conv2D` at load time, `UnifiedTransformerBlock` at `16..64`, pure-4D `UnifiedSwinBlock` `w=8` at `128, 256`, channel-reordered `DepthToSpace(2)` + `Sigmoid` RGB head)
- [x] Build `imtalker_renderer_export.exe` (first-try zero-error MSVC build!), export `litert_poc/renderer_decode.tflite` (**285.66 MB** FP32, down from `543.09 MB`), and verify CPU parity: **`PSNR = 112.96 dB`**, `MAE = 0.0003 / 255` (`306.8 ms/frame` on CPU)
- [x] Benchmark 512Ã—512 live `<canvas>` rendering in Chrome (`@litertjs/core@2.6.0`, WebGPU on NVIDIA Blackwell) â€” **100% WebGPU-accelerated (0 unsupported ops!)**:

| Backend (`renderer_decode.tflite`, 285.66 MB) | Compile | Parity vs PyTorch (`PSNR` / MAE) | Live 512Ã—512 End-to-End (w/ readback + `<canvas>`) | Real-Time Margin (vs 25 fps) |
| --- | --- | --- | --- | --- |
| **Chrome WebGPU (FP32)** | **530.3 ms** | **`118.23 dB`** / **`0.0001/255`** | **`23.53 ms/frame` (`42.5 FPS`)** | **1.70Ã— realtime** |
| **Chrome WebGPU (FP16)** | **538.8 ms** | `42.72 dB` / `0.9496/255` | **`19.64 ms/frame` (`50.9 FPS`)** | **2.04Ã— realtime** |
| **Native Windows CPU (TFLite XNNPACK)** | â€” | **`112.96 dB`** / `0.0003/255` | `306.8 ms/frame` (`3.3 FPS`) | `0.13Ã—` |

### Milestone 3 â€” `wav2vec2_2s.tflite` + End-to-End Browser Demo â€” COMPLETE âœ…
- [x] Author `wav2vec2_2s` + `audio_projection` + `pose_projection` (`[1, 32000]` + `[1, 50, 3]` â†’ `[1, 50, 32]` + `[1, 50, 32]`) in C++ ([litert_poc/wav2vec2_export.cc](wav2vec2_export.cc)) â€” 7Ã— Conv1D + GroupNorm + `ResizeBilinear(align_corners=true)` + 16-slice weight-norm `pos_conv_embed` + 12 post-LN Transformer layers
- [x] Build `imtalker_wav2vec2_export.exe` (3rd consecutive zero-error MSVC first-try build!), export `litert_poc/wav2vec2_2s.tflite` (**360.20 MB** FP32), and verify parity vs PyTorch:
  - **`a_now`**: **`cos_sim = 1.00000000`**, `mean_abs_err = 0.000002`, `max_abs_err = 0.000045`
  - **`pose_emb`**: **`cos_sim = 1.00000000`**, `mean_abs_err = 0.000000`, `max_abs_err = 0.000000`
- [x] Precompute 9.17 MB NHWC face caches for **both Puck & Gretchen** (`litert_poc/avatars/{puck,gretchen}/`), export `blink_delta.bin` (`[9, 32]`), and synthesize real Windows SAPI demo WAVs (`puck_demo.wav`, `gretchen_demo.wav`)
- [x] Build standalone interactive Chrome WebGPU Studio ([litert_poc/web/demo.html](web/demo.html)) with instant Puck â†” Gretchen switching, Web Audio sync, speaking blinks, calm head sway, audio file upload, and mic recording
- [x] Verify full 3-model end-to-end live playback in Chrome WebGPU (`nvidia blackwell`, **0 unsupported ops across all 3 models!**):

| Pipeline Stage (Chrome WebGPU FP32, `@litertjs/core@2.6.0`) | Model Size (FP32) | Steady-State WebGPU Time | Real-Time Speedup | Parity vs PyTorch |
| --- | --- | --- | --- | --- |
| **1. `wav2vec2_2s.tflite`** (2.0 s audio + pose) | `360.20 MB` | **`12.6 â€“ 15.0 ms` / 2.0 s** | **~145Ã— realtime** | **`cos_sim = 1.00000000`** (`MAE = 2e-6`) |
| **2. `fm_step.tflite`** (6-step ODE, 2.0 s motion) | `148.20 MB` | **`18.6 â€“ 20.4 ms` / 2.0 s** | **~103Ã— realtime** | **`cos_sim = 1.00000000`** (`MAE = 4.5e-4`) |
| **3. `renderer_decode.tflite`** (512Ã—512 live `<canvas>`) | `285.66 MB` | **`23.16 ms / frame` (`43.2 FPS`)** | **1.73Ã— realtime** (vs 25 fps) | **`PSNR = 118.23 dB`** (`MAE = 0.0001/255`) |
| **End-to-End Test (`gretchen_demo.wav`, 6.77 s)** | `794.1 MB` total | **170 / 170 live frames (0 drops)** | **Real-time 25 fps synced** | **0 console errors / 0 GPU warnings** |

### Milestone 4 â€” Conversational Studio (Chrome Gemini Nano + Voice/Text Q&A + Warm SAPI TTS) â€” COMPLETE âœ…
- [x] Build persistent warm Windows SAPI 5 COM worker ([litert_poc/web/sapi_worker.ps1](web/sapi_worker.ps1)) pre-warmed with `Andrew (Natural HD)` (Puck) & `Jenny(Natural)` (Gretchen) (`63` voices detected)
- [x] Add `POST /api/tts` (warm SAPI 16 kHz WAV) and `POST /api/llm` (optional local LM Studio `127.0.0.1:1234` fallback proxy) to [litert_poc/web/server.mjs](web/server.mjs)
- [x] Upgrade [litert_poc/web/demo.html](web/demo.html) with:
  - Chrome Built-in **Gemini Nano (`LanguageModel` Prompt API)** with `availability()` check, user-gesture download button, streaming replies, and automatic Chromium-stub detection â†’ Local LLM fallback
  - Text Q&A input + **ðŸŽ¤ Ask by Voice** (`SpeechRecognition` live mic transcription â†’ auto-send)
  - Shared multi-turn Puck & Gretchen conversation memory + `"Hey Puck"` / `"Hey Gretchen"` automatic avatar & voice routing
- [x] Verified live end-to-end conversational flow in browser (`LanguageModel` â†’ warm SAPI `Jenny(Natural)` â†’ `wav2vec2_2s` + `fm_step` + `renderer_decode` â†’ **503 / 503 consecutive live 512Ã—512 frames at `23.3 ms/frame` = `43.0 FPS`**, zero dropped frames)

### Milestone 5 â€” 100% Browser-Local Pipeline (Auto-Loading WebLLM + SAPI + Live Combined VRAM Audit) â€” COMPLETE âœ…
- [x] Installed `@mlc-ai/web-llm@0.2.89` locally in [litert_poc/web/package.json](web/package.json) (self-contained Rollup ESM bundle, zero bundler required)
- [x] Switched [litert_poc/web/server.mjs](web/server.mjs) COEP header from `require-corp` â†’ `credentialless` so cross-origin WebLLM model shard fetches from Hugging Face CDN succeed while keeping `SharedArrayBuffer` enabled
- [x] Upgraded [litert_poc/web/demo.html](web/demo.html) with:
  - **Zero-click automatic in-browser WebGPU LLM** (`Llama-3.2-1B-Instruct-q4f16_1-MLC` default, plus `gemma-2-2b-it-q4f16_1-MLC-1k` and `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` options) â€” auto-downloads once (`664 MB`) into browser Cache API, then boots from cache in **`1.5 s`** in **any** WebGPU browser (no Chrome flags, no user-gesture gate, no LM Studio needed!)
  - Seamless **Chrome Built-in Gemini Nano** option (auto-selected when branded Google Chrome reports `"available"`, with automatic WebLLM failover if unbranded Chromium's echo stub is ever detected)
  - Live **`Precision: FP32 / FP16`** WebGPU selector (also via `?precision=fp16`)
- [x] Verified full 100% browser-local E2E runs (`?autotest=webllm`) in both **FP32** and **FP16** with **0 console errors / 0 WebGPU warnings**, and measured exact live dedicated GPU VRAM (`Get-Counter '\GPU Process Memory(*)\Dedicated Usage'` + `nvidia-smi` on RTX 5090):

| Full Conversational Avatar Stack (Measured Live on RTX 5090) | LLM Engine | TTS Engine | Avatar 512Ã—512 Engine | Process Dedicated VRAM | **Net AI VRAM** (above idle) | Reduction vs Desktop |
| --- | --- | --- | --- | --- | --- | --- |
| **Desktop PyTorch + LM Studio (Baseline)** | LM Studio `llama-server` (`5.71 GB`) | Edge/SAPI (`0 GB`) | Python PyTorch IMTalker + Whisper (`10.57 GB`) | `16.28 GB` | **`16.28 GB`** | `1.0Ã—` |
| **Chrome 157 Gemini Nano v3 (Default) + IMTalker** | Chrome `OnDeviceModelService` (`nano_v3_gpu_component`, **`3.10 GB`**) | Warm Windows SAPI (`0 GB`) | LiteRT.js WebGPU in `chrome gpu-process` (`4.95 GB`, `+4.80 GB` above idle) | **`8.05 GB`** (`5,069 + 3,177 MB`) | **`~7.90 GB`** | **`2.1Ã—` less VRAM** |
| **100% Browser-Local Studio â€” WebLLM FP32** | WebLLM `Llama-3.2-1B-q4f16_1` (`~0.88 GB`, **835 ms** warm) | Warm Windows SAPI (`0 GB`, **292 ms**) | LiteRT.js WebGPU FP32 (`~3.12 GB`, **`43.6 FPS`**, `118.2 dB`) | **`4.11 GB`** (`4,204.4 MB`) | **`4.00 GB`** | **`4.1Ã—` less VRAM** |
| **100% Browser-Local Studio â€” WebLLM FP16** | WebLLM `Llama-3.2-1B-q4f16_1` (`~0.88 GB`, **835 ms** warm) | Warm Windows SAPI (`0 GB`, **365 ms**) | LiteRT.js WebGPU FP16 (`~2.12 GB`, `42.7 dB` visually lossless) | **`3.11 GB`** (`3,180.3 MB`) | **`3.00 GB`** | **`5.4Ã—` less VRAM** |

*(Browser GPU process idle baseline = `0.11 GB`. Cold WebLLM download = `76 s` for `664 MB` on first visit; subsequent reloads boot the entire studio from browser cache in **`5.76 s`**, with WebLLM loading in **`1.5 s`** and generating replies in **`835 ms`**!)*

---

## Key Learnings Log

- **2026-11-25 â€” No `grid_sample` or `torch.qr` in active `IMTRenderer`:** Although `ToFlow` (`F.grid_sample`) and `Direction` (`torch.qr`) exist in `renderer/lia_resblocks.py`, `IMTRenderer` in `renderer/models.py` never imports or calls them. Active ops are strictly standard Convs, `StyledConv`, `SDPA` attention, `Swin` windowed attention, `GuidedResampler` (`argmax` + `gather`), and `PixelShuffle(2)` (`DepthToSpace`).
- **2026-11-25 â€” `FMGenerator` is already static-windowed:** `wav2vec_sec=2.0`, `fps=25`, `num_prev_frames=10` â†’ fixed `seq_len=60` (`10` context + `50` generated frames) with a 6-step Euler ODE loop. Zero dynamic-shape surgery needed for `fm_step.tflite`.
- **2026-11-25 â€” 49% of `renderer.safetensors` is unnecessary in browser:** `criterion_vgg` (`49.39 MB`), `dense_feature_encoder` (`133.88 MB`), and `latent_token_encoder` (`81.87 MB`) account for `265.14 MB` of `543.09 MB`. Stripping them leaves `277.84 MB` FP32 (~`139 MB` FP16) for `renderer_decode.tflite`.
- **2026-11-25 â€” `generator.safetensors` schema is ultra-clean:** 110 tensors total (`38.86M` params): 8 standard DiT/adaLN-Zero blocks (`dim_h=512`, `num_heads=8`, `mlp=2048`), `pos_embed [1,60,512]`, `alignment_mask BOOL [60,60]`, `t_embedder`, `x_embedder`, `c_embedder`, `decoder`, and 4 tiny condition projections.
- **2026-11-25 â€” Native Windows CMake + VS 2022 MSVC 19.44 works cleanly (no WSL needed):** Pass `-DLITERT_ENABLE_NPU=OFF -DLITERT_ENABLE_GPU=OFF -DLITERT_BUILD_TESTS=OFF -DLITERT_DISABLE_KLEIDIAI=ON` when configuring `LiteRT/litert` to skip multi-GB NPU vendor SDK downloads. Incremental rebuilds of `imtalker_fm_export.exe` take ~5 seconds.
- **2026-11-25 â€” Critical WebGPU Delegate Rules (learned live in Chrome):**
  1. **Keep ALL tensors rank â‰¤ 4:** A single 5D `Reshape({2, 60, 3, 8, 64})` caused `RESHAPE: Tensor dimensions must be less than 5.` and forced 341/419 ops onto CPU fallback. Replacing it with 3D `Slice` + 4D `Reshape({2, 60, 8, 64})` made **100% of ops WebGPU-native**.
  2. **Avoid `Split` on WebGPU:** `SPLIT` failed with `Unknown layout.` on ML Drift WebGPU â€” replacing all `Split` calls with explicit 3D `Slice` achieved 0 unsupported ops.
  3. **Use FP16-safe attention mask (`-1e4f`, not `-1e9f`):** `-1e9f` overflows IEEE-754 FP16 (`max = 65504`) and produced `NaN` in `gpuOptions: { precision: 'fp16' }`. Switching to `-1e4f` fixed FP16 immediately (`cos_sim = 0.9980`).
  4. **Load LiteRT.js with `{ jspi: true }`:** Avoids `ReferenceError: Asyncify is not defined` if any mixed GPU/CPU fallback ever occurs.
  5. **WebGPU FP32 is the sweet spot on RTX 5090:** **`26.34 ms` for 6 ODE steps (75.9Ã— realtime!)** with exact **`cos_sim = 1.00000000`** (`MAE = 0.000455`) â€” actually slightly faster than FP16 (`28.12 ms`) with zero precision loss!
- **2026-11-25 â€” Milestone 2 (`renderer_decode.tflite`) Breakthroughs:**
  1. **Zero-copy `GuidedResampler` algebraic collapse (`0.000000000` error):** For `k_top_samples=1`, `GuidedResampler` is bit-for-bit identical to precomputing `AvgPool2D(f_r)` offline (`f_r_4_pool [4096, 64]` + `f_r_5_pool [4096, 32]`), then running `ArgMax(kI32) â†’ Gather â†’ ResizeNearestNeighbor`! Cuts avatar face cache to **9.17 MB** and eliminates 1M+ per-frame gather indices.
  2. **Pure-4D `SwinUnifiedAttention` + algebraic `ModulatedConv2d` demod:** Writing `window_partition`/`window_reverse` in two 4D steps (`{nH, w, nW, w*C} â†” {nH, nW, w, w*C}`), `CyclicRollNHWC` via 4D `Slice+Concatenation`, and `demod = rsqrt(W_sq @ styleÂ² + 1e-8)` achieved **0 unsupported GPU ops** across the entire 512Ã—512 StyleGAN/Swin synthesis network!
  3. **Exact `Blur` padding in `ModulatedConv2d(upsample=True)`:** `p = (4 - 2) - (3 - 1) = 0` â†’ `pad = (1, 1)` (not `(2, 1)`), yielding `(2H+1) + 2 - 4 + 1 = 2H` and **`118.23 dB PSNR`** on Chrome WebGPU FP32!
  4. **PyTorch `PixelShuffle(2)` vs TFLite `DepthToSpace(2)`:** Reordering the 12 output channels of `final_conv.1` from `(3, 2, 2)` â†’ `(2, 2, 3)` at C++ load time matches `PixelShuffle(2)` bit-for-bit with zero extra runtime ops.
  5. **Full 512Ã—512 Browser FPS Exceeds Real-Time (25 fps) in FP32:** **`42.5 FPS` (`23.53 ms/frame`) in FP32 (`118.23 dB PSNR`)** and **`50.9 FPS` (`19.64 ms/frame`) in FP16 (`42.72 dB PSNR`)** â€” including full GPU sync, readback, and `<canvas>` paint! Combined with `fm_step.tflite` (`26.3 ms` per 2.0 s clip = `0.53 ms/frame` amortized), **real-time 512Ã—512 @ 25 fps IMTalker in Chrome WebGPU is 100% proven!**
- **2026-11-25 â€” Milestone 3 (`wav2vec2_2s.tflite` + E2E Browser Studio) Breakthroughs:**
  1. **16-slice `pos_conv_embed` + `(64, 63)` asymmetric padding:** Folding Hugging Face weight-norm (`weight_g / ||weight_v||`) at load time and lowering `groups=16, k=128` into 16 standard `groups=1` `Conv2D` slices (with `(64, 63)` padding matching `SamePadLayer`) ran **100% on WebGPU (`12.6â€“15.0 ms` per 2.0 s chunk!)** with **`cos_sim = 1.00000000` (`MAE = 0.000002`)**!
  2. **WASM32 2 GB Heap Limit & Clean Stage Swapping:** Compiling all 3 FP32 `.tflite` flatbuffers (`794.1 MB` total) simultaneously inside a single WASM32 instance hits WASM32's ~2 GB linear memory cap (`Aborted()`). Keeping the raw `.tflite` `Uint8Array`s in JS heap (outside WASM32) and staging `wav2vec2_2s` (`217 ms` recompile) â†” `renderer_decode` (`427 ms` recompile) around `fm_step` keeps WASM32 heap comfortably under ~1 GB and ran **170/170 live frames at 43.2 FPS** cleanly. *(Optional next optimization: store model weights as FP16/INT8 to cut total flatbuffer size from `794 MB` â†’ `~397 MB` so all 3 stay compiled in WASM32 simultaneously with 0 ms stage swap!)*
- **2026-11-25 â€” Milestone 4 (Gemini Nano + SAPI + Live Q&A) Learnings:**
  1. **Brand-checked Chrome vs. unbranded Chromium for Gemini Nano:** In unbranded Chromium (such as embedded webviews), `LanguageModel.availability()` reports `"downloadable"` and completes `"100%"`, then returns an echo stub (`"On-device model is not available in Chromium..."`). `demo.html` detects this stub automatically and falls back cleanly to **Local LLM (`127.0.0.1:1234`)**, while opening `http://127.0.0.1:8765/` in branded **Google Chrome** runs real on-device **Gemini Nano**.
  2. **Live browser WebGPU VRAM:** Measured **`~3.4 GB` VRAM per active tab** in FP32 (`IMT` `64Ã—64` attention is `512 MB` FP32 + `Swin` + Dawn buffer pool) vs `10.57 GB` for the Python PyTorch + Whisper server. Using warm Windows SAPI (`sapi_worker.ps1`) keeps TTS VRAM overhead at **0 GB**.
- **2026-11-25 â€” Milestone 5 (100% Browser-Local WebLLM + Combined FP32 vs FP16 VRAM) Learnings:**
  1. **COEP `credentialless` unlocks WebLLM CDN downloads alongside `SharedArrayBuffer`:** Switching `Cross-Origin-Embedder-Policy` in `server.mjs` from `require-corp` to `credentialless` kept cross-origin isolation active for LiteRT.js WASM while letting `@mlc-ai/web-llm` fetch model shards directly from Hugging Face CDN with zero CORS/COEP errors.
  2. **Automatic caching & sub-second warm WebGPU LLM:** First-visit download of `Llama-3.2-1B-Instruct-q4f16_1-MLC` (`664 MB`) takes ~76 s and persists in the browser Cache API. Subsequent reloads boot WebLLM in **`1.5 s`** (entire studio ready in **`5.76 s`**) and generate full replies in **`835 ms`** (~86 tok/s on WebGPU).
  3. **Exact live combined VRAM on RTX 5090 (WebLLM + SAPI + 3-model LiteRT.js IMTalker):**
     - **FP32 mode:** **`4.11 GB`** total browser GPU process (`~4.00 GB` net AI VRAM: `~0.88 GB` WebLLM + `~3.12 GB` IMTalker + `0 GB` SAPI) â€” **`4.1Ã—` smaller** than the `16.28 GB` desktop Python + LM Studio stack.
     - **FP16 mode:** **`3.11 GB`** total browser GPU process (`~3.00 GB` net AI VRAM: `~0.88 GB` WebLLM + `~2.12 GB` IMTalker + `0 GB` SAPI) â€” **`5.4Ã—` smaller** than desktop, saving **`1.00 GB`** over FP32 with visually indistinguishable `42.7 dB` 512Ã—512 output.
     - **Further VRAM headroom if needed later:** Exporting `.tflite` weights directly as FP16/INT8 (reducing the 3 `.tflite` flatbuffers from `794 MB` â†’ `~397 MB` FP16 or `~200 MB` INT8) would shrink GPU weight buffers further and eliminate the `wav2vec2_2s â†” renderer_decode` WASM32 stage swap.
  4. **Real Google Chrome 157 Gemini Nano v3 (`default`):** Set `Chrome Gemini Nano (Built-in Â· Default)` as the default backend in [demo.html](web/demo.html) (auto-prewarms on boot when `availability() === 'available'`, zero WebLLM download unless explicitly selected or needed as fallback). Verified live on **Google Chrome `157.0.8103.79`** with **`nano_v3_gpu_component` (`2025.8.8.1141`, `4,072.1 MB` `weights.bin`)**: Chrome spawns `chrome utility on_device_model.mojom.OnDeviceModelService` using **`3,177.3 MB` (`3.10 GB`)** dedicated GPU VRAM alongside `chrome gpu-process` (`5,069.4 MB`), for **`~7.90 GB` net VRAM** total (`2.1Ã—` smaller than desktop `16.28 GB`, vs `3.0â€“4.0 GB` net when using the smaller `1B` WebLLM).
- **2026-11-25 â€” Live Memory HUD, 3-Turn Puck â†” Gretchen Demo & Zero-Leak Flat VRAM Fix (`--fp16-weights`):**
  1. **Root cause of per-turn VRAM climb:** Swapping `renderer_decode â†” wav2vec2_2s` via `model.delete()` + `loadAndCompile()` on every turn leaked/retained old ML Drift WebGPU buffers (~`+2â€“3 GB` per turn in Chrome, reaching `9.02 GB` WebGPU while Chrome Gemini Nano stayed flat at `3.12 GB`).
  2. **Permanent fix (`--fp16-weights` in all 3 C++ LiteRT exporters):** Storing weights (`>= 4096` elems) as `Type::kFP16` + `Dequantize` cut total model size **2.0Ã—** (`794.1 MB` â†’ **`397.7 MB`**: `wav2vec2_2s_fp16w.tflite` `180.4 MB` `cos_sim=0.99999996`, `fm_step_fp16w.tflite` `74.3 MB` `cos_sim=0.999858`, `renderer_decode_fp16w.tflite` `143.1 MB` **`PSNR=63.61 dB`**) with **0 unsupported WebGPU ops**.
  3. **All 3 models now compile ONCE at boot & stay permanently resident** (zero stage swapping, zero `model.delete()`/`loadAndCompile()` per turn, plus JS `.tflite` buffers freed right after compile â†’ JS heap dropped **`0.84 GB` â†’ `0.07 GB`**!).
  4. **Verified flat VRAM across 3 live 512Ã—512 turns (`923/923` frames @ `44.2â€“44.7 FPS`):**
     - After Turn 1: **`4.16 GB`** (`Peak 4.18G`, `JS 0.07G`)
     - After Turn 2: **`4.16 GB`** (`Peak 4.18G`, `JS 0.06G`)
     - After Turn 3: **`4.16 GB`** (`Peak 4.18G`, `JS 0.08G`) â€” **`0.00 GB` drift!**
  5. **Live Top-Right Memory HUD ([vram_worker.ps1](web/vram_worker.ps1) + `GET /api/vram`):** Uses warm `Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory` (`~25 ms` poll) to show real-time `Nano`, `WebGPU`, `JS`, and `Peak` VRAM right in the top-right header.


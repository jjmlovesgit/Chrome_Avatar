# IMTalker → LiteRT Tensor API → LiteRT.js (WebGPU) Build Tracker

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
| `checkpoints/wav2vec2-base-960h/model.safetensors` | **360.02 MB** | 211 tensors (`94.37M` params, `211 F32`) | **Graph 1:** `wav2vec2_2s.tflite` (fixed 2.0 s @ 16 kHz → `[1, 50, 32]`) | **~180 MB** |
| `driver_video/blink_delta.pt` | **2.7 KB** | `[9, 32]` F32 | Shipped as `1.15 KB` raw Float32Array in JS for speaking blinks | **~1 KB** |

### Renderer Module Breakdown (`renderer.safetensors`)
- **Kept in Browser (`renderer_decode.tflite` = 277.84 MB FP32 / ~139 MB FP16):**
  - `frame_decoder` (`SynthesisNetwork`): `197.02 MB`
  - `latent_token_decoder` (`MotionDecoder`): `66.42 MB`
  - `imt` (`CrossAttention` + `GuidedResampler`): `9.27 MB`
  - `adapt` (`IdentidyAdaptive`): `5.13 MB`
- **Eliminated from Browser (265.14 MB FP32 saved!):**
  - `dense_feature_encoder`: `133.88 MB` *(precomputed once per portrait offline → shipped as avatar cache)*
  - `latent_token_encoder`: `81.87 MB` *(precomputed once per portrait offline → shipped as avatar cache)*
  - `criterion_vgg`: `49.39 MB` *(training-only VGG perceptual loss)*

**Total Browser Download (FP16):** **~393 MB** (down from `1,051 MB` FP32 — a **2.7× reduction** before INT8 quantization).

---

## Milestones & Checklist

### Milestone 0 — Setup & Toolchain Verification
- [x] Inspect all 3 `.safetensors` checkpoints and verify tensor keys/shapes/sizes
- [x] Create `build.md` progress tracker
- [x] Clone `google-ai-edge/LiteRT` to `C:\Projects\LiteRT`
- [x] Inspect `tensor/CMakeLists.txt` and available Tensor API ops / `ModelFactory::Save`
- [x] Dump PyTorch golden reference tensors for `FlowMatchingTransformer` (`litert_poc/ref_data/`, 94 tensors loaded with `missing=0, unexpected=0`)
- [x] Build LiteRT Tensor API (`litert_tensor_api_core` + `litert_tensor_api_tflite_backend`) natively on Windows with CMake + VS 2022 MSVC 19.44 — **0 errors, no WSL required!**

### Milestone 1 — `fm_step.tflite` (`generator.safetensors`, Proof of Concept) — COMPLETE ✅
- [x] Read exact forward pass of `FlowMatchingTransformer` ([generator/FMT.py](file:///c:/Projects/FlashIMTalkerV2/generator/FMT.py)) and `FMGenerator.sample()` ([generator/FM.py](file:///c:/Projects/FlashIMTalkerV2/generator/FM.py))
- [x] Author `fm_step` in C++ ([litert_poc/fm_step_export.cc](file:///c:/Projects/FlashIMTalkerV2/litert_poc/fm_step_export.cc)) using LiteRT Tensor API + `SafetensorLoader` + built-in `tflite::Interpreter` `--verify` parity check
- [x] Build `imtalker_fm_export.exe` and serialize `litert_poc/fm_step.tflite` (`148.20 MB`) via `ModelFactory::Save` (first-try zero-error MSVC build!)
- [x] Verify numerical parity against PyTorch golden tensors: **`cos_sim = 1.00000000`**, `mean_abs_err = 0.000440`, `max_abs_err = 0.009754` (on range `[-56.86, +45.30]`, `std = 13.93`), **20.42 ms/step on native CPU (XNNPACK)**
- [x] Run 6-step Euler ODE loop in Chrome with `@litertjs/core@2.6.0` (WebGPU on NVIDIA Blackwell) — **100% WebGPU-accelerated (0 unsupported ops!)**:

| Backend (`fm_step.tflite`, 148.20 MB) | Compile | Parity vs PyTorch (`cos_sim` / MAE) | 6-Step ODE (2.0 s clip) | Avg / Step | Speedup |
| --- | --- | --- | --- | --- | --- |
| **Chrome WebGPU (FP32)** | **110.7 ms** | **`1.00000000`** / `0.000455` | **26.34 ms** | **4.39 ms** | **75.9× realtime** |
| **Chrome WebGPU (FP16)** | **124.4 ms** | `0.99803799` / `0.621211` | **28.12 ms** | **4.69 ms** | **71.1× realtime** |
| **Native Windows CPU (TFLite XNNPACK)** | — | **`1.00000000`** / `0.000440` | **~122.5 ms** | **20.42 ms** | **16.3× realtime** |
| **Chrome WASM (CPU)** | **20.5 ms** | **`1.00000000`** / `0.000448` | **854.36 ms** | **142.39 ms** | **2.3× realtime** |

### Milestone 2 — `renderer_decode.tflite` (`renderer.safetensors` + Avatar Cache) — COMPLETE ✅
- [x] Dump offline Puck NHWC face cache (`litert_poc/ref_renderer/`, **9.17 MB total**) + PyTorch golden 512×512 frame (`frame_ref.png`), and verify exact `GuidedResampler` algebraic identity (`max_abs_err = 0.000000000` on both 128×128 and 256×256 levels!)
- [x] Author `renderer_decode` (`batch=1`, 100% NHWC, rank ≤ 4) in C++ ([litert_poc/renderer_decode_export.cc](file:///c:/Projects/FlashIMTalkerV2/litert_poc/renderer_decode_export.cc)):
  - `IdentidyAdaptive` (5 `EqualLinear` layers)
  - `MotionDecoder` (13 `StyledConv`s with algebraic `demod = rsqrt(W_sq @ style² + 1e-8)` + static `Conv2D` / `TransposeConv` + `Blur` `pad=(1,1)` `DepthwiseConv2D`)
  - `IMT` (`StandardUnifiedAttention` at `8..64`, bit-exact `GuidedResampler` `ArgMax(kI32) + Gather + ResizeNearestNeighbor` at `128, 256`)
  - `SynthesisNetwork` (all `BatchNorm2d` folded into `Conv2D` at load time, `UnifiedTransformerBlock` at `16..64`, pure-4D `UnifiedSwinBlock` `w=8` at `128, 256`, channel-reordered `DepthToSpace(2)` + `Sigmoid` RGB head)
- [x] Build `imtalker_renderer_export.exe` (first-try zero-error MSVC build!), export `litert_poc/renderer_decode.tflite` (**285.66 MB** FP32, down from `543.09 MB`), and verify CPU parity: **`PSNR = 112.96 dB`**, `MAE = 0.0003 / 255` (`306.8 ms/frame` on CPU)
- [x] Benchmark 512×512 live `<canvas>` rendering in Chrome (`@litertjs/core@2.6.0`, WebGPU on NVIDIA Blackwell) — **100% WebGPU-accelerated (0 unsupported ops!)**:

| Backend (`renderer_decode.tflite`, 285.66 MB) | Compile | Parity vs PyTorch (`PSNR` / MAE) | Live 512×512 End-to-End (w/ readback + `<canvas>`) | Real-Time Margin (vs 25 fps) |
| --- | --- | --- | --- | --- |
| **Chrome WebGPU (FP32)** | **530.3 ms** | **`118.23 dB`** / **`0.0001/255`** | **`23.53 ms/frame` (`42.5 FPS`)** | **1.70× realtime** |
| **Chrome WebGPU (FP16)** | **538.8 ms** | `42.72 dB` / `0.9496/255` | **`19.64 ms/frame` (`50.9 FPS`)** | **2.04× realtime** |
| **Native Windows CPU (TFLite XNNPACK)** | — | **`112.96 dB`** / `0.0003/255` | `306.8 ms/frame` (`3.3 FPS`) | `0.13×` |

### Milestone 3 — `wav2vec2_2s.tflite` + End-to-End Browser Demo — COMPLETE ✅
- [x] Author `wav2vec2_2s` + `audio_projection` + `pose_projection` (`[1, 32000]` + `[1, 50, 3]` → `[1, 50, 32]` + `[1, 50, 32]`) in C++ ([litert_poc/wav2vec2_export.cc](file:///c:/Projects/FlashIMTalkerV2/litert_poc/wav2vec2_export.cc)) — 7× Conv1D + GroupNorm + `ResizeBilinear(align_corners=true)` + 16-slice weight-norm `pos_conv_embed` + 12 post-LN Transformer layers
- [x] Build `imtalker_wav2vec2_export.exe` (3rd consecutive zero-error MSVC first-try build!), export `litert_poc/wav2vec2_2s.tflite` (**360.20 MB** FP32), and verify parity vs PyTorch:
  - **`a_now`**: **`cos_sim = 1.00000000`**, `mean_abs_err = 0.000002`, `max_abs_err = 0.000045`
  - **`pose_emb`**: **`cos_sim = 1.00000000`**, `mean_abs_err = 0.000000`, `max_abs_err = 0.000000`
- [x] Precompute 9.17 MB NHWC face caches for **both Puck & Gretchen** (`litert_poc/avatars/{puck,gretchen}/`), export `blink_delta.bin` (`[9, 32]`), and synthesize real Windows SAPI demo WAVs (`puck_demo.wav`, `gretchen_demo.wav`)
- [x] Build standalone interactive Chrome WebGPU Studio ([litert_poc/web/demo.html](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/demo.html)) with instant Puck ↔ Gretchen switching, Web Audio sync, speaking blinks, calm head sway, audio file upload, and mic recording
- [x] Verify full 3-model end-to-end live playback in Chrome WebGPU (`nvidia blackwell`, **0 unsupported ops across all 3 models!**):

| Pipeline Stage (Chrome WebGPU FP32, `@litertjs/core@2.6.0`) | Model Size (FP32) | Steady-State WebGPU Time | Real-Time Speedup | Parity vs PyTorch |
| --- | --- | --- | --- | --- |
| **1. `wav2vec2_2s.tflite`** (2.0 s audio + pose) | `360.20 MB` | **`12.6 – 15.0 ms` / 2.0 s** | **~145× realtime** | **`cos_sim = 1.00000000`** (`MAE = 2e-6`) |
| **2. `fm_step.tflite`** (6-step ODE, 2.0 s motion) | `148.20 MB` | **`18.6 – 20.4 ms` / 2.0 s** | **~103× realtime** | **`cos_sim = 1.00000000`** (`MAE = 4.5e-4`) |
| **3. `renderer_decode.tflite`** (512×512 live `<canvas>`) | `285.66 MB` | **`23.16 ms / frame` (`43.2 FPS`)** | **1.73× realtime** (vs 25 fps) | **`PSNR = 118.23 dB`** (`MAE = 0.0001/255`) |
| **End-to-End Test (`gretchen_demo.wav`, 6.77 s)** | `794.1 MB` total | **170 / 170 live frames (0 drops)** | **Real-time 25 fps synced** | **0 console errors / 0 GPU warnings** |

### Milestone 4 — Conversational Studio (Chrome Gemini Nano + Voice/Text Q&A + Warm SAPI TTS) — COMPLETE ✅
- [x] Build persistent warm Windows SAPI 5 COM worker ([litert_poc/web/sapi_worker.ps1](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/sapi_worker.ps1)) pre-warmed with `Andrew (Natural HD)` (Puck) & `Jenny(Natural)` (Gretchen) (`63` voices detected)
- [x] Add `POST /api/tts` (warm SAPI 16 kHz WAV) and `POST /api/llm` (optional local LM Studio `127.0.0.1:1234` fallback proxy) to [litert_poc/web/server.mjs](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/server.mjs)
- [x] Upgrade [litert_poc/web/demo.html](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/demo.html) with:
  - Chrome Built-in **Gemini Nano (`LanguageModel` Prompt API)** with `availability()` check, user-gesture download button, streaming replies, and automatic Chromium-stub detection → Local LLM fallback
  - Text Q&A input + **🎤 Ask by Voice** (`SpeechRecognition` live mic transcription → auto-send)
  - Shared multi-turn Puck & Gretchen conversation memory + `"Hey Puck"` / `"Hey Gretchen"` automatic avatar & voice routing
- [x] Verified live end-to-end conversational flow in browser (`LanguageModel` → warm SAPI `Jenny(Natural)` → `wav2vec2_2s` + `fm_step` + `renderer_decode` → **503 / 503 consecutive live 512×512 frames at `23.3 ms/frame` = `43.0 FPS`**, zero dropped frames)

### Milestone 5 — 100% Browser-Local Pipeline (Auto-Loading WebLLM + SAPI + Live Combined VRAM Audit) — COMPLETE ✅
- [x] Installed `@mlc-ai/web-llm@0.2.89` locally in [litert_poc/web/package.json](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/package.json) (self-contained Rollup ESM bundle, zero bundler required)
- [x] Switched [litert_poc/web/server.mjs](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/server.mjs) COEP header from `require-corp` → `credentialless` so cross-origin WebLLM model shard fetches from Hugging Face CDN succeed while keeping `SharedArrayBuffer` enabled
- [x] Upgraded [litert_poc/web/demo.html](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/demo.html) with:
  - **Zero-click automatic in-browser WebGPU LLM** (`Llama-3.2-1B-Instruct-q4f16_1-MLC` default, plus `gemma-2-2b-it-q4f16_1-MLC-1k` and `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` options) — auto-downloads once (`664 MB`) into browser Cache API, then boots from cache in **`1.5 s`** in **any** WebGPU browser (no Chrome flags, no user-gesture gate, no LM Studio needed!)
  - Seamless **Chrome Built-in Gemini Nano** option (auto-selected when branded Google Chrome reports `"available"`, with automatic WebLLM failover if unbranded Chromium's echo stub is ever detected)
  - Live **`Precision: FP32 / FP16`** WebGPU selector (also via `?precision=fp16`)
- [x] Verified full 100% browser-local E2E runs (`?autotest=webllm`) in both **FP32** and **FP16** with **0 console errors / 0 WebGPU warnings**, and measured exact live dedicated GPU VRAM (`Get-Counter '\GPU Process Memory(*)\Dedicated Usage'` + `nvidia-smi` on RTX 5090):

| Full Conversational Avatar Stack (Measured Live on RTX 5090) | LLM Engine | TTS Engine | Avatar 512×512 Engine | Process Dedicated VRAM | **Net AI VRAM** (above idle) | Reduction vs Desktop |
| --- | --- | --- | --- | --- | --- | --- |
| **Desktop PyTorch + LM Studio (Baseline)** | LM Studio `llama-server` (`5.71 GB`) | Edge/SAPI (`0 GB`) | Python PyTorch IMTalker + Whisper (`10.57 GB`) | `16.28 GB` | **`16.28 GB`** | `1.0×` |
| **Chrome 157 Gemini Nano v3 (Default) + IMTalker** | Chrome `OnDeviceModelService` (`nano_v3_gpu_component`, **`3.10 GB`**) | Warm Windows SAPI (`0 GB`) | LiteRT.js WebGPU in `chrome gpu-process` (`4.95 GB`, `+4.80 GB` above idle) | **`8.05 GB`** (`5,069 + 3,177 MB`) | **`~7.90 GB`** | **`2.1×` less VRAM** |
| **100% Browser-Local Studio — WebLLM FP32** | WebLLM `Llama-3.2-1B-q4f16_1` (`~0.88 GB`, **835 ms** warm) | Warm Windows SAPI (`0 GB`, **292 ms**) | LiteRT.js WebGPU FP32 (`~3.12 GB`, **`43.6 FPS`**, `118.2 dB`) | **`4.11 GB`** (`4,204.4 MB`) | **`4.00 GB`** | **`4.1×` less VRAM** |
| **100% Browser-Local Studio — WebLLM FP16** | WebLLM `Llama-3.2-1B-q4f16_1` (`~0.88 GB`, **835 ms** warm) | Warm Windows SAPI (`0 GB`, **365 ms**) | LiteRT.js WebGPU FP16 (`~2.12 GB`, `42.7 dB` visually lossless) | **`3.11 GB`** (`3,180.3 MB`) | **`3.00 GB`** | **`5.4×` less VRAM** |

*(Browser GPU process idle baseline = `0.11 GB`. Cold WebLLM download = `76 s` for `664 MB` on first visit; subsequent reloads boot the entire studio from browser cache in **`5.76 s`**, with WebLLM loading in **`1.5 s`** and generating replies in **`835 ms`**!)*

---

## Key Learnings Log

- **2026-11-25 — No `grid_sample` or `torch.qr` in active `IMTRenderer`:** Although `ToFlow` (`F.grid_sample`) and `Direction` (`torch.qr`) exist in `renderer/lia_resblocks.py`, `IMTRenderer` in `renderer/models.py` never imports or calls them. Active ops are strictly standard Convs, `StyledConv`, `SDPA` attention, `Swin` windowed attention, `GuidedResampler` (`argmax` + `gather`), and `PixelShuffle(2)` (`DepthToSpace`).
- **2026-11-25 — `FMGenerator` is already static-windowed:** `wav2vec_sec=2.0`, `fps=25`, `num_prev_frames=10` → fixed `seq_len=60` (`10` context + `50` generated frames) with a 6-step Euler ODE loop. Zero dynamic-shape surgery needed for `fm_step.tflite`.
- **2026-11-25 — 49% of `renderer.safetensors` is unnecessary in browser:** `criterion_vgg` (`49.39 MB`), `dense_feature_encoder` (`133.88 MB`), and `latent_token_encoder` (`81.87 MB`) account for `265.14 MB` of `543.09 MB`. Stripping them leaves `277.84 MB` FP32 (~`139 MB` FP16) for `renderer_decode.tflite`.
- **2026-11-25 — `generator.safetensors` schema is ultra-clean:** 110 tensors total (`38.86M` params): 8 standard DiT/adaLN-Zero blocks (`dim_h=512`, `num_heads=8`, `mlp=2048`), `pos_embed [1,60,512]`, `alignment_mask BOOL [60,60]`, `t_embedder`, `x_embedder`, `c_embedder`, `decoder`, and 4 tiny condition projections.
- **2026-11-25 — Native Windows CMake + VS 2022 MSVC 19.44 works cleanly (no WSL needed):** Pass `-DLITERT_ENABLE_NPU=OFF -DLITERT_ENABLE_GPU=OFF -DLITERT_BUILD_TESTS=OFF -DLITERT_DISABLE_KLEIDIAI=ON` when configuring `LiteRT/litert` to skip multi-GB NPU vendor SDK downloads. Incremental rebuilds of `imtalker_fm_export.exe` take ~5 seconds.
- **2026-11-25 — Critical WebGPU Delegate Rules (learned live in Chrome):**
  1. **Keep ALL tensors rank ≤ 4:** A single 5D `Reshape({2, 60, 3, 8, 64})` caused `RESHAPE: Tensor dimensions must be less than 5.` and forced 341/419 ops onto CPU fallback. Replacing it with 3D `Slice` + 4D `Reshape({2, 60, 8, 64})` made **100% of ops WebGPU-native**.
  2. **Avoid `Split` on WebGPU:** `SPLIT` failed with `Unknown layout.` on ML Drift WebGPU — replacing all `Split` calls with explicit 3D `Slice` achieved 0 unsupported ops.
  3. **Use FP16-safe attention mask (`-1e4f`, not `-1e9f`):** `-1e9f` overflows IEEE-754 FP16 (`max = 65504`) and produced `NaN` in `gpuOptions: { precision: 'fp16' }`. Switching to `-1e4f` fixed FP16 immediately (`cos_sim = 0.9980`).
  4. **Load LiteRT.js with `{ jspi: true }`:** Avoids `ReferenceError: Asyncify is not defined` if any mixed GPU/CPU fallback ever occurs.
  5. **WebGPU FP32 is the sweet spot on RTX 5090:** **`26.34 ms` for 6 ODE steps (75.9× realtime!)** with exact **`cos_sim = 1.00000000`** (`MAE = 0.000455`) — actually slightly faster than FP16 (`28.12 ms`) with zero precision loss!
- **2026-11-25 — Milestone 2 (`renderer_decode.tflite`) Breakthroughs:**
  1. **Zero-copy `GuidedResampler` algebraic collapse (`0.000000000` error):** For `k_top_samples=1`, `GuidedResampler` is bit-for-bit identical to precomputing `AvgPool2D(f_r)` offline (`f_r_4_pool [4096, 64]` + `f_r_5_pool [4096, 32]`), then running `ArgMax(kI32) → Gather → ResizeNearestNeighbor`! Cuts avatar face cache to **9.17 MB** and eliminates 1M+ per-frame gather indices.
  2. **Pure-4D `SwinUnifiedAttention` + algebraic `ModulatedConv2d` demod:** Writing `window_partition`/`window_reverse` in two 4D steps (`{nH, w, nW, w*C} ↔ {nH, nW, w, w*C}`), `CyclicRollNHWC` via 4D `Slice+Concatenation`, and `demod = rsqrt(W_sq @ style² + 1e-8)` achieved **0 unsupported GPU ops** across the entire 512×512 StyleGAN/Swin synthesis network!
  3. **Exact `Blur` padding in `ModulatedConv2d(upsample=True)`:** `p = (4 - 2) - (3 - 1) = 0` → `pad = (1, 1)` (not `(2, 1)`), yielding `(2H+1) + 2 - 4 + 1 = 2H` and **`118.23 dB PSNR`** on Chrome WebGPU FP32!
  4. **PyTorch `PixelShuffle(2)` vs TFLite `DepthToSpace(2)`:** Reordering the 12 output channels of `final_conv.1` from `(3, 2, 2)` → `(2, 2, 3)` at C++ load time matches `PixelShuffle(2)` bit-for-bit with zero extra runtime ops.
  5. **Full 512×512 Browser FPS Exceeds Real-Time (25 fps) in FP32:** **`42.5 FPS` (`23.53 ms/frame`) in FP32 (`118.23 dB PSNR`)** and **`50.9 FPS` (`19.64 ms/frame`) in FP16 (`42.72 dB PSNR`)** — including full GPU sync, readback, and `<canvas>` paint! Combined with `fm_step.tflite` (`26.3 ms` per 2.0 s clip = `0.53 ms/frame` amortized), **real-time 512×512 @ 25 fps IMTalker in Chrome WebGPU is 100% proven!**
- **2026-11-25 — Milestone 3 (`wav2vec2_2s.tflite` + E2E Browser Studio) Breakthroughs:**
  1. **16-slice `pos_conv_embed` + `(64, 63)` asymmetric padding:** Folding Hugging Face weight-norm (`weight_g / ||weight_v||`) at load time and lowering `groups=16, k=128` into 16 standard `groups=1` `Conv2D` slices (with `(64, 63)` padding matching `SamePadLayer`) ran **100% on WebGPU (`12.6–15.0 ms` per 2.0 s chunk!)** with **`cos_sim = 1.00000000` (`MAE = 0.000002`)**!
  2. **WASM32 2 GB Heap Limit & Clean Stage Swapping:** Compiling all 3 FP32 `.tflite` flatbuffers (`794.1 MB` total) simultaneously inside a single WASM32 instance hits WASM32's ~2 GB linear memory cap (`Aborted()`). Keeping the raw `.tflite` `Uint8Array`s in JS heap (outside WASM32) and staging `wav2vec2_2s` (`217 ms` recompile) ↔ `renderer_decode` (`427 ms` recompile) around `fm_step` keeps WASM32 heap comfortably under ~1 GB and ran **170/170 live frames at 43.2 FPS** cleanly. *(Optional next optimization: store model weights as FP16/INT8 to cut total flatbuffer size from `794 MB` → `~397 MB` so all 3 stay compiled in WASM32 simultaneously with 0 ms stage swap!)*
- **2026-11-25 — Milestone 4 (Gemini Nano + SAPI + Live Q&A) Learnings:**
  1. **Brand-checked Chrome vs. unbranded Chromium for Gemini Nano:** In unbranded Chromium (such as embedded webviews), `LanguageModel.availability()` reports `"downloadable"` and completes `"100%"`, then returns an echo stub (`"On-device model is not available in Chromium..."`). `demo.html` detects this stub automatically and falls back cleanly to **Local LLM (`127.0.0.1:1234`)**, while opening `http://127.0.0.1:8765/` in branded **Google Chrome** runs real on-device **Gemini Nano**.
  2. **Live browser WebGPU VRAM:** Measured **`~3.4 GB` VRAM per active tab** in FP32 (`IMT` `64×64` attention is `512 MB` FP32 + `Swin` + Dawn buffer pool) vs `10.57 GB` for the Python PyTorch + Whisper server. Using warm Windows SAPI (`sapi_worker.ps1`) keeps TTS VRAM overhead at **0 GB**.
- **2026-11-25 — Milestone 5 (100% Browser-Local WebLLM + Combined FP32 vs FP16 VRAM) Learnings:**
  1. **COEP `credentialless` unlocks WebLLM CDN downloads alongside `SharedArrayBuffer`:** Switching `Cross-Origin-Embedder-Policy` in `server.mjs` from `require-corp` to `credentialless` kept cross-origin isolation active for LiteRT.js WASM while letting `@mlc-ai/web-llm` fetch model shards directly from Hugging Face CDN with zero CORS/COEP errors.
  2. **Automatic caching & sub-second warm WebGPU LLM:** First-visit download of `Llama-3.2-1B-Instruct-q4f16_1-MLC` (`664 MB`) takes ~76 s and persists in the browser Cache API. Subsequent reloads boot WebLLM in **`1.5 s`** (entire studio ready in **`5.76 s`**) and generate full replies in **`835 ms`** (~86 tok/s on WebGPU).
  3. **Exact live combined VRAM on RTX 5090 (WebLLM + SAPI + 3-model LiteRT.js IMTalker):**
     - **FP32 mode:** **`4.11 GB`** total browser GPU process (`~4.00 GB` net AI VRAM: `~0.88 GB` WebLLM + `~3.12 GB` IMTalker + `0 GB` SAPI) — **`4.1×` smaller** than the `16.28 GB` desktop Python + LM Studio stack.
     - **FP16 mode:** **`3.11 GB`** total browser GPU process (`~3.00 GB` net AI VRAM: `~0.88 GB` WebLLM + `~2.12 GB` IMTalker + `0 GB` SAPI) — **`5.4×` smaller** than desktop, saving **`1.00 GB`** over FP32 with visually indistinguishable `42.7 dB` 512×512 output.
     - **Further VRAM headroom if needed later:** Exporting `.tflite` weights directly as FP16/INT8 (reducing the 3 `.tflite` flatbuffers from `794 MB` → `~397 MB` FP16 or `~200 MB` INT8) would shrink GPU weight buffers further and eliminate the `wav2vec2_2s ↔ renderer_decode` WASM32 stage swap.
  4. **Real Google Chrome 157 Gemini Nano v3 (`default`):** Set `Chrome Gemini Nano (Built-in · Default)` as the default backend in [demo.html](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/demo.html) (auto-prewarms on boot when `availability() === 'available'`, zero WebLLM download unless explicitly selected or needed as fallback). Verified live on **Google Chrome `157.0.8103.79`** with **`nano_v3_gpu_component` (`2025.8.8.1141`, `4,072.1 MB` `weights.bin`)**: Chrome spawns `chrome utility on_device_model.mojom.OnDeviceModelService` using **`3,177.3 MB` (`3.10 GB`)** dedicated GPU VRAM alongside `chrome gpu-process` (`5,069.4 MB`), for **`~7.90 GB` net VRAM** total (`2.1×` smaller than desktop `16.28 GB`, vs `3.0–4.0 GB` net when using the smaller `1B` WebLLM).
- **2026-11-25 — Live Memory HUD, 3-Turn Puck ↔ Gretchen Demo & Zero-Leak Flat VRAM Fix (`--fp16-weights`):**
  1. **Root cause of per-turn VRAM climb:** Swapping `renderer_decode ↔ wav2vec2_2s` via `model.delete()` + `loadAndCompile()` on every turn leaked/retained old ML Drift WebGPU buffers (~`+2–3 GB` per turn in Chrome, reaching `9.02 GB` WebGPU while Chrome Gemini Nano stayed flat at `3.12 GB`).
  2. **Permanent fix (`--fp16-weights` in all 3 C++ LiteRT exporters):** Storing weights (`>= 4096` elems) as `Type::kFP16` + `Dequantize` cut total model size **2.0×** (`794.1 MB` → **`397.7 MB`**: `wav2vec2_2s_fp16w.tflite` `180.4 MB` `cos_sim=0.99999996`, `fm_step_fp16w.tflite` `74.3 MB` `cos_sim=0.999858`, `renderer_decode_fp16w.tflite` `143.1 MB` **`PSNR=63.61 dB`**) with **0 unsupported WebGPU ops**.
  3. **All 3 models now compile ONCE at boot & stay permanently resident** (zero stage swapping, zero `model.delete()`/`loadAndCompile()` per turn, plus JS `.tflite` buffers freed right after compile → JS heap dropped **`0.84 GB` → `0.07 GB`**!).
  4. **Verified flat VRAM across 3 live 512×512 turns (`923/923` frames @ `44.2–44.7 FPS`):**
     - After Turn 1: **`4.16 GB`** (`Peak 4.18G`, `JS 0.07G`)
     - After Turn 2: **`4.16 GB`** (`Peak 4.18G`, `JS 0.06G`)
     - After Turn 3: **`4.16 GB`** (`Peak 4.18G`, `JS 0.08G`) — **`0.00 GB` drift!**
  5. **Live Top-Right Memory HUD ([vram_worker.ps1](file:///c:/Projects/FlashIMTalkerV2/litert_poc/web/vram_worker.ps1) + `GET /api/vram`):** Uses warm `Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory` (`~25 ms` poll) to show real-time `Nano`, `WebGPU`, `JS`, and `Peak` VRAM right in the top-right header.
- **2026-11-25 — Milestone 6 (Native Offline Android App `Pixel_Avatar` on Google Pixel 10 Tensor G5):**
  1. **Hardware & Baseline:** Google Pixel 10 (`frankel`, Tensor G5, PowerVR D-Series GPU, Android 17 / SDK 37, 12 GB RAM, AICore Gemini Nano built-in). Chrome 149 WebGPU ran `renderer_decode_fp16w` at `752.1 ms` (`1.33 FPS`).
  2. **100% Offline Native Kotlin + Compose Stack ([pixel_avatar](file:///c:/Projects/FlashIMTalkerV2/pixel_avatar)):** LiteRT `2.3.0` (`litert` + `litert-gpu`) + ML Kit GenAI Prompt (`genai-prompt:1.0.0-beta5` → AICore Gemini Nano `1.3 s/reply`) + offline Android Neural `TextToSpeech.synthesizeToFile()` (`~207 ms`).
  3. **LiteRT GPU Backend & Architecture Optimization Progression (512×512 on Pixel 10):**
     - Chrome WebGPU (`renderer_decode_fp16w`, `63.6 dB`): `752.1 ms` (**`1.33 FPS`**).
     - Native LiteRT default GPU (`renderer_decode_fp16w`, `63.6 dB`): `357.0 ms` (**`2.80 FPS`**, `2,136 MB` PSS).
     - Native LiteRT `WGPU-FP16` (Vulkan) + `--fast-mobile` (`32×32` `imt.3` + `GuidedResampler`, `28.0 dB` — visibly soft): `263.5 ms` (**`3.80 FPS`**).
     - Native LiteRT `WGPU-FP16` + `--fast-mobile --fast-swin256 --fast-swin128` (`27.6 dB` — soft): `179.1 ms` (**`5.58 FPS`**).
     - **Winning Razor-Sharp Production Config (`renderer_decode_hd_fast.tflite` = `--fp16-weights --fast-synth64 --fast-swin256 --fast-swin128`):** Keeps `imt.3` + `GuidedResampler` at full `64×64` (`PSNR = 38.15 dB`, `MAE = 1.92/255`, visually indistinguishable from PyTorch!) while running `SynthesisNetwork` `64/128/256` attention at half spatial resolution with full-res residual + MLP → **`237.3 ms` (`4.21 FPS` raw, `+50%` over native baseline, `3.2×` over Chrome WebGPU, `3.4 s` cold boot, `0 s` turn buffering in default `Live` mode, optional `12.5/25 FPS Smooth` toggle)**!
  4. **Key GPU Delegate Discovery:** Standalone per-channel `INT8 → FP32` `Dequantize` failed compilation across all 4 LiteRT GPU backends (`CL-BUF`, `CL-TEX`, `WGPU`, `GL`) and fell back to CPU (`0.64 FPS`); LiteRT GPU requires `FP16 → FP32` weight dequantization (or AOT-compiled Tensor G5 NPU INT8 models via `litert-npu-runtime-google-tensor`).
- **2026-11-25 — Milestone 7 (Distilled `6.74 MB` Mobile Student Renderer → `42.70 FPS` Live on Pixel 10!):**
  1. **Teacher Dataset on RTX 5090 ([distill_mobile_renderer.py](file:///c:/Projects/FlashIMTalkerV2/litert_poc/distill_mobile_renderer.py)):** Generated `7,240` teacher frames (`3,500` Puck + `3,500` Gretchen + `240` unseen validation speech frames across real phonetically balanced WAVs, manifold covariance augmentation, poses & blinks) at **`129 FPS`** in `65.8 s`.
  2. **FiLM-Conditioned CoordConv Mobile U-Net (`3.39M` params = `6.74 MB` FP16 — `21.2×` smaller than `143.1 MB` teacher!):** Pure LiteRT GPU ops (`Conv2D`, `FullyConnected`, `ResizeBilinear`, `Concat`, `Add`, `Mul`, `LeakyRelu`, `Logistic`), BatchNorm folded 100% into Conv2D weights/biases at export, zero Swin/Attention ops.
  3. **Training & Export Results (100 epochs on RTX 5090, `17.8 min`):**
     - **Unseen Speech Validation:** **`PSNR = 36.64 dB`**, **`MouthMAE = 2.75 / 255`** (~1.0% pixel error on unseen Puck & Gretchen speech!).
     - **C++ LiteRT Export ([student_renderer_export.cc](file:///c:/Projects/FlashIMTalkerV2/litert_poc/student_renderer_export.cc)):** `student_renderer_256_fp16w.tflite` (`6.74 MB`), **`PSNR = 73.29 dB`** vs PyTorch FP32 (`MAE = 0.0244/255`), `10.5 ms` even on PC CPU (`95 FPS`).
  4. **Verified Live on Google Pixel 10 (`Tensor G5`, `WGPU-FP16`, 100% Offline):**
     - **Render Time:** **`23.4 ms/frame`** (**`42.70 FPS`** — **`32.1×` faster** than Chrome Mobile WebGPU `1.33 FPS`, **`15.3×` faster** than native LiteRT teacher `2.80 FPS`, **`1.71×` faster than 25 FPS realtime**!)
     - **Compile Time:** **`61 ms`** renderer compile (`2.08 s` total 3-model cold boot!)
     - **App RAM (`PSS`):** **`~850 MB`** (`-1.28 GB` lower than teacher!).

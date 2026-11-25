# Chrome Avatar — 100% Browser-Local Conversational 512×512 AI Studio (WebGPU)

Real-time **512×512 talking-head avatars** (**Puck** & **Gretchen**) running **100% locally in Chrome via WebGPU** — combining **Chrome Built-in Gemini Nano** (or automatic in-browser **WebLLM**), **warm Windows SAPI neural voices** (`0 GB` VRAM), and Google's **LiteRT Tensor API → LiteRT.js (`@litertjs/core`) WebGPU** inference stack.

No Python inference server. No PyTorch runtime. No cloud LLM APIs. Built natively on Windows (VS 2022 MSVC + CMake, zero WSL).

![IMTalker Browser Studio — Live 512x512 WebGPU Avatar with Flat VRAM HUD and 3-Turn Dialogue](docs/studio_screenshot.png)

---

## Highlights

- **44+ FPS live 512×512 WebGPU avatar rendering** (`22.4 ms/frame` FP32 · `19.6 ms/frame` FP16 on NVIDIA RTX 5090 — **1.8× faster than 25 fps realtime**), **0 unsupported WebGPU ops** across all 3 models.
- **Zero-leak, rock-solid flat VRAM (`0.00 GB` drift across turns):** All 3 FP16-weight `.tflite` models (`397.7 MB` total) compile **once at boot** and stay permanently resident on WebGPU — verified **`4.16 GB → 4.16 GB → 4.16 GB`** across consecutive live turns (`923 / 923` frames, `0` drops).
- **4×–5.4× less VRAM than desktop PyTorch + LM Studio:** **`3.00–4.00 GB` net VRAM** with in-browser WebLLM (`Llama 3.2 1B`), or **`~7.90 GB` net VRAM** with real **Google Chrome Built-in Gemini Nano v3** (`OnDeviceModelService`), vs. **`16.28 GB`** for the desktop Python + LM Studio stack.
- **Direct C++ LiteRT Tensor API authoring (zero Python converters):** All 3 `.tflite` graphs (`wav2vec2_2s`, `fm_step`, `renderer_decode`) are authored directly in C++ from `.safetensors` checkpoints via `litert::tensor::ModelFactory::Save` with `--fp16-weights` (`Type::kFP16` + `Dequantize`) and bit-exact parity verification against PyTorch.
- **Multi-avatar conversational studio:**
  - Instant **Puck ↔ Gretchen** switching (click buttons or say `"Hey Puck..."` / `"Hey Gretchen..."`) with per-avatar portraits, GPU face caches (`9.17 MB` each), SAPI voices (`Andrew (Natural HD)` / `Jenny(Natural)`), and persona prompts sharing one room conversation history.
  - **🎭 3-Turn Demo Chat:** One-click live 3-turn back-and-forth conversation between Puck and Gretchen on preset or custom topics (default: *"What you did last weekend"*).
  - **Live Top-Right Memory HUD:** Real-time GPU VRAM (`Nano`, `WebGPU`, `JS`, and session `Peak`) polled in `~25 ms` via a warm Windows GPU performance counter worker (`GET /api/vram`).
  - **Text + Voice Q&A:** Type or click **🎤 Ask** (`Web SpeechRecognition`) to converse hands-free.

---

## Architecture

```mermaid
flowchart LR
    U["User (Text / Voice / 3-Turn Demo)"] --> LLM["In-Browser LLM<br/>Chrome Gemini Nano v3 (Default)<br/>or WebLLM (Llama 3.2 1B / Gemma 2 2B / Qwen 2.5 1.5B)"]
    LLM -->|"Streaming reply + shared roomHistory"| SAPI["Warm Windows SAPI 5 Worker<br/>Andrew HD (Puck) / Jenny (Gretchen)<br/>0 GB VRAM · ~250 ms"]
    SAPI -->|"16 kHz PCM WAV"| W2V["1. wav2vec2_2s_fp16w.tflite<br/>180.4 MB · WebGPU (~13 ms / 2s)"]
    W2V -->|"a_now + pose_emb"| FM["2. fm_step_fp16w.tflite<br/>74.3 MB · 6-step ODE WebGPU (~19 ms / 2s)"]
    FM -->|"32-D motion tokens + blinks"| REND["3. renderer_decode_fp16w.tflite<br/>143.1 MB · 512×512 WebGPU (22.5 ms / 44.5 FPS)"]
    REND --> CANVAS["HTML5 512×512 Canvas<br/>Synced with Web Audio"]
```

---

## Live Benchmarks & GPU VRAM Comparison (RTX 5090)

Measured live via Windows GPU Process Memory (`Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory`) + `nvidia-smi`:

| Full Conversational Avatar Stack | LLM Engine | TTS Engine | 512×512 Avatar Engine | **Net AI VRAM** | vs. Desktop |
| --- | --- | --- | --- | --- | --- |
| **Desktop Python PyTorch + LM Studio** *(baseline)* | `llama-server` (`5.71 GB`) | Edge / SAPI (`0 GB`) | Python IMTalker + Whisper (`10.57 GB`) | **`16.28 GB`** | `1.0×` |
| **Chrome 157 Gemini Nano v3 (Default) + IMTalker** | Chrome `OnDeviceModelService` (`3.10 GB`) | Warm SAPI (`0 GB`) | LiteRT.js WebGPU (`chrome gpu-process`) | **`~7.90 GB`** *(~6.9 GB FP16)* | **`2.1×` less** |
| **Browser-Local WebLLM + IMTalker (FP32)** | WebLLM `Llama-3.2-1B-q4f16_1` (`~0.88 GB`, `480 ms` warm) | Warm SAPI (`0 GB`) | LiteRT.js WebGPU FP32 (`44.5 FPS`, `63.6 dB`) | **`4.05 GB`** (`4.16 GB` flat) | **`4.0×` less** |
| **Browser-Local WebLLM + IMTalker (FP16)** | WebLLM `Llama-3.2-1B-q4f16_1` (`~0.88 GB`, `276–412 ms` warm) | Warm SAPI (`0 GB`) | LiteRT.js WebGPU FP16 (`54.6 FPS`, `42.7 dB`) | **`3.00 GB`** (`3.11 GB` flat) | **`5.4×` less** |

### Resident FP16-Weight `.tflite` Models (Zero Per-Turn Recompilation)

| Model | FP32 Size | **FP16-Weight Size** | Parity vs. PyTorch FP32 | Steady WebGPU Speed | Unsupported Ops |
| --- | --- | --- | --- | --- | --- |
| `wav2vec2_2s_fp16w.tflite` | `360.20 MB` | **`180.40 MB`** | `cos_sim = 0.99999996` (`MAE = 3e-5`) | **`~13 ms` / 2.0 s** (`~145×` RT) | **0** |
| `fm_step_fp16w.tflite` | `148.20 MB` | **`74.27 MB`** | `cos_sim = 0.99985790` | **`~19 ms` / 2.0 s** (`~103×` RT) | **0** |
| `renderer_decode_fp16w.tflite` | `285.66 MB` | **`143.06 MB`** | **`PSNR = 63.61 dB`** (`MAE = 0.066/255`) | **`22.5 ms/frame` (`44.5 FPS`)** | **0** |
| **Total** | `794.06 MB` | **`397.73 MB`** | **Visually lossless** | **Zero stage swaps** | **0** |

### Verified Flat VRAM Across 3 Consecutive Live Turns (`923 / 923` Frames)

| Turn (Puck ↔ Gretchen Live Dialogue) | Live FPS (`512×512`) | Motion Time | Browser VRAM | Peak VRAM | JS Heap |
| --- | --- | --- | --- | --- | --- |
| **After Turn 1/3** (Puck · `359` frames) | `44.5 FPS` (`22.5 ms`) | `284 ms` | **`4.16 GB`** | `4.18 GB` | `0.07 GB` |
| **After Turn 2/3** (Gretchen · `366` frames) | `44.2 FPS` (`22.6 ms`) | `273 ms` | **`4.16 GB`** | `4.18 GB` | `0.06 GB` |
| **After Turn 3/3** (Puck · `198` frames) | `44.7 FPS` (`22.4 ms`) | `125 ms` | **`4.16 GB`** | `4.18 GB` | `0.08 GB` |

---

## Repository Layout

```text
Chrome_Avatar/
├── README.md                    # This file
├── build.md                     # Complete engineering build log, milestones & WebGPU learnings
├── Start_Studio.bat             # 1-click launcher: starts hidden server.mjs & opens Google Chrome
├── Stop_Studio.bat              # 1-click clean shutdown for the hidden background server
├── CMakeLists.txt               # Portable CMake target definitions for LiteRT Tensor API
├── fm_step_export.cc            # C++ LiteRT exporter: Flow Matching DiT -> fm_step_fp16w.tflite
├── renderer_decode_export.cc    # C++ LiteRT exporter: StyleGAN + Swin + IMT -> renderer_decode_fp16w.tflite
├── wav2vec2_export.cc           # C++ LiteRT exporter: Wav2Vec2 + projections -> wav2vec2_2s_fp16w.tflite
├── docs/
│   └── studio_screenshot.png    # Live verified studio screenshot
├── avatars/
│   ├── blink_delta.bin          # [9, 32] speaking blink delta tokens
│   ├── puck/                    # Precomputed 9.17 MB NHWC WebGPU face cache + 512x512 portrait
│   └── gretchen/                # Precomputed 9.17 MB NHWC WebGPU face cache + 512x512 portrait
└── web/
    ├── demo.html                # 100% browser-local conversational studio UI
    ├── server.mjs               # Zero-framework Node HTTP server (COOP/COEP + /api/tts + /api/vram)
    ├── sapi_worker.ps1          # Persistent warm Windows SAPI 5 COM worker (~250 ms, 0 GB VRAM)
    ├── vram_worker.ps1          # Persistent warm Windows GPU VRAM counter worker (~25 ms poll)
    ├── package.json             # @litertjs/core + @mlc-ai/web-llm
    ├── index.html               # Standalone fm_step.tflite WebGPU parity benchmark
    ├── renderer.html            # Standalone renderer_decode.tflite 512x512 WebGPU benchmark
    └── samples/                 # Real Windows SAPI demo WAVs for Puck & Gretchen
```

---

## Quickstart

### Prerequisites
- **Windows 10/11** with **Node.js 20+** and a **WebGPU-capable GPU**
- **Google Chrome** (for built-in **Gemini Nano** Prompt API — default) or any WebGPU browser (automatically supports **WebLLM** `Llama 3.2 1B` / `Gemma 2 2B` / `Qwen 2.5 1.5B`)
- The 3 exported FP16-weight `.tflite` models placed in the repository root *(excluded from git due to GitHub's 100 MB file limit; see [Exporting the Models](#exporting-the-3-tflite-models-native-windows-c) below)*:
  - `wav2vec2_2s_fp16w.tflite` (`180.4 MB`)
  - `fm_step_fp16w.tflite` (`74.3 MB`)
  - `renderer_decode_fp16w.tflite` (`143.1 MB`)

### 1. One-Click Launch (Recommended — Zero Terminal Window)

Double-click **`Start_Studio.bat`** in the repository root:
- Automatically runs `npm install` inside `web/` on the first launch.
- Starts `node server.mjs` **completely hidden in the background** (no terminal window left open).
- Waits until `http://127.0.0.1:8765/` is ready and opens **Google Chrome** directly to the studio.
- When you're done, double-click **`Stop_Studio.bat`** to cleanly shut down the background server and its SAPI/VRAM workers.

*(Want to run manually in a terminal instead? Run `cd web && npm install && node server.mjs`, then open http://127.0.0.1:8765/ in Google Chrome.)*

### 2. Why Is `node server.mjs` Needed If All AI Runs in Chrome?

**Zero AI inference runs in `server.mjs` (`0 GB` VRAM).** Every neural model (Chrome Gemini Nano / WebLLM + `wav2vec2_2s`, `fm_step`, and `renderer_decode`) executes **100% inside Chrome on WebGPU**. `server.mjs` is only a lightweight local bridge for 3 things Chrome's security sandbox forbids from a bare `file://` HTML file:

1. **Local HTTP + `COOP` / `COEP` headers:** Chrome blocks ES modules, local `.tflite` fetches, `window.LanguageModel` (Gemini Nano), and `SharedArrayBuffer` (required by LiteRT.js WASM) over `file://`.
2. **Raw 16 kHz PCM audio from Windows SAPI voices (`POST /api/tts`):** Browsers can *play* OS voices via `speechSynthesis`, but intentionally block capturing raw PCM audio bytes from it — and `wav2vec2_2s.tflite` needs the raw waveform to drive exact lip-sync (`0 GB` VRAM).
3. **Live GPU VRAM HUD (`GET /api/vram`):** Browsers hide exact GPU VRAM usage from JavaScript for privacy/fingerprinting reasons, so `vram_worker.ps1` polls Windows' GPU performance counter (~25 ms) for the top-right HUD.

### 3. Open in Google Chrome & Precision Modes

Open a **normal (non-Incognito) Google Chrome** window *(use Google Chrome rather than Edge/Firefox for built-in Gemini Nano)*:

- **FP32 (`63.6 dB` renderer · `~44.5 FPS`):** http://127.0.0.1:8765/
- **FP16 (`-1.0 GB` VRAM · `~54.5 FPS` · `42.7 dB` visually lossless):** http://127.0.0.1:8765/web/demo.html?precision=fp16

> **Enabling Chrome Built-in Gemini Nano (Default):**
> 1. Make sure you are in **Google Chrome** on `http://127.0.0.1:8765/` (or `http://localhost:8765/`), not Incognito.
> 2. If Chrome reports `LanguageModel` unavailable, open `chrome://flags/#prompt-api-for-gemini-nano`, set it to **Enabled**, and relaunch Chrome.
> 3. If Gemini Nano is already downloaded (`OptGuideOnDeviceModel`), it pre-warms automatically on boot (`Gemini Nano Ready`). Otherwise click **⬇ Download Nano** or send your first message to trigger Chrome's one-time on-device download.
> 4. **No Chrome / want minimum VRAM?** Select **WebLLM: Llama 3.2 1B (0.88 GB VRAM)** from the **In-Browser LLM** dropdown — it auto-downloads once into browser cache and runs 100% on WebGPU in any browser (**`3.11 GB` total VRAM** in FP16).

### 4. Using the Studio

- **Text or Voice Q&A:** Type in the chat box (press **Enter**) or click **🎤 Ask** to speak into your mic.
- **Swap Avatars with Shared Memory:** Click **Puck** / **Gretchen** (or address them by name, e.g. *"Hey Gretchen, do you agree with Puck?"*) to swap the 512×512 avatar, Windows SAPI voice (`Andrew HD` ↔ `Jenny`), and personality prompt while sharing the same room conversation history.
- **🎭 3-Turn Demo Chat:** Pick a topic from **Demo Topic** (default: *"What you did last weekend"*, or type a custom topic first) and click **🎭 3-Turn Demo Chat** to watch Puck and Gretchen have a live 3-turn back-and-forth conversation — then jump right in afterward. Click the red **■** button anytime to stop.
- **Live Precision & Memory HUD:** Switch **Precision (`FP32` / `FP16`)** or **In-Browser LLM** directly in the UI — old WebGPU devices/engines are explicitly torn down and the top-right **Memory HUD** (`VRAM`, `Peak`, `Nano`, `WebGPU`, `JS`) recalculates automatically. Click the HUD anytime to force-refresh and reset `Peak` *(keep only one studio tab open for single-tab VRAM readings)*.

---

## Exporting the 3 `.tflite` Models (Native Windows C++, Zero WSL)

All 3 exporters use Google's **LiteRT Tensor API** directly from `.safetensors` checkpoints (`generator.safetensors`, `renderer.safetensors`, `wav2vec2-base-960h/model.safetensors`):

### 1. Build the 3 C++ Exporters Inside `google-ai-edge/LiteRT`

```powershell
git clone --depth 1 https://github.com/google-ai-edge/LiteRT.git C:\Projects\LiteRT

# Point LiteRT's tensor/CMakeLists.txt to this repo's CMakeLists.txt
# (add_subdirectory(examples/imtalker) with -DIMTALKER_SRC_DIR=C:/Projects/Chrome_Avatar)
cmake -G "Visual Studio 17 2022" -A x64 `
  -S C:\Projects\LiteRT\litert `
  -B C:\Projects\LiteRT\cmake_build_win `
  -DLITERT_ENABLE_NPU=OFF -DLITERT_ENABLE_GPU=OFF `
  -DLITERT_BUILD_TESTS=OFF -DLITERT_DISABLE_KLEIDIAI=ON `
  -DIMTALKER_SRC_DIR=C:/Projects/Chrome_Avatar

cmake --build C:\Projects\LiteRT\cmake_build_win --config Release `
  --target imtalker_wav2vec2_export imtalker_fm_export imtalker_renderer_export -- /m
```

### 2. Export FP16-Weight `.tflite` Files

```powershell
$bin = "C:\Projects\LiteRT\cmake_build_win\tensor\examples\imtalker\Release"

& "$bin\imtalker_wav2vec2_export.exe" `
  checkpoints\wav2vec2-base-960h\model.safetensors `
  checkpoints\generator.safetensors `
  wav2vec2_2s_fp16w.tflite --fp16-weights

& "$bin\imtalker_fm_export.exe" `
  checkpoints\generator.safetensors `
  fm_step_fp16w.tflite --fp16-weights

& "$bin\imtalker_renderer_export.exe" `
  checkpoints\renderer.safetensors `
  renderer_decode_fp16w.tflite --fp16-weights
```

---

## Key Engineering Learnings

See [build.md](build.md) for the complete milestone-by-milestone breakdown, including:
- Stripping `265.14 MB` (49%) of training-only VGG loss + offline-precomputed face encoders out of `renderer.safetensors`
- Pure rank-≤-4 `SwinUnifiedAttention`, algebraic `ModulatedConv2d` demodulation, zero-copy `GuidedResampler` (`ArgMax(kI32) + Gather + ResizeNearestNeighbor`), channel-reordered `DepthToSpace(2)` for `PixelShuffle(2)`, and 16-slice weight-norm `pos_conv_embed` for **0 unsupported WebGPU ops**
- Why per-turn `model.delete()` + `loadAndCompile()` leaked WebGPU buffers in `@litertjs/core`, and how `--fp16-weights` (`397.7 MB` total) let all 3 models stay permanently resident with **`0.00 GB` VRAM drift**

---

## Acknowledgements

- [IMTalker](https://github.com/cbsjtu01/IMTalker) for the talking-head motion & synthesis architecture
- [Google AI Edge LiteRT](https://github.com/google-ai-edge/LiteRT) & [`@litertjs/core`](https://www.npmjs.com/package/@litertjs/core) for the C++ Tensor API and WebGPU runtime
- [MLC WebLLM](https://github.com/mlc-ai/web-llm) & Chrome Built-in AI (`LanguageModel` Prompt API / Gemini Nano)

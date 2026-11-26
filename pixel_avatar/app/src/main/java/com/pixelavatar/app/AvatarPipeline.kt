package com.pixelavatar.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

data class AvatarCache(
    val key: String,
    val displayName: String,
    val tR: FloatArray,
    val featureBuffers: List<FloatArray>, // exact order matching renderer_decode inputs [1..11]
    val portrait: Bitmap,
    val portrait256Rgb: FloatArray,       // [256 * 256 * 3] NHWC float in [0, 1] for 256px student
    val portrait512Rgb: FloatArray,       // [512 * 512 * 3] NHWC float in [0, 1] for native 512px HD student v2
)

data class PrevState(
    val prevX: FloatArray = FloatArray(10 * 32),
    val prevA: FloatArray = FloatArray(10 * 32),
    val prevPose: FloatArray = FloatArray(10 * 32),
)

data class Wav2VecOut(val aNow: FloatArray, val poseEmb: FloatArray, val ms: Double)
data class FmOut(val tokens: FloatArray, val nextPrev: PrevState, val ms: Double)

data class BenchResult(
    val acceleratorLabel: String,
    val w2vMs: Double,
    val fmMs: Double,
    val rendMs: Double,
    val fps: Double,
)

class AvatarPipeline(private val context: Context) {

    companion object {
        const val MODEL_W2V = "wav2vec2_2s_fp16w.tflite"
        const val MODEL_FM = "fm_step_fp16w.tflite"
        const val MODEL_REND = "student_renderer_512_fp16w.tflite"
        private const val MODEL_REND_256 = "student_renderer_256_fp16w.tflite"
        private const val MODEL_REND_HD = "renderer_decode_hd_fast.tflite"
        private const val MODEL_REND_LEGACY = "renderer_decode_fp16w.tflite"

        const val RELEASE_BASE = "https://github.com/jjmlovesgit/Chrome_Avatar/releases/download/v1.0.0"

        // Exact renderer_decode.tflite input order [1..11] after t_c [0]
        private val FEATURE_FILES = listOf(
            "i_r.bin",
            "ma_r_0.bin",
            "ma_r_1.bin",
            "ma_r_2.bin",
            "ma_r_3.bin",
            "f_r_0.bin",
            "f_r_1.bin",
            "f_r_2.bin",
            "f_r_3.bin",
            "f_r_4_pool.bin",
            "f_r_5_pool.bin",
        )
    }

    private var env: Environment? = null
    private var w2vModel: CompiledModel? = null
    private var fmModel: CompiledModel? = null
    private var rendModel: CompiledModel? = null

    private var w2vIn: List<TensorBuffer> = emptyList()
    private var w2vOut: List<TensorBuffer> = emptyList()
    private var fmIn: List<TensorBuffer> = emptyList()
    private var fmOut: List<TensorBuffer> = emptyList()
    private var rendIn: List<TensorBuffer> = emptyList()
    private var rendOut: List<TensorBuffer> = emptyList()

    private val pixelBuf = IntArray(512 * 512)
    private var boundAvatarKey: String? = null
    var renderSide: Int = 512
        private set

    lateinit var blinkDelta: FloatArray
        private set
    val avatars = mutableMapOf<String, AvatarCache>()

    var activeAcceleratorLabel: String = "GPU"
        private set

    fun modelsDir(): File = File(context.filesDir, "models").apply { mkdirs() }

    fun resolveRendererFile(): File {
        val dir = modelsDir()
        for (name in listOf(MODEL_REND, MODEL_REND_256, MODEL_REND_HD, MODEL_REND_LEGACY)) {
            val f = File(dir, name)
            if (f.exists() && f.length() >= 1_000_000L) return f
        }
        return File(dir, MODEL_REND)
    }

    fun missingModels(): List<String> {
        val dir = modelsDir()
        val missing = mutableListOf<String>()
        for (name in listOf(MODEL_W2V, MODEL_FM)) {
            val f = File(dir, name)
            if (!f.exists() || f.length() < 10_000_000L) missing.add(name)
        }
        val rend = resolveRendererFile()
        if (!rend.exists() || rend.length() < 1_000_000L) missing.add(MODEL_REND)
        return missing
    }

    fun loadAvatarsFromAssets(onLog: (String) -> Unit) {
        blinkDelta = readAssetFloats("avatars/blink_delta.bin")
        avatars["puck"] = loadSingleAvatar("puck", "Puck")
        avatars["gretchen"] = loadSingleAvatar("gretchen", "Gretchen")
        onLog("Loaded Puck & Gretchen face caches (9.17 MB each) + blink_delta.")
    }

    private fun loadSingleAvatar(key: String, displayName: String): AvatarCache {
        val base = "avatars/$key"
        val tR = readAssetFloats("$base/t_r.bin")
        val feats = FEATURE_FILES.map { f -> readAssetFloats("$base/$f") }
        val bmp = context.assets.open("$base/portrait.png").use { BitmapFactory.decodeStream(it) }
        val scaled = if (bmp.width == 512 && bmp.height == 512) bmp
        else Bitmap.createScaledBitmap(bmp, 512, 512, true)
        val bmp256 = Bitmap.createScaledBitmap(scaled, 256, 256, true)
        val px = IntArray(256 * 256)
        bmp256.getPixels(px, 0, 256, 0, 0, 256, 256)
        bmp256.recycle()
        val rgb256 = FloatArray(256 * 256 * 3)
        for (i in px.indices) {
            val c = px[i]
            val b = i * 3
            rgb256[b] = ((c shr 16) and 0xFF) / 255f
            rgb256[b + 1] = ((c shr 8) and 0xFF) / 255f
            rgb256[b + 2] = (c and 0xFF) / 255f
        }
        val px512 = IntArray(512 * 512)
        scaled.getPixels(px512, 0, 512, 0, 0, 512, 512)
        val rgb512 = FloatArray(512 * 512 * 3)
        for (i in px512.indices) {
            val c = px512[i]
            val b = i * 3
            rgb512[b] = ((c shr 16) and 0xFF) / 255f
            rgb512[b + 1] = ((c shr 8) and 0xFF) / 255f
            rgb512[b + 2] = (c and 0xFF) / 255f
        }
        return AvatarCache(key, displayName, tR, feats, scaled, rgb256, rgb512)
    }

    private fun readAssetFloats(assetPath: String): FloatArray {
        val bytes = context.assets.open(assetPath).use { it.readBytes() }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        bb.asFloatBuffer().get(out)
        return out
    }

    fun initModels(preferGpu: Boolean, onLog: (String) -> Unit) {
        releaseModels()
        val environment = runCatching { Environment.create(context) }.getOrElse { Environment.create() }
        env = environment
        val avail = runCatching { environment.getAvailableAccelerators() }.getOrElse { emptySet() }
        onLog("LiteRT available accelerators: $avail")

        val dir = modelsDir()
        val cacheDir = File(context.cacheDir, "litert_gpu_cache").apply { mkdirs() }.absolutePath

        fun makeGpuOpts(
            backend: CompiledModel.GpuOptions.Backend = CompiledModel.GpuOptions.Backend.AUTOMATIC,
            storage: CompiledModel.GpuOptions.BufferStorageType = CompiledModel.GpuOptions.BufferStorageType.DEFAULT,
            texWeights: Boolean = false,
            cacheKey: String = "model",
        ): CompiledModel.Options {
            val opts = CompiledModel.Options(Accelerator.GPU)
            opts.gpuOptions = CompiledModel.GpuOptions(
                constantTensorSharing = true,
                allowSrcQuantizedFcConvOps = true,
                precision = CompiledModel.GpuOptions.Precision.FP16,
                bufferStorageType = storage,
                preferTextureWeights = texWeights,
                serializationDir = cacheDir,
                modelCacheKey = cacheKey,
                serializeProgramCache = true,
                backend = backend,
                priority = CompiledModel.GpuOptions.Priority.HIGH,
            )
            return opts
        }

        fun compileOne(name: String, opts: CompiledModel.Options, label: String): Pair<CompiledModel, String> {
            val path = File(dir, name).absolutePath
            if (preferGpu) {
                try {
                    val t0 = System.nanoTime()
                    val m = CompiledModel.create(path, opts, environment)
                    val ms = (System.nanoTime() - t0) / 1e6
                    onLog("  [$label] $name compiled (${"%.0f".format(ms)} ms)")
                    return m to label
                } catch (e: Throwable) {
                    onLog("  [$label fallback->CPU] $name: ${e.message}")
                }
            }
            val t0 = System.nanoTime()
            val m = CompiledModel.create(path, CompiledModel.Options(Accelerator.CPU), environment)
            val ms = (System.nanoTime() - t0) / 1e6
            onLog("  [CPU] $name compiled (${"%.0f".format(ms)} ms)")
            return m to "CPU"
        }

        val (mW2v, accW2v) = compileOne(
            MODEL_W2V,
            makeGpuOpts(cacheKey = "w2v_fp16"),
            "GPU-FP16",
        )
        val (mFm, accFm) = compileOne(
            MODEL_FM,
            makeGpuOpts(cacheKey = "fm_fp16"),
            "GPU-FP16",
        )

        // Lock directly to Pixel 10's fastest proven backend: WGPU-FP16 (Vulkan) -> CL-FP16-TEX -> CPU
        val rendFile = resolveRendererFile()
        val rendName = rendFile.name
        val rendCacheTag = "${rendFile.length()}_${rendFile.lastModified()}"
        if (preferGpu) {
            val wgpuOpts = makeGpuOpts(CompiledModel.GpuOptions.Backend.WEBGPU, CompiledModel.GpuOptions.BufferStorageType.DEFAULT, false, "rend_wgpu_$rendCacheTag")
            val clTexOpts = makeGpuOpts(CompiledModel.GpuOptions.Backend.OPENCL, CompiledModel.GpuOptions.BufferStorageType.TEXTURE_2D, true, "rend_cltex_$rendCacheTag")
            val rendPath = rendFile.absolutePath
            var compiled: Pair<CompiledModel, String>? = null
            for ((label, opts) in listOf("WGPU-FP16" to wgpuOpts, "CL-FP16-TEX" to clTexOpts)) {
                try {
                    val t0 = System.nanoTime()
                    val m = CompiledModel.create(rendPath, opts, environment)
                    val ms = (System.nanoTime() - t0) / 1e6
                    onLog("  [$label] $rendName compiled (${"%.0f".format(ms)} ms)")
                    compiled = m to label
                    break
                } catch (e: Throwable) {
                    onLog("  [$label fallback] $rendName: ${e.message?.take(100)}")
                }
            }
            val (mRend, accRend) = compiled ?: compileOne(rendName, CompiledModel.Options(Accelerator.CPU), "CPU")
            w2vModel = mW2v; fmModel = mFm; rendModel = mRend
            activeAcceleratorLabel = accRend
        } else {
            val (mRend, accRend) = compileOne(rendName, CompiledModel.Options(Accelerator.CPU), "CPU")
            w2vModel = mW2v; fmModel = mFm; rendModel = mRend
            activeAcceleratorLabel = accRend
        }

        onLog("Active winner: w2v=$accW2v, fm=$accFm, renderer=$activeAcceleratorLabel")

        w2vIn = w2vModel!!.createInputBuffers()
        w2vOut = w2vModel!!.createOutputBuffers()
        fmIn = fmModel!!.createInputBuffers()
        fmOut = fmModel!!.createOutputBuffers()
        rendIn = rendModel!!.createInputBuffers()
        rendOut = rendModel!!.createOutputBuffers()
        renderSide = if (rendOut[0].readFloat().size == 256 * 256 * 3) 256 else 512
        boundAvatarKey = null
    }

    fun bindAvatarToRenderer(avatar: AvatarCache) {
        if (boundAvatarKey == avatar.key) return
        if (rendIn.size == 2) {
            // 2-input distilled student renderer: [t_c, src_img] (auto-selects 512px HD v2 or 256px v1)
            if (renderSide == 512) rendIn[1].writeFloat(avatar.portrait512Rgb)
            else rendIn[1].writeFloat(avatar.portrait256Rgb)
        } else {
            // Full 12-input IMTalker teacher renderer: [t_c, i_r, ma_r_0..3, f_r_0..5]
            for (i in avatar.featureBuffers.indices) {
                rendIn[i + 1].writeFloat(avatar.featureBuffers[i])
            }
        }
        boundAvatarKey = avatar.key
    }

    fun runWav2VecChunk(pcm32k: FloatArray, pose50x3: FloatArray): Wav2VecOut {
        val m = requireNotNull(w2vModel) { "wav2vec2 model not initialized" }
        val t0 = System.nanoTime()
        w2vIn[0].writeFloat(pcm32k)
        w2vIn[1].writeFloat(pose50x3)
        m.run(w2vIn, w2vOut)
        val aNow = w2vOut[0].readFloat()
        val poseEmb = w2vOut[1].readFloat()
        val ms = (System.nanoTime() - t0) / 1e6
        return Wav2VecOut(aNow, poseEmb, ms)
    }

    fun runFmChunk(
        aNow: FloatArray,
        poseEmb: FloatArray,
        refX: FloatArray,
        prevState: PrevState,
        cfgScale: Float = 2.6f,
        poseWeight: Float = 0.45f,
        nfe: Int = 6,
        chunkIdx: Int = 0,
    ): FmOut {
        val m = requireNotNull(fmModel) { "fm_step model not initialized" }
        val t0 = System.nanoTime()
        val dDim = 32
        val lPrev = 10
        val lClip = 50
        val lTot = 60

        val poseScaled = FloatArray(lClip * dDim) { i -> poseEmb[i] * poseWeight }
        val condIn = FloatArray(2 * lTot * dDim)
        for (f in 0 until lPrev) {
            for (d in 0 until dDim) {
                val v = prevState.prevA[f * dDim + d] + prevState.prevPose[f * dDim + d]
                condIn[(0 * lTot + f) * dDim + d] = v
                condIn[(1 * lTot + f) * dDim + d] = v
            }
        }
        for (f in 0 until lClip) {
            for (d in 0 until dDim) {
                val p = poseScaled[f * dDim + d]
                val a = aNow[f * dDim + d]
                condIn[(0 * lTot + lPrev + f) * dDim + d] = p
                condIn[(1 * lTot + lPrev + f) * dDim + d] = a + p
            }
        }

        val z = makeGaussian(lClip * dDim, 42 + chunkIdx * 101)
        val xIn = FloatArray(2 * lTot * 64)
        for (b in 0 until 2) {
            for (f in 0 until lTot) {
                val base = (b * lTot + f) * 64
                for (d in 0 until dDim) xIn[base + d] = refX[d]
                if (f < lPrev) {
                    for (d in 0 until dDim) xIn[base + 32 + d] = prevState.prevX[f * dDim + d]
                }
            }
        }

        fmIn[2].writeFloat(condIn)
        fmIn[3].writeFloat(floatArrayOf(cfgScale))
        val dt = 1.0f / (nfe - 1)
        val tArr = FloatArray(1)

        for (step in 0 until (nfe - 1)) {
            for (b in 0 until 2) {
                for (f in 0 until lClip) {
                    val base = (b * lTot + lPrev + f) * 64 + 32
                    val zBase = f * dDim
                    System.arraycopy(z, zBase, xIn, base, dDim)
                }
            }
            tArr[0] = step * dt
            fmIn[0].writeFloat(tArr)
            fmIn[1].writeFloat(xIn)
            m.run(fmIn, fmOut)
            val vel = fmOut[0].readFloat()
            for (i in z.indices) z[i] += dt * vel[i]
        }

        val nextPrev = PrevState(
            prevX = z.copyOfRange((lClip - lPrev) * dDim, lClip * dDim),
            prevA = aNow.copyOfRange((lClip - lPrev) * dDim, lClip * dDim),
            prevPose = poseScaled.copyOfRange((lClip - lPrev) * dDim, lClip * dDim),
        )
        val ms = (System.nanoTime() - t0) / 1e6
        return FmOut(z, nextPrev, ms)
    }

    fun generateMotionTokens(
        pcm16k: FloatArray,
        durationSec: Double,
        avatar: AvatarCache,
        blinkStrength: Float = 1.35f,
    ): Triple<FloatArray, Int, Double> {
        val totalFrames = max(1, ceil(durationSec * 25.0).toInt())
        val numChunks = ceil(totalFrames / 50.0).toInt()
        val allTokens = FloatArray(numChunks * 50 * 32)
        var prev = PrevState()
        var motionMs = 0.0

        for (c in 0 until numChunks) {
            val chunkPcm = normalizeChunk(pcm16k, c * 32000)
            val pose = buildCalmPose(c * 50)
            val w2v = runWav2VecChunk(chunkPcm, pose)
            val fm = runFmChunk(w2v.aNow, w2v.poseEmb, avatar.tR, prev, chunkIdx = c)
            System.arraycopy(fm.tokens, 0, allTokens, c * 50 * 32, 50 * 32)
            prev = fm.nextPrev
            motionMs += w2v.ms + fm.ms
        }
        applySpeakingBlinks(allTokens, totalFrames, blinkStrength)
        return Triple(allTokens, totalFrames, motionMs)
    }

    fun renderFrame(avatar: AvatarCache, token32: FloatArray, outBitmap: Bitmap): Double {
        val m = requireNotNull(rendModel) { "renderer_decode model not initialized" }
        val t0 = System.nanoTime()
        bindAvatarToRenderer(avatar)
        rendIn[0].writeFloat(token32)
        m.run(rendIn, rendOut)
        val rgb = rendOut[0].readFloat()

        val side = if (rgb.size == 256 * 256 * 3) 256 else 512
        renderSide = side
        val n = side * side
        val dst = pixelBuf
        var base = 0
        var i = 0
        while (i < n) {
            var r = (rgb[base] * 255f + 0.5f).toInt()
            var g = (rgb[base + 1] * 255f + 0.5f).toInt()
            var b = (rgb[base + 2] * 255f + 0.5f).toInt()
            if (r < 0) r = 0 else if (r > 255) r = 255
            if (g < 0) g = 0 else if (g > 255) g = 255
            if (b < 0) b = 0 else if (b > 255) b = 255
            dst[i] = -16777216 or (r shl 16) or (g shl 8) or b
            base += 3
            i++
        }
        if (outBitmap.width == side && outBitmap.height == side) {
            outBitmap.setPixels(dst, 0, side, 0, 0, side, side)
        } else {
            val tmp = Bitmap.createBitmap(dst, 0, side, side, side, Bitmap.Config.ARGB_8888)
            val canvas = android.graphics.Canvas(outBitmap)
            val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
            canvas.drawBitmap(tmp, null, android.graphics.Rect(0, 0, outBitmap.width, outBitmap.height), paint)
            tmp.recycle()
        }
        return (System.nanoTime() - t0) / 1e6
    }

    fun runBenchmark(avatar: AvatarCache, outBitmap: Bitmap, nFrames: Int = 10): BenchResult {
        // Use a realistic quiet speech-level test waveform (avoiding all-zero LayerNorm degeneracy in FP16)
        val dummyPcm = FloatArray(32000) { i -> (0.02 * sin(2.0 * PI * 220.0 * i / 16000.0)).toFloat() }
        val w2v = runWav2VecChunk(dummyPcm, buildCalmPose(0))
        val fm = runFmChunk(w2v.aNow, w2v.poseEmb, avatar.tR, PrevState(), chunkIdx = 0)
        // Warmup + benchmark on valid neutral/speech tokens
        renderFrame(avatar, avatar.tR, outBitmap)
        var sumR = 0.0
        for (f in 0 until nFrames) {
            sumR += renderFrame(avatar, avatar.tR, outBitmap)
        }
        val avgR = sumR / nFrames
        return BenchResult(
            acceleratorLabel = activeAcceleratorLabel,
            w2vMs = w2v.ms,
            fmMs = fm.ms,
            rendMs = avgR,
            fps = 1000.0 / avgR,
        )
    }

    private fun normalizeChunk(pcm: FloatArray, startSample: Int): FloatArray {
        val out = FloatArray(32000)
        val lastIdx = max(0, pcm.size - 1)
        if (pcm.isNotEmpty()) {
            for (i in 0 until 32000) {
                val idx = startSample + i
                out[i] = if (idx < pcm.size) pcm[idx] else pcm[lastIdx]
            }
        }
        var mean = 0.0
        for (v in out) mean += v
        mean /= 32000.0
        var variance = 0.0
        for (i in 0 until 32000) {
            val d = out[i] - mean
            out[i] = d.toFloat()
            variance += d * d
        }
        val invStd = (1.0 / sqrt(variance / 32000.0 + 1e-7)).toFloat()
        for (i in 0 until 32000) out[i] *= invStd
        return out
    }

    private fun buildCalmPose(frameOffset: Int): FloatArray {
        val pose = FloatArray(50 * 3)
        for (f in 0 until 50) {
            val t = (frameOffset + f) / 25.0
            pose[f * 3 + 0] = (0.025 * sin(2.0 * PI * 0.20 * t)).toFloat()
            pose[f * 3 + 1] = (0.035 * sin(2.0 * PI * 0.20 * 0.73 * t + 1.1)).toFloat()
            pose[f * 3 + 2] = (0.012 * sin(2.0 * PI * 0.20 * 0.57 * t + 2.3)).toFloat()
        }
        return pose
    }

    private fun applySpeakingBlinks(allTokens: FloatArray, totalFrames: Int, strength: Float) {
        if (strength <= 0f || blinkDelta.size < 9 * 32) return
        val intervalFrames = (3.4 * 25.0).toInt()
        var start = 35
        while (start + 9 < totalFrames) {
            for (k in 0 until 9) {
                val base = (start + k) * 32
                val bBase = k * 32
                for (d in 0 until 32) allTokens[base + d] += strength * blinkDelta[bBase + d]
            }
            start += intervalFrames
        }
    }

    private fun makeGaussian(len: Int, seed: Int): FloatArray {
        var s = seed.toUInt()
        fun nextRand(): Double {
            s = s * 1664525u + 1013904223u
            return (s.toDouble() + 0.5) / 4294967296.0
        }
        val out = FloatArray(len)
        var i = 0
        while (i < len) {
            val u1 = max(1e-7, nextRand())
            val u2 = nextRand()
            val r = sqrt(-2.0 * ln(u1))
            val th = 2.0 * PI * u2
            out[i] = (r * cos(th)).toFloat()
            if (i + 1 < len) out[i + 1] = (r * sin(th)).toFloat()
            i += 2
        }
        return out
    }

    fun releaseModels() {
        fun closeAll(list: List<TensorBuffer>) = list.forEach { runCatching { it.close() } }
        closeAll(w2vIn); closeAll(w2vOut)
        closeAll(fmIn); closeAll(fmOut)
        closeAll(rendIn); closeAll(rendOut)
        w2vIn = emptyList(); w2vOut = emptyList()
        fmIn = emptyList(); fmOut = emptyList()
        rendIn = emptyList(); rendOut = emptyList()
        runCatching { w2vModel?.close() }; w2vModel = null
        runCatching { fmModel?.close() }; fmModel = null
        runCatching { rendModel?.close() }; rendModel = null
        runCatching { env?.close() }; env = null
        boundAvatarKey = null
    }
}

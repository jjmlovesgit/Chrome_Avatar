package com.pixelavatar.app

import android.app.ActivityManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.os.Bundle
import android.os.Debug
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.*
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : ComponentActivity() {

    private lateinit var pipeline: AvatarPipeline
    private lateinit var nanoLlm: NanoLlmClient
    private lateinit var ttsEngine: AndroidTtsEngine
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private var activeAudioTrack: AudioTrack? = null
    private var activeJob: Job? = null
    private var speechRecognizer: SpeechRecognizer? = null

    // Dedicated single thread for LiteRT GPU/OpenGL ES inference so the EGLContext
    // stays permanently bound to the exact thread that compiled the models (normal priority so AudioFlinger never starves).
    private val litertDispatcher = java.util.concurrent.Executors.newSingleThreadExecutor { r ->
        Thread(r, "LiteRT-GPU-Thread")
    }.asCoroutineDispatcher()

    /**
     * Hardware-clock PCM player built on Android AudioTrack + AudioTimestamp.
     * Tracks the exact acoustic playback time exiting the physical speaker / Bluetooth DAC
     * with sub-millisecond accuracy and zero cumulative drift over long utterances.
     */
    private class HardwareAudioPlayer(
        private val audio: TtsAudio,
        private val audioManager: AudioManager?,
        private val isAutomotive: Boolean,
    ) {
        val sampleRate: Int = audio.rawSampleRate // 48000 Hz native AudioFlinger HAL rate
        val totalFrames: Long = audio.rawTotalFrames.toLong()
        val durationMs: Double = audio.durationSec * 1000.0
        // 48 kHz STEREO 16-bit = 4 bytes/frame; pad 180 ms trailing silence so the final syllable never clips
        private val bytesPerFrame: Int = 4
        private val tailPadBytes: Int = ((sampleRate * 0.18).toInt().coerceAtLeast(1024)) * bytesPerFrame
        private val paddedPcm: ByteArray = audio.pcmRawBytes + ByteArray(tailPadBytes)
        private val totalPaddedFrames: Long = (paddedPcm.size / bytesPerFrame).toLong()
        private val useStaticMode: Boolean = paddedPcm.size in 4..(4 * 1024 * 1024)

        val track: AudioTrack
        private val ts = AudioTimestamp()
        private var focusRequest: AudioFocusRequest? = null
        @Volatile var finished: Boolean = false
            private set

        private var hasHwTs: Boolean = false
        private var anchorFramePos: Long = 0L
        private var anchorNanoTime: Long = 0L
        private var playStartNano: Long = 0L
        private var lastReturnedMs: Double = 0.0
        private var writerJob: Job? = null

        init {
            val minBuf = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,
            ).coerceAtLeast(16384)
            val bufBytes = if (useStaticMode) {
                paddedPcm.size
            } else {
                maxOf(minBuf * 4, sampleRate * bytesPerFrame) // 1.0s ring buffer for long utterances
            }
            val isEmulator = android.os.Build.HARDWARE.contains("ranchu", ignoreCase = true) ||
                android.os.Build.FINGERPRINT.contains("generic", ignoreCase = true)
            val attrs = AudioAttributes.Builder()
                // On real AAOS cars use USAGE_ASSISTANT; on x86_64 emulator use USAGE_MEDIA to avoid bus2 SODA contention
                .setUsage(if (isAutomotive && !isEmulator) AudioAttributes.USAGE_ASSISTANT else AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            if (isAutomotive && audioManager != null) {
                val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                    .setAudioAttributes(attrs)
                    .build()
                focusRequest = req
                runCatching { audioManager?.requestAudioFocus(req) }
            }
            track = AudioTrack.Builder()
                .setAudioAttributes(attrs)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                        .build()
                )
                .setBufferSizeInBytes(bufBytes)
                .setTransferMode(if (useStaticMode) AudioTrack.MODE_STATIC else AudioTrack.MODE_STREAM)
                .build()
        }

        fun start(scope: CoroutineScope) {
            if (useStaticMode) {
                // Entire utterance (+ trailing silence) lives in AudioFlinger shared memory:
                // zero background writes, zero underruns even when Qwen3 prefetches on CPU!
                track.write(paddedPcm, 0, paddedPcm.size)
                playStartNano = System.nanoTime()
                track.play()

                writerJob = scope.launch(Dispatchers.IO) {
                    try {
                        val minWaitMs = (durationMs + 100.0).toLong()
                        val maxWaitMs = (durationMs + 700.0).toLong()
                        while (isActive) {
                            val elapsedMs = (System.nanoTime() - playStartNano) / 1_000_000L
                            val head = track.playbackHeadPosition.toLong() and 0xFFFFFFFFL
                            if ((head >= totalPaddedFrames - 256 && elapsedMs >= minWaitMs) || elapsedMs >= maxWaitMs) break
                            delay(15)
                        }
                    } finally {
                        finished = true
                    }
                }
            } else {
                val prefill = minOf(paddedPcm.size, sampleRate * bytesPerFrame / 2) // 500 ms stereo prefill
                var offset = if (prefill > 0) track.write(paddedPcm, 0, prefill).coerceAtLeast(0) else 0
                playStartNano = System.nanoTime()
                track.play()

                writerJob = scope.launch(Dispatchers.IO) {
                    runCatching { android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO) }
                    try {
                        val chunkBytes = 16384
                        while (isActive && offset < paddedPcm.size) {
                            val toWrite = minOf(chunkBytes, paddedPcm.size - offset)
                            val written = track.write(paddedPcm, offset, toWrite, AudioTrack.WRITE_BLOCKING)
                            if (written <= 0) break
                            offset += written
                        }
                        val minWaitMs = (durationMs + 100.0).toLong()
                        val maxWaitMs = (durationMs + 700.0).toLong()
                        while (isActive) {
                            val elapsedMs = (System.nanoTime() - playStartNano) / 1_000_000L
                            val head = track.playbackHeadPosition.toLong() and 0xFFFFFFFFL
                            if ((head >= totalPaddedFrames - 256 && elapsedMs >= minWaitMs) || elapsedMs >= maxWaitMs) break
                            delay(15)
                        }
                    } finally {
                        finished = true
                    }
                }
            }
        }

        fun isHwTimestampLocked(): Boolean = hasHwTs

        /**
         * Returns the exact acoustic playback time (in ms) currently exiting the speaker/earbuds.
         * Smoothly interpolated between AudioTimestamp hardware DAC locks, never jumps backward.
         */
        fun acousticPositionMs(): Double {
            val nowNs = System.nanoTime()
            val wasLocked = hasHwTs
            if (runCatching { track.getTimestamp(ts) }.getOrDefault(false)) {
                val ageMs = (nowNs - ts.nanoTime) / 1_000_000.0
                if (ts.framePosition > 0 && ageMs in -200.0..2000.0) {
                    if (!hasHwTs || ts.framePosition >= anchorFramePos) {
                        anchorFramePos = ts.framePosition
                        anchorNanoTime = ts.nanoTime
                        hasHwTs = true
                    }
                }
            }
            val rawMs = if (hasHwTs) {
                val dtSec = (nowNs - anchorNanoTime) / 1_000_000_000.0
                val exactFrames = anchorFramePos + dtSec * sampleRate
                (exactFrames / sampleRate) * 1000.0
            } else {
                val head = track.playbackHeadPosition.toLong() and 0xFFFFFFFFL
                if (head > 0L) {
                    ((head.toDouble() / sampleRate) * 1000.0 - 45.0).coerceAtLeast(0.0)
                } else {
                    0.0
                }
            }
            val clamped = rawMs.coerceIn(0.0, durationMs)
            // On first hardware DAC lock, adopt true acoustic clock directly; thereafter stay monotonic
            if (!wasLocked && hasHwTs) {
                lastReturnedMs = clamped
            } else if (clamped >= lastReturnedMs) {
                lastReturnedMs = clamped
            }
            return lastReturnedMs
        }

        fun release() {
            val wasFinished = finished
            writerJob?.cancel()
            writerJob = null
            if (!wasFinished) {
                runCatching { track.pause() }
                runCatching { track.flush() }
            }
            runCatching { track.stop() }
            runCatching { track.release() }
            focusRequest?.let { req -> runCatching { audioManager?.abandonAudioFocusRequest(req) } }
            focusRequest = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pipeline = AvatarPipeline(this)
        nanoLlm = NanoLlmClient(this)
        ttsEngine = AndroidTtsEngine(this)

        setContent {
            PixelAvatarScreen()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching { speechRecognizer?.destroy() }
        speechRecognizer = null
        stopActivePlayback()
        scope.cancel()
        runCatching {
            kotlinx.coroutines.runBlocking(litertDispatcher) {
                pipeline.releaseModels()
            }
        }
        runCatching { litertDispatcher.close() }
        nanoLlm.close()
        ttsEngine.close()
    }

    private fun stopActivePlayback() {
        activeJob?.cancel()
        activeJob = null
        runCatching { activeAudioTrack?.pause() }
        runCatching { activeAudioTrack?.flush() }
        runCatching { activeAudioTrack?.stop() }
        runCatching { activeAudioTrack?.release() }
        activeAudioTrack = null
    }

    private fun readMemHud(): Pair<String, String> {
        val pssMb = Debug.getPss() / 1024.0
        val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val availGb = mi.availMem / (1024.0 * 1024.0 * 1024.0)
        val totalGb = mi.totalMem / (1024.0 * 1024.0 * 1024.0)
        return "APP: ${"%.0f".format(pssMb)} MB" to "Free ${"%.1f".format(availGb)} / ${"%.1f".format(totalGb)}G"
    }

    private fun detectAddressedAvatar(text: String, current: String): String {
        val lower = text.lowercase()
        val idxG = Regex("\\b(gretchen|gretchin|gretchan)\\b").find(lower)?.range?.first ?: -1
        val idxP = Regex("\\b(puck)\\b").find(lower)?.range?.first ?: -1
        if (idxG >= 0 && (idxP < 0 || idxG < idxP)) return "gretchen"
        if (idxP >= 0 && (idxG < 0 || idxP < idxG)) return "puck"
        return current
    }

    @Composable
    private fun PixelAvatarScreen() {
        var ready by remember { mutableStateOf(false) }
        var busy by remember { mutableStateOf(false) }
        var preferGpu by remember { mutableStateOf(true) }
        // 2 = Live (zero-wait raw GPU, default), 0 = 12.5 FPS Smooth, 1 = 25 FPS Smooth
        var playbackMode by remember { mutableIntStateOf(2) }
        var activeAvatar by remember { mutableStateOf("puck") }
        var voiceLabel by remember { mutableStateOf("tpd") }
        var puckVoiceShort by remember { mutableStateOf("tpd") }
        var gretchenVoiceShort by remember { mutableStateOf("iog") }
        val isAutomotiveDevice = remember {
            packageManager.hasSystemFeature(PackageManager.FEATURE_AUTOMOTIVE)
        }
        val audioManager = remember {
            getSystemService(AUDIO_SERVICE) as? AudioManager
        }
        val screenConfig = androidx.compose.ui.platform.LocalConfiguration.current
        // 0 = Flagship Studio (phone), 1 = Classic Console, 2 = Auto Cockpit (AAOS / widescreen)
        var uiMode by remember {
            mutableIntStateOf(if (isAutomotiveDevice || screenConfig.screenWidthDp >= 720) 2 else 0)
        }
        var isDrivingMode by remember { mutableStateOf(false) }
        var studyDeck by remember { mutableStateOf(StudyGuideParser.loadBundledSolarSystem(this@MainActivity)) }
        var quizActive by remember { mutableStateOf(false) }
        var quizCardIdx by remember { mutableIntStateOf(0) }
        var quizMastered by remember { mutableIntStateOf(0) }
        var quizRetryCount by remember { mutableIntStateOf(0) }
        var quizAwaitingAnswer by remember { mutableStateOf(false) }
        var quizCompleted by remember { mutableStateOf(false) }
        var quizLastQuestion by remember { mutableStateOf("") }
        var gmailAccessGranted by remember { mutableStateOf(GmailNotificationListener.isPermissionGranted(this@MainActivity)) }
        var gmailCachedCount by remember { mutableIntStateOf(GmailNotificationListener.cachedCount()) }
        val defaultDemoTopic = "What you did last weekend"
        val xShowcaseTopic = "★ X Demo Script (LiteRT Showcase)"
        var demoTopic by remember { mutableStateOf(defaultDemoTopic) }
        var showTopicDialog by remember { mutableStateOf(false) }
        var statusBadge by remember { mutableStateOf("Booting LiteRT…") }
        var nanoBadge by remember { mutableStateOf("Checking AICore…") }
        var inputText by remember { mutableStateOf("") }
        var showLog by remember { mutableStateOf(false) }
        var logText by remember { mutableStateOf("Booting Pixel Avatar (100% Offline Tensor G5)…\n") }
        var memMain by remember { mutableStateOf("APP: -- MB") }
        var memSub by remember { mutableStateOf("Measuring RAM…") }
        var metricLlm by remember { mutableStateOf("—") }
        var metricTts by remember { mutableStateOf("—") }
        var metricMotion by remember { mutableStateOf("—") }
        var metricRend by remember { mutableStateOf("—") }

        val chatHistory = remember { mutableStateListOf<ChatTurn>() }
        val chatListState = rememberLazyListState()
        val uiScope = rememberCoroutineScope()

        // Triple-buffered mutable bitmaps sized to renderer output (256x256 or 512x512)
        // so Compose RenderThread never reads a bitmap while GPU readback writes the next one.
        var frameBmp by remember { mutableStateOf<Bitmap?>(null) }
        var frameVersion by remember { mutableIntStateOf(0) }
        var bmpPool by remember {
            mutableStateOf(
                arrayOf(
                    Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888),
                    Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888),
                    Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888),
                )
            )
        }
        var poolIdx by remember { mutableIntStateOf(0) }
        fun nextWorkBmp(): Bitmap {
            val want = pipeline.renderSide
            if (bmpPool[0].width != want) {
                bmpPool = arrayOf(
                    Bitmap.createBitmap(want, want, Bitmap.Config.ARGB_8888),
                    Bitmap.createBitmap(want, want, Bitmap.Config.ARGB_8888),
                    Bitmap.createBitmap(want, want, Bitmap.Config.ARGB_8888),
                )
            }
            poolIdx = (poolIdx + 1) % bmpPool.size
            return bmpPool[poolIdx]
        }

        fun appendLog(msg: String) {
            logText += msg + "\n"
            android.util.Log.i("PixelAvatar", msg)
        }

        fun refreshMem() {
            val (m, s) = readMemHud()
            memMain = m
            memSub = "${pipeline.activeAcceleratorLabel} · $s"
            gmailAccessGranted = GmailNotificationListener.isPermissionGranted(this@MainActivity)
            gmailCachedCount = GmailNotificationListener.cachedCount()
        }

        fun selectAvatar(key: String) {
            activeAvatar = key
            voiceLabel = ttsEngine.shortVoiceName(key)
            pipeline.avatars[key]?.let { av ->
                frameBmp = av.portrait
                frameVersion++
            }
        }

        suspend fun downloadMissingModels() {
            val missing = pipeline.missingModels()
            if (missing.isEmpty()) return
            withContext(Dispatchers.IO) {
                val dir = pipeline.modelsDir()
                for (name in missing) {
                    withContext(Dispatchers.Main) {
                        statusBadge = "Downloading $name…"
                        appendLog("Downloading $name from Release v1.0.0…")
                    }
                    val tmp = File(dir, "$name.part")
                    val dst = File(dir, name)
                    val conn = (URL("${AvatarPipeline.RELEASE_BASE}/$name").openConnection() as HttpURLConnection).apply {
                        instanceFollowRedirects = true
                        connectTimeout = 15000
                        readTimeout = 30000
                    }
                    conn.inputStream.use { inp ->
                        tmp.outputStream().use { out ->
                            val buf = ByteArray(65536)
                            var total = 0L
                            while (true) {
                                val r = inp.read(buf)
                                if (r < 0) break
                                out.write(buf, 0, r)
                                total += r
                                if (total % (16L * 1024 * 1024) < 65536) {
                                    val mb = total / (1024.0 * 1024.0)
                                    withContext(Dispatchers.Main) { statusBadge = "$name ${"%.0f".format(mb)} MB" }
                                }
                            }
                        }
                    }
                    tmp.renameTo(dst)
                    withContext(Dispatchers.Main) { appendLog("Downloaded $name (${dst.length() / 1048576} MB)") }
                }
            }
        }

        suspend fun bootPipeline(useGpu: Boolean) {
            ready = false
            busy = true
            try {
                withContext(Dispatchers.IO) {
                    if (pipeline.avatars.isEmpty()) {
                        pipeline.loadAvatarsFromAssets { msg -> scope.launch { appendLog(msg) } }
                    }
                }
                selectAvatar(activeAvatar)
                downloadMissingModels()

                statusBadge = "Compiling LiteRT + Loading Local LLM…"
                val t0 = System.nanoTime()
                coroutineScope {
                    val nanoJob = async(Dispatchers.IO) {
                        nanoLlm.prepare(
                            preferGpu = useGpu,
                            onProgress = { prog -> uiScope.launch { statusBadge = prog } },
                        ) { msg -> scope.launch { appendLog(msg) } }
                    }
                    val ttsJob = async(Dispatchers.IO) { ttsEngine.prepareGenAiVoices { msg -> scope.launch { appendLog(msg) } } }
                    val pipeJob = async(litertDispatcher) {
                        pipeline.initModels(preferGpu = useGpu) { msg -> scope.launch { appendLog(msg) } }
                    }
                    pipeJob.await()
                    nanoJob.await()
                    ttsJob.await()
                }
                val bootSec = (System.nanoTime() - t0) / 1e9
                appendLog("LiteRT 3-model stack + ${nanoLlm.activeEngineShort} + ${ttsEngine.statusLabel} ready in ${"%.2f".format(bootSec)}s (${pipeline.activeAcceleratorLabel})")
                val nanoShort = if (nanoLlm.isReady) nanoLlm.activeEngineShort else nanoLlm.statusLabel
                val ttsShort = if (ttsEngine.statusLabel.contains("GenAI")) "GenAI 24kHz" else "Neural TTS"
                nanoBadge = "$nanoShort · $ttsShort"
                puckVoiceShort = ttsEngine.shortVoiceName("puck")
                gretchenVoiceShort = ttsEngine.shortVoiceName("gretchen")
                voiceLabel = ttsEngine.shortVoiceName(activeAvatar)

                statusBadge = "Benchmarking GPU…"
                val targetBmp = nextWorkBmp()
                val bench = withContext(litertDispatcher) {
                    val av = pipeline.avatars[activeAvatar]!!
                    pipeline.runBenchmark(av, targetBmp, nFrames = 6)
                }
                selectAvatar(activeAvatar)
                metricMotion = "${"%.0f".format(bench.w2vMs + bench.fmMs)} ms"
                metricRend = "${"%.1f".format(bench.rendMs)} ms (${"%.1f".format(bench.fps)} FPS)"
                appendLog(
                    "PIXEL 10 BENCH (${bench.acceleratorLabel}): w2v=${"%.1f".format(bench.w2vMs)}ms | " +
                        "fm=${"%.1f".format(bench.fmMs)}ms | rend=${"%.1f".format(bench.rendMs)}ms (${"%.2f".format(bench.fps)} FPS)"
                )
                statusBadge = "Ready · ${"%.1f".format(bench.fps)} FPS"
                refreshMem()
                ready = true
            } catch (e: Throwable) {
                appendLog("BOOT ERROR: ${e.stackTraceToString().take(600)}")
                statusBadge = "Error"
            } finally {
                busy = false
            }
        }

        suspend fun speakReply(avatarKey: String, replyText: String, preSynthAudio: TtsAudio? = null) {
            val avatar = pipeline.avatars[avatarKey] ?: return
            selectAvatar(avatarKey)
            val ttsAudio = if (preSynthAudio != null) {
                preSynthAudio
            } else {
                statusBadge = "TTS (${avatar.displayName})…"
                withContext(Dispatchers.IO) { ttsEngine.synthesize(replyText, avatarKey) }
            }
            metricTts = "${"%.0f".format(ttsAudio.synthMs)} ms"
            appendLog("TTS (${avatar.displayName}): ${"%.0f".format(ttsAudio.synthMs)} ms (${"%.2f".format(ttsAudio.durationSec)}s)")

            statusBadge = "Motion (${avatar.displayName})…"
            val (allTokens, totalFrames, motionMs) = withContext(litertDispatcher) {
                pipeline.generateMotionTokens(ttsAudio.pcm16k, ttsAudio.durationSec, avatar)
            }
            metricMotion = "${"%.0f".format(motionMs)} ms"

            if (playbackMode != 2) {
                // Rolling-buffer Smooth playback (12.5 FPS or 25 FPS): pre-renders crisp 256x256 display frames
                // (rendered at 512x512 then downscaled to lightweight 256x256 ARGB = 256 KB/frame)
                // so a 5s reply uses only ~15-30 MB RAM and plays back butter-smooth synced with zero drops!
                val is25 = (playbackMode == 1)
                val targetFps = if (is25) 25.0 else 12.5
                val fpsLabel = if (is25) "25 FPS" else "12.5 FPS"
                val stride = if (is25) 1 else 2
                val frameIndices = (0 until totalFrames step stride).toList()
                val nOut = frameIndices.size
                val renderedFrames = arrayOfNulls<Bitmap>(nOut)
                var sumRendMs = 0.0

                // Pre-render frames with live progress; once remaining render time < audio duration,
                // start audio playback and overlap the tail frames in parallel!
                val renderJob = uiScope.launch(litertDispatcher) {
                    val side = pipeline.renderSide
                    val tmp = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
                    for (k in 0 until nOut) {
                        if (!isActive) break
                        val fIdx = frameIndices[k]
                        val tok = allTokens.copyOfRange(fIdx * 32, (fIdx + 1) * 32)
                        val rMs = pipeline.renderFrame(avatar, tok, tmp)
                        sumRendMs += rMs
                        renderedFrames[k] = if (side == 256) {
                            tmp.copy(Bitmap.Config.ARGB_8888, false)
                        } else {
                            Bitmap.createScaledBitmap(tmp, 256, 256, true)
                        }
                        val avgR = sumRendMs / (k + 1)
                        withContext(Dispatchers.Main) {
                            metricRend = "${"%.0f".format(avgR)} ms (${"%.1f".format(1000.0 / avgR)} raw)"
                            if (activeAudioTrack == null) {
                                statusBadge = "Buffering ${k + 1}/$nOut…"
                            }
                        }
                    }
                    tmp.recycle()
                }

                // Wait until buffer ahead guarantees zero underruns (or all frames done)
                while (currentCoroutineContext().isActive) {
                    val doneCount = renderedFrames.count { it != null }
                    if (doneCount >= nOut) break
                    if (doneCount >= 3) {
                        val avgSec = (sumRendMs / doneCount) / 1000.0
                        val remRenderSec = (nOut - doneCount) * avgSec
                        if (remRenderSec < ttsAudio.durationSec * 0.80) break
                    }
                    delay(40)
                }

                // Present Frame 0 immediately before hardware audio starts
                renderedFrames[0]?.let { frameBmp = it; frameVersion++ }
                statusBadge = "Speaking (${avatar.displayName} · $fpsLabel)"
                val player = HardwareAudioPlayer(ttsAudio, audioManager, isAutomotiveDevice || uiMode == 2)
                activeAudioTrack = player.track
                var shown = 0
                try {
                    player.start(uiScope)

                    // Wait until hardware audio starts draining samples
                    val waitStart = System.nanoTime()
                    while (currentCoroutineContext().isActive && !player.finished &&
                        player.acousticPositionMs() <= 0.0 &&
                        (System.nanoTime() - waitStart) < 350_000_000L
                    ) {
                        delay(5)
                    }

                    val displayLeadMs = 16.0 // ~1 VSYNC display scanout lead
                    var lastK = 0
                    while (currentCoroutineContext().isActive && !player.finished) {
                        val posMs = player.acousticPositionMs()
                        val k = (((posMs + displayLeadMs) / 1000.0) * targetFps).toInt()
                        if (k >= nOut) break
                        if (k <= lastK) {
                            delay(5)
                            continue
                        }
                        val bmp = renderedFrames[k]
                        if (bmp != null) {
                            lastK = k
                            frameBmp = bmp
                            frameVersion++
                            shown++
                        } else {
                            delay(8)
                        }
                    }
                    while (!player.finished && currentCoroutineContext().isActive) delay(15)
                } finally {
                    renderJob.cancel()
                    player.release()
                    if (activeAudioTrack === player.track) activeAudioTrack = null
                }

                val rawAvg = if (sumRendMs > 0) sumRendMs / renderedFrames.count { it != null }.coerceAtLeast(1) else 0.0
                appendLog("[${avatar.displayName} SMOOTH] $shown/$nOut frames @ $fpsLabel synced (hwTs=${player.isHwTimestampLocked()}, raw GPU ${"%.0f".format(rawAvg)}ms)")
                statusBadge = "Ready · $fpsLabel Smooth"
                refreshMem()
            } else {
                // Zero-wait Live mode locked to hardware AudioTrack AudioTimestamp with render-ahead pacing:
                // Pre-renders frame idx (emaRendMs + 1 VSYNC) BEFORE its acoustic timestamp so every mouth
                // shape hits the OLED right as that exact audio sample exits the physical speaker/earbuds!
                val initBmp = nextWorkBmp()
                val firstTok = allTokens.copyOfRange(0, 32)
                var emaRendMs = withContext(litertDispatcher) { pipeline.renderFrame(avatar, firstTok, initBmp) }
                frameBmp = initBmp
                frameVersion++

                statusBadge = "Speaking (${avatar.displayName})"
                val player = HardwareAudioPlayer(ttsAudio, audioManager, isAutomotiveDevice || uiMode == 2)
                activeAudioTrack = player.track
                var rendered = 1
                var sumRendMs = emaRendMs
                var maxSyncErrMs = 0.0
                var sumAbsSyncErrMs = 0.0
                try {
                    player.start(uiScope)

                    // Wait until hardware audio starts draining samples
                    val waitStart = System.nanoTime()
                    while (currentCoroutineContext().isActive && !player.finished &&
                        player.acousticPositionMs() <= 0.0 &&
                        (System.nanoTime() - waitStart) < 350_000_000L
                    ) {
                        delay(4)
                    }

                    val displayLeadMs = 16.0 // ~1 VSYNC display pipeline
                    var lastIdx = 0
                    while (currentCoroutineContext().isActive && !player.finished) {
                        val posNowMs = player.acousticPositionMs()
                        val renderAheadMs = emaRendMs.coerceIn(10.0, 50.0) + displayLeadMs
                        val targetIdx = ((posNowMs + renderAheadMs) / 40.0).toInt()
                        if (targetIdx >= totalFrames) break
                        if (targetIdx <= lastIdx) {
                            delay(4)
                            continue
                        }
                        lastIdx = targetIdx
                        val tok = allTokens.copyOfRange(targetIdx * 32, (targetIdx + 1) * 32)
                        val targetBmp = nextWorkBmp()
                        val rMs = withContext(litertDispatcher) { pipeline.renderFrame(avatar, tok, targetBmp) }
                        emaRendMs = 0.8 * emaRendMs + 0.2 * rMs

                        // Pace presentation to hit exact acoustic timestamp (minus 1 VSYNC scanout)
                        val presentAtMs = targetIdx * 40.0 - displayLeadMs
                        val posAfterMs = player.acousticPositionMs()
                        val waitMs = (presentAtMs - posAfterMs).toLong()
                        if (waitMs in 2L..38L && !player.finished) {
                            delay(waitMs)
                        }
                        val posPresentMs = player.acousticPositionMs()
                        val errMs = (posPresentMs + displayLeadMs) - (targetIdx * 40.0)
                        val absErr = kotlin.math.abs(errMs)
                        sumAbsSyncErrMs += absErr
                        if (absErr > maxSyncErrMs) maxSyncErrMs = absErr

                        frameBmp = targetBmp
                        frameVersion++
                        sumRendMs += rMs
                        rendered++
                        val avgR = sumRendMs / rendered
                        metricRend = "${"%.1f".format(avgR)} ms (${"%.1f".format(1000.0 / avgR)} FPS)"
                    }
                    while (!player.finished && currentCoroutineContext().isActive) delay(15)
                } finally {
                    player.release()
                    if (activeAudioTrack === player.track) activeAudioTrack = null
                }

                val avgR = if (rendered > 0) sumRendMs / rendered else 0.0
                val fps = if (avgR > 0) 1000.0 / avgR else 0.0
                val meanSyncErr = if (rendered > 1) sumAbsSyncErrMs / (rendered - 1) else 0.0
                appendLog(
                    "[${avatar.displayName}] $rendered/$totalFrames frames | motion=${"%.0f".format(motionMs)}ms | " +
                        "rend=${"%.1f".format(avgR)}ms (${"%.1f".format(fps)} FPS) | " +
                        "syncErr=${"%.1f".format(meanSyncErr)}ms (max=${"%.0f".format(maxSyncErrMs)}ms, hwTs=${player.isHwTimestampLocked()})"
                )
                statusBadge = "Ready · ${"%.1f".format(fps)} FPS"
                refreshMem()
            }
        }

        suspend fun askQuizCardSuspended(idx: Int) {
            if (idx >= studyDeck.cards.size) {
                quizCompleted = true
                quizAwaitingAnswer = false
                val avatarKey = activeAvatar
                val name = pipeline.avatars[avatarKey]?.displayName ?: "Puck"
                val wrap = "Awesome job! You finished all ${studyDeck.cards.size} cards in ${studyDeck.title}, mastering $quizMastered of them!"
                chatHistory.add(ChatTurn(name, wrap, avatarKey))
                runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }
                speakReply(avatarKey, wrap)
                return
            }
            quizCardIdx = idx
            quizRetryCount = 0
            quizAwaitingAnswer = false
            val card = studyDeck.cards[idx]
            val avatarKey = activeAvatar
            val name = pipeline.avatars[avatarKey]?.displayName ?: "Puck"
            val bubbleIdx = chatHistory.size
            chatHistory.add(ChatTurn("$name · Card ${idx + 1}/${studyDeck.cards.size}", "Preparing question…", avatarKey))
            runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }

            statusBadge = "Card ${idx + 1}/${studyDeck.cards.size} (${card.topic})…"
            val (question, llmMs) = if (nanoLlm.isReady) {
                withContext(Dispatchers.IO) {
                    nanoLlm.askStudyQuestion(avatarKey, card, idx, studyDeck.cards.size) { partial ->
                        uiScope.launch {
                            if (bubbleIdx < chatHistory.size) {
                                chatHistory[bubbleIdx] = ChatTurn("$name · ${card.topic}", "$partial▌", avatarKey)
                            }
                        }
                    }
                }
            } else {
                StudyGuideParser.fallbackQuestion(card, idx, studyDeck.cards.size) to 1.0
            }
            metricLlm = if (nanoLlm.isReady) "${"%.0f".format(llmMs)} ms" else "Card"
            quizLastQuestion = question
            chatHistory[bubbleIdx] = ChatTurn("$name · ${card.topic}", question, avatarKey)
            appendLog("[QUIZ ASK ${idx + 1}/${studyDeck.cards.size} · ${card.topic}] (${"%.0f".format(llmMs)}ms): $question")
            speakReply(avatarKey, question)
            quizAwaitingAnswer = true
            statusBadge = "Your turn · ${card.topic}"
        }

        fun startStudyQuiz() {
            if (busy || !ready || studyDeck.cards.isEmpty()) return
            quizActive = true
            quizCardIdx = 0
            quizMastered = 0
            quizRetryCount = 0
            quizCompleted = false
            quizAwaitingAnswer = false
            activeJob = uiScope.launch {
                busy = true
                try {
                    chatHistory.add(ChatTurn("System", "Voice Study Quiz: ${studyDeck.title} (${studyDeck.cards.size} cards)"))
                    runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }
                    askQuizCardSuspended(0)
                } catch (_: CancellationException) {
                    statusBadge = "Stopped"
                } catch (e: Throwable) {
                    appendLog("QUIZ ERROR: ${e.message}")
                    statusBadge = "Ready"
                } finally {
                    busy = false
                }
            }
        }

        fun skipQuizCard() {
            if (!quizActive || busy || !ready) return
            activeJob = uiScope.launch {
                busy = true
                try {
                    askQuizCardSuspended(quizCardIdx + 1)
                } catch (_: CancellationException) {
                    statusBadge = "Stopped"
                } catch (e: Throwable) {
                    appendLog("QUIZ SKIP ERROR: ${e.message}")
                } finally {
                    busy = false
                }
            }
        }

        fun endStudyQuiz() {
            stopActivePlayback()
            quizActive = false
            quizAwaitingAnswer = false
            quizCompleted = false
            chatHistory.add(ChatTurn("System", "Study quiz ended ($quizMastered/${studyDeck.cards.size} mastered)."))
            statusBadge = "Ready"
        }

        fun isEmailQuery(text: String): Boolean {
            val lower = text.lowercase()
            return Regex("\\b(email|emails|gmail|inbox|mail)\\b").containsMatchIn(lower) &&
                Regex("\\b(read|check|last|latest|recent|new|any|summarize|summary|what|tell|got)\\b").containsMatchIn(lower)
        }

        fun readGmailBrief(userPrompt: String = "Read my last email") {
            if (busy || !ready) return
            val target = detectAddressedAvatar(userPrompt, activeAvatar)
            selectAvatar(target)
            activeJob = uiScope.launch {
                busy = true
                try {
                    val name = pipeline.avatars[target]?.displayName ?: "Puck"
                    chatHistory.add(ChatTurn("You", userPrompt))
                    val bubbleIdx = chatHistory.size
                    chatHistory.add(ChatTurn("$name · Gmail", "Checking notifications…", target))
                    runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }

                    gmailAccessGranted = GmailNotificationListener.isPermissionGranted(this@MainActivity)
                    if (!gmailAccessGranted) {
                        val msg = "Notification access for Gmail isn't enabled yet. I'm opening Android Notification Access now — toggle Pixel Avatar on, then ask me again!"
                        chatHistory[bubbleIdx] = ChatTurn("$name · Gmail", msg, target)
                        appendLog("[GMAIL] Permission not granted — opening ACTION_NOTIFICATION_LISTENER_SETTINGS")
                        speakReply(target, msg)
                        GmailNotificationListener.openNotificationAccessSettings(this@MainActivity)
                        return@launch
                    }

                    val emails = GmailNotificationListener.getLatestEmails(3)
                    gmailCachedCount = emails.size
                    if (emails.isEmpty()) {
                        val msg = "I don't see any active Gmail notifications in your shade right now. Send yourself a quick test email and ask me again!"
                        chatHistory[bubbleIdx] = ChatTurn("$name · Gmail", msg, target)
                        appendLog("[GMAIL] Connected, 0 active/recent Gmail notifications found.")
                        speakReply(target, msg)
                        return@launch
                    }

                    statusBadge = "Reading Gmail ($name)…"
                    val (brief, llmMs) = if (nanoLlm.isReady) {
                        withContext(Dispatchers.IO) {
                            nanoLlm.summarizeEmails(target, emails, userPrompt) { partial ->
                                uiScope.launch {
                                    if (bubbleIdx < chatHistory.size) {
                                        chatHistory[bubbleIdx] = ChatTurn("$name · Gmail", "$partial▌", target)
                                    }
                                }
                            }
                        }
                    } else {
                        nanoLlm.fallbackEmailBrief(emails) to 1.0
                    }
                    metricLlm = if (nanoLlm.isReady) "${"%.0f".format(llmMs)} ms" else "Local"
                    chatHistory[bubbleIdx] = ChatTurn("$name · Gmail", brief, target)
                    appendLog("[GMAIL BRIEF · ${emails.size} msg(s)] (${"%.0f".format(llmMs)}ms): $brief")
                    speakReply(target, brief)
                } catch (_: CancellationException) {
                    statusBadge = "Stopped"
                } catch (e: Throwable) {
                    appendLog("GMAIL ERROR: ${e.message}")
                    statusBadge = "Ready"
                } finally {
                    busy = false
                }
            }
        }

        fun sendQuestion(q: String) {
            val trimmed = q.trim()
            if (trimmed.isEmpty() || busy || !ready) return
            inputText = ""

            // Route spoken/typed Gmail requests directly to our on-device NotificationListenerService!
            if (!quizActive && isEmailQuery(trimmed)) {
                readGmailBrief(trimmed)
                return
            }

            // Route spoken/typed answer directly into the Voice Study Quiz evaluator when active!
            if (quizActive && quizAwaitingAnswer && !quizCompleted && quizCardIdx < studyDeck.cards.size) {
                activeJob = uiScope.launch {
                    busy = true
                    quizAwaitingAnswer = false
                    try {
                        val card = studyDeck.cards[quizCardIdx]
                        val avatarKey = activeAvatar
                        val name = pipeline.avatars[avatarKey]?.displayName ?: "Puck"
                        val allowRetry = (quizRetryCount == 0)

                        chatHistory.add(ChatTurn("You", trimmed))
                        val evalIdx = chatHistory.size
                        chatHistory.add(ChatTurn(name, "Checking answer…", avatarKey))
                        runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }

                        statusBadge = "Evaluating (${card.topic})…"
                        val (feedback, passed, llmMs) = if (nanoLlm.isReady) {
                            withContext(Dispatchers.IO) {
                                nanoLlm.evaluateStudyAnswer(avatarKey, card, quizLastQuestion, trimmed, allowRetry) { partial ->
                                    uiScope.launch {
                                        if (evalIdx < chatHistory.size) {
                                            chatHistory[evalIdx] = ChatTurn(name, "$partial▌", avatarKey)
                                        }
                                    }
                                }
                            }
                        } else {
                            val (fb, ok) = StudyGuideParser.fallbackEvaluate(card, trimmed, allowRetry)
                            Triple(fb, ok, 1.0)
                        }
                        metricLlm = if (nanoLlm.isReady) "${"%.0f".format(llmMs)} ms" else "Card"
                        val tag = if (passed) "[PASS]" else "[RETRY]"
                        chatHistory[evalIdx] = ChatTurn("$name · ${if (passed) "✓ ${card.topic}" else "Hint"}", feedback, avatarKey)
                        appendLog("[QUIZ EVAL ${card.topic} $tag] (${"%.0f".format(llmMs)}ms): $feedback")

                        speakReply(avatarKey, feedback)

                        if (passed) {
                            quizMastered++
                            delay(350)
                            askQuizCardSuspended(quizCardIdx + 1)
                        } else {
                            quizRetryCount++
                            quizAwaitingAnswer = true
                            statusBadge = "Try again · ${card.topic}"
                        }
                    } catch (_: CancellationException) {
                        statusBadge = "Stopped"
                    } catch (e: Throwable) {
                        appendLog("QUIZ EVAL ERROR: ${e.message}")
                        chatHistory.add(ChatTurn("System", "Error: ${e.message}"))
                        quizAwaitingAnswer = true
                        statusBadge = "Ready"
                    } finally {
                        busy = false
                    }
                }
                return
            }

            activeJob = uiScope.launch {
                busy = true
                try {
                    val target = detectAddressedAvatar(trimmed, activeAvatar)
                    selectAvatar(target)
                    chatHistory.add(ChatTurn("You", trimmed))
                    val thinkingIdx = chatHistory.size
                    val name = pipeline.avatars[target]?.displayName ?: "Puck"
                    chatHistory.add(ChatTurn(name, "Thinking…", target))
                    runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }

                    statusBadge = "Gemini Nano ($name)…"
                    val (reply, llmMs) = if (nanoLlm.isReady) {
                        withContext(Dispatchers.IO) {
                            nanoLlm.replyAs(target, chatHistory.dropLast(1), trimmed) { partial ->
                                uiScope.launch {
                                    if (thinkingIdx < chatHistory.size) {
                                        chatHistory[thinkingIdx] = ChatTurn(name, "$partial▌", target)
                                    }
                                }
                            }
                        }
                    } else {
                        "Hi! I am $name running 100 percent offline on your Pixel 10 Tensor G5 right now." to 1.0
                    }
                    metricLlm = "${"%.0f".format(llmMs)} ms"
                    chatHistory[thinkingIdx] = ChatTurn(name, reply, target)
                    appendLog("[$name via Gemini Nano] (${"%.0f".format(llmMs)}ms): $reply")
                    speakReply(target, reply)
                } catch (_: CancellationException) {
                    statusBadge = "Stopped"
                } catch (e: Throwable) {
                    appendLog("ERROR: ${e.message}")
                    chatHistory.add(ChatTurn("System", "Error: ${e.message}"))
                    statusBadge = "Ready"
                } finally {
                    busy = false
                }
            }
        }

        fun runThreeTurnDemo() {
            if (busy || !ready) return
            val topic = demoTopic.trim().ifBlank { defaultDemoTopic }
            val isScriptedShowcase = (topic == xShowcaseTopic)
            activeJob = uiScope.launch {
                busy = true
                try {
                    chatHistory.add(
                        ChatTurn(
                            "System",
                            if (isScriptedShowcase) "3-Turn X Showcase: Puck & Gretchen — LiteRT + Gemini Nano"
                            else "3-Turn Demo: Puck & Gretchen — $topic",
                        )
                    )
                    val scriptedLines = listOf(
                        "puck" to "Hey everyone on X! I'm Puck, and right now you're watching me render in native 512 by 512 HD, one hundred percent offline on a Google Pixel 10 — zero cloud servers, zero Python, even in Airplane Mode. Gretchen, tell them how we pulled this off right on a phone!",
                        "gretchen" to "Thanks, Puck! Every model in our stack was authored directly from safetensors in C++ using Google's LiteRT Tensor API, with zero unsupported GPU ops. We then distilled a 143-megabyte StyleGAN and Swin teacher down into an 11.4-megabyte native 512 by 512 student trained with perceptual and PatchGAN sharpness — running real-time on Pixel 10's GPU under one gigabyte of RAM!",
                        "puck" to "And that exact same 11.4-megabyte LiteRT model flies at over 85 frames per second in desktop Chrome WebGPU using just one gigabyte of VRAM — with Gemini Nano built right in on both Chrome and Pixel 10! Check out the open-source Chrome Avatar repo on GitHub to run both yourself — and shout out to the LiteRT team!",
                    )
                    val turns = if (isScriptedShowcase) {
                        scriptedLines
                    } else {
                        listOf(
                            "puck" to "Start a fun casual chat with Gretchen about: $topic. Share one thought and ask Gretchen what she thinks.",
                            "gretchen" to "Respond naturally to Puck about: $topic. Share your own take and ask Puck a quick follow-up.",
                            "puck" to "Respond warmly and wittily to Gretchen to wrap up this 3-turn chat about: $topic.",
                        )
                    }
                    // Pipelined lookahead: while Turn i is speaking, Turn i+1 prepares reply + TTS
                    // in the background so the gap between speakers stays tiny!
                    var prefetched: Deferred<Triple<String, Double, TtsAudio>>? = null

                    for ((idx, pair) in turns.withIndex()) {
                        val (speakerKey, cueOrScript) = pair
                        val name = pipeline.avatars[speakerKey]!!.displayName
                        selectAvatar(speakerKey)
                        val bubbleIdx = chatHistory.size
                        chatHistory.add(ChatTurn("$name (${idx + 1}/3)", if (isScriptedShowcase) cueOrScript else "Thinking…", speakerKey))
                        runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) }

                        val reply: String
                        val llmMs: Double
                        val preAudio: TtsAudio?

                        val pf = prefetched
                        prefetched = null
                        if (pf != null) {
                            statusBadge = "Turn ${idx + 1}/3 · Ready ($name)…"
                            val (r, ms, audio) = pf.await()
                            reply = r; llmMs = ms; preAudio = audio
                        } else if (isScriptedShowcase) {
                            reply = cueOrScript; llmMs = 0.0; preAudio = null
                        } else {
                            statusBadge = "Turn ${idx + 1}/3 · Nano ($name)…"
                            val pairRes = if (nanoLlm.isReady) {
                                withContext(Dispatchers.IO) {
                                    nanoLlm.replyAs(speakerKey, chatHistory.dropLast(1), cueOrScript) { partial ->
                                        uiScope.launch {
                                            if (bubbleIdx < chatHistory.size) {
                                                chatHistory[bubbleIdx] = ChatTurn(name, "$partial▌", speakerKey)
                                            }
                                        }
                                    }
                                }
                            } else {
                                when (idx) {
                                    0 -> "I hiked a misty trail and found an amazing espresso spot. What did you do, Gretchen?"
                                    1 -> "That sounds awesome! I baked fresh sourdough and read all afternoon. Where was that coffee shop?"
                                    else -> "Right behind the bookstore — next weekend let's grab espresso and bring your sourdough!"
                                } to 1.0
                            }
                            reply = pairRes.first; llmMs = pairRes.second; preAudio = null
                        }

                        metricLlm = if (isScriptedShowcase) "Script" else "${"%.0f".format(llmMs)} ms"
                        chatHistory[bubbleIdx] = ChatTurn(name, reply, speakerKey)
                        appendLog("[Turn ${idx + 1}/3 · $name] (${if (isScriptedShowcase) "script" else "${"%.0f".format(llmMs)}ms"}): $reply")

                        // Kick off next turn's LLM + TTS in parallel while this turn speaks ONLY when LLM runs on NPU/GPU
                        // (on CPU-bound LLM devices/emulators, keep CPU free during speech so AudioFlinger has zero static/underruns!)
                        if (idx + 1 < turns.size && (isScriptedShowcase || !nanoLlm.usingCpuLlm)) {
                            val (nextKey, nextCue) = turns[idx + 1]
                            val historySnap = chatHistory.toList()
                            prefetched = uiScope.async(Dispatchers.IO) {
                                val (nr, nms) = if (isScriptedShowcase) {
                                    nextCue to 0.0
                                } else if (nanoLlm.isReady) {
                                    nanoLlm.replyAs(nextKey, historySnap, nextCue)
                                } else {
                                    (if (idx + 1 == 1) "That sounds awesome! I baked fresh sourdough and read all afternoon. Where was that coffee shop?"
                                    else "Right behind the bookstore — next weekend let's grab espresso and bring your sourdough!") to 1.0
                                }
                                val na = ttsEngine.synthesize(nr, nextKey)
                                Triple(nr, nms, na)
                            }
                        }

                        speakReply(speakerKey, reply, preSynthAudio = preAudio)
                        delay(120)
                    }
                    chatHistory.add(ChatTurn("System", "3-turn demo complete — ask Puck or Gretchen anything!"))
                } catch (_: CancellationException) {
                    chatHistory.add(ChatTurn("System", "3-turn demo stopped."))
                    statusBadge = "Stopped"
                } catch (e: Throwable) {
                    appendLog("DEMO ERROR: ${e.message}")
                    statusBadge = "Ready"
                } finally {
                    busy = false
                }
            }
        }

        var isListening by remember { mutableStateOf(false) }
        var listeningPartial by remember { mutableStateOf("") }
        var micRmsDb by remember { mutableStateOf(0f) }

        val voiceLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
            val spoken = res.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
            if (!spoken.isNullOrBlank()) sendQuestion(spoken)
        }

        fun startInAppSpeechRecognizer() {
            if (!SpeechRecognizer.isRecognitionAvailable(this@MainActivity)) {
                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_PROMPT, "Ask Puck or Gretchen…")
                }
                runCatching { voiceLauncher.launch(intent) }.onFailure {
                    appendLog("Speech recognition unavailable on this device/image.")
                    chatHistory.add(ChatTurn("System", "No speech recognizer found on this device."))
                    uiScope.launch { runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) } }
                }
                return
            }
            runCatching { speechRecognizer?.destroy() }
            val sr = SpeechRecognizer.createSpeechRecognizer(this@MainActivity)
            speechRecognizer = sr
            isListening = true
            listeningPartial = ""
            micRmsDb = 0f
            var peakRms = -100f
            statusBadge = "Listening…"
            sr.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    appendLog("Mic listening (SpeechRecognizer)…")
                }
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {
                    micRmsDb = rmsdB
                    if (rmsdB > peakRms) peakRms = rmsdB
                }
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    statusBadge = "Transcribing…"
                }
                override fun onError(error: Int) {
                    isListening = false
                    listeningPartial = ""
                    micRmsDb = 0f
                    val silentInput = peakRms <= 0.5f
                    val msg = when (error) {
                        SpeechRecognizer.ERROR_AUDIO -> "Audio recording error — check Emulator Extended Controls (⋯) → Microphone → enable 'Virtual microphone uses host audio input'."
                        SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> if (silentInput) {
                            "Mic received silence — in Emulator Extended Controls (⋯) → Microphone, turn ON 'Virtual microphone uses host audio input', or type in the Parked box below."
                        } else {
                            "Didn't catch that — tap mic and speak clearly, or type below."
                        }
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission required."
                        SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Speech service network error (offline pack missing)."
                        else -> "Speech recognizer error ($error)."
                    }
                    appendLog("Mic error ($error, peakRms=${"%.1f".format(peakRms)}): $msg")
                    if (error != SpeechRecognizer.ERROR_CLIENT) {
                        chatHistory.add(ChatTurn("System", msg))
                        uiScope.launch { runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) } }
                    }
                    statusBadge = if (ready) "Ready" else statusBadge
                }
                override fun onResults(results: Bundle?) {
                    isListening = false
                    micRmsDb = 0f
                    val spoken = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty().trim()
                    listeningPartial = ""
                    statusBadge = if (ready) "Ready" else statusBadge
                    if (spoken.isNotBlank()) {
                        appendLog("Heard: \"$spoken\"")
                        sendQuestion(spoken)
                    } else {
                        chatHistory.add(ChatTurn("System", "Didn't catch any words — try again or type below."))
                        uiScope.launch { runCatching { chatListState.animateScrollToItem(chatHistory.lastIndex) } }
                    }
                }
                override fun onPartialResults(partialResults: Bundle?) {
                    val partial = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
                    if (partial.isNotBlank()) listeningPartial = partial
                }
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }
            sr.startListening(intent)
        }

        val micPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startInAppSpeechRecognizer()
            } else {
                appendLog("RECORD_AUDIO permission denied.")
                chatHistory.add(ChatTurn("System", "Microphone permission denied — allow Microphone in App Settings."))
            }
        }

        val studyFileLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                runCatching {
                    val raw = contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }.orEmpty()
                    val parsed = StudyGuideParser.parse(raw, "Custom Study Guide")
                    if (parsed.cards.isNotEmpty()) {
                        studyDeck = parsed
                        quizActive = false
                        quizCardIdx = 0
                        quizMastered = 0
                        chatHistory.add(ChatTurn("System", "Loaded study guide: ${parsed.title} (${parsed.cards.size} cards). Tap ★ Quiz to begin!"))
                        appendLog("Loaded custom study guide: ${parsed.title} (${parsed.cards.size} cards)")
                    } else {
                        chatHistory.add(ChatTurn("System", "No study cards found in selected file. Use [CARD 1] / TOPIC: / FACTS: format."))
                    }
                }.onFailure { e ->
                    appendLog("Failed reading study guide: ${e.message}")
                }
            }
        }

        LaunchedEffect(Unit) {
            bootPipeline(preferGpu)
        }

        LaunchedEffect(Unit) {
            while (true) {
                refreshMem()
                delay(3000)
            }
        }

        val bg = Color(0xFF0B0E14)
        val card = Color(0xFF141923)
        val border = Color(0xFF2B3240)
        val accent = Color(0xFF58A6FF)
        val green = Color(0xFF3FB950)

        val studioState = StudioUiState(
            ready = ready,
            busy = busy,
            preferGpu = preferGpu,
            playbackMode = playbackMode,
            activeAvatar = activeAvatar,
            puckVoiceShort = puckVoiceShort,
            gretchenVoiceShort = gretchenVoiceShort,
            demoTopic = demoTopic,
            defaultDemoTopic = defaultDemoTopic,
            xShowcaseTopic = xShowcaseTopic,
            statusBadge = statusBadge,
            nanoReady = nanoLlm.isReady,
            nanoBadge = nanoBadge,
            llmEngineLabel = nanoLlm.activeEngineShort,
            inputText = inputText,
            logText = logText,
            memMain = memMain,
            memSub = memSub,
            metricLlm = metricLlm,
            metricTts = metricTts,
            metricMotion = metricMotion,
            metricRend = metricRend,
            renderSide = pipeline.renderSide,
            acceleratorLabel = pipeline.activeAcceleratorLabel,
            frameBmp = frameBmp,
            frameVersion = frameVersion,
            chatHistory = chatHistory,
            chatListState = chatListState,
            quiz = QuizUiSnapshot(
                active = quizActive,
                deckTitle = studyDeck.title,
                cardIndex = quizCardIdx,
                totalCards = studyDeck.cards.size,
                currentTopic = studyDeck.cards.getOrNull(quizCardIdx)?.topic ?: "",
                masteredCount = quizMastered,
                awaitingAnswer = quizAwaitingAnswer,
                retryCount = quizRetryCount,
                completed = quizCompleted,
            ),
            gmailAccessGranted = gmailAccessGranted,
            gmailCachedCount = gmailCachedCount,
            isListening = isListening,
            listeningPartial = listeningPartial,
            micRmsDb = micRmsDb,
        )

        val launchMic: () -> Unit = {
            if (isListening) {
                runCatching { speechRecognizer?.stopListening() }
                isListening = false
                listeningPartial = ""
                statusBadge = if (ready) "Ready" else statusBadge
            } else if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                startInAppSpeechRecognizer()
            } else {
                micPermLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
            }
        }

        val cycleAvatarVoice: (String) -> Unit = { avatarKey ->
            val short = ttsEngine.cycleVoice(avatarKey)
            if (avatarKey == "gretchen") gretchenVoiceShort = short else puckVoiceShort = short
            if (avatarKey == activeAvatar) voiceLabel = short
            ttsEngine.previewCurrentVoice(avatarKey)
        }

        if (uiMode == 2) {
            AutomotiveCockpitUi(
                state = studioState,
                isDriving = isDrivingMode,
                onToggleDriving = { isDrivingMode = !isDrivingMode },
                onSelectAvatar = { selectAvatar(it) },
                onPushToTalk = launchMic,
                onRunThreeTurnDemo = {
                    if (quizActive) endStudyQuiz()
                    runThreeTurnDemo()
                },
                onStartStudyQuiz = { startStudyQuiz() },
                onEndStudyQuiz = { endStudyQuiz() },
                onSkipQuizCard = { skipQuizCard() },
                onReadLastEmail = { readGmailBrief("Read my last email") },
                onInputChange = { inputText = it },
                onQuickAsk = { sendQuestion(it) },
                onStop = { stopActivePlayback() },
                onCycleVoice = cycleAvatarVoice,
                onPreferGpuChange = { gpu ->
                    if (!busy) {
                        preferGpu = gpu
                        uiScope.launch { bootPipeline(gpu) }
                    }
                },
                onRefreshMem = { refreshMem() },
                onSwitchUiMode = { uiMode = it },
            )
            return
        }

        if (uiMode == 0) {
            StudioUiV2(
                state = studioState,
                onSelectAvatar = { selectAvatar(it) },
                onRunThreeTurnDemo = {
                    if (quizActive) endStudyQuiz()
                    runThreeTurnDemo()
                },
                onStartStudyQuiz = { startStudyQuiz() },
                onSkipQuizCard = { skipQuizCard() },
                onEndStudyQuiz = { endStudyQuiz() },
                onLoadStudyGuideFile = { runCatching { studyFileLauncher.launch(arrayOf("text/plain", "text/markdown", "*/*")) } },
                onResetStudyGuide = {
                    studyDeck = StudyGuideParser.loadBundledSolarSystem(this@MainActivity)
                },
                onOpenGmailAccessSettings = { GmailNotificationListener.openNotificationAccessSettings(this@MainActivity) },
                onReadLastEmail = { readGmailBrief("Read my last email") },
                onStop = { stopActivePlayback() },
                onInputChange = { inputText = it },
                onSendQuestion = { sendQuestion(it) },
                onMicClick = launchMic,
                onCycleVoice = cycleAvatarVoice,
                onPlaybackModeChange = { playbackMode = it },
                onPreferGpuChange = { gpu ->
                    if (!busy) {
                        preferGpu = gpu
                        uiScope.launch { bootPipeline(gpu) }
                    }
                },
                onDemoTopicSave = { newTopic, runNow ->
                    demoTopic = newTopic
                    if (runNow && ready && !busy) runThreeTurnDemo()
                },
                onRefreshMem = { refreshMem() },
                onSwitchToClassic = { uiMode = 1 },
                onSwitchToAutoCockpit = { uiMode = 2 },
            )
            return
        }

        Surface(modifier = Modifier.fillMaxSize(), color = bg) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .navigationBarsPadding()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // Header + Live Memory HUD (Classic Console v1 + 1-tap switch back to Studio v2)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Text("Pixel Avatar", color = accent, fontSize = 19.sp, fontWeight = FontWeight.Bold)
                            Text(
                                text = "Studio UI",
                                color = Color.White,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(999.dp))
                                    .background(Color(0xFF1F6FEB))
                                    .clickable { uiMode = 0 }
                                    .padding(horizontal = 8.dp, vertical = 3.dp),
                            )
                        }
                        Text(nanoBadge, color = if (nanoLlm.isReady) green else Color(0xFFD29922), fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                    }
                    Column(
                        horizontalAlignment = Alignment.End,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .background(card)
                            .border(1.dp, border, RoundedCornerShape(8.dp))
                            .clickable { refreshMem() }
                            .padding(horizontal = 10.dp, vertical = 5.dp),
                    ) {
                        Text(memMain, color = green, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                        Text(memSub, color = Color(0xFF8B949E), fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                    }
                }

                // 512x512 Live Avatar Stage
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(265.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(card)
                        .border(1.dp, border, RoundedCornerShape(12.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    val bmp = frameBmp
                    if (bmp != null) {
                        key(frameVersion) {
                            Image(
                                bitmap = bmp.asImageBitmap(),
                                contentDescription = "Live 512x512 Avatar",
                                modifier = Modifier
                                    .size(252.dp)
                                    .clip(RoundedCornerShape(10.dp)),
                                contentScale = ContentScale.Crop,
                            )
                        }
                    }
                    Text(
                        text = statusBadge,
                        color = green,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(10.dp)
                            .clip(RoundedCornerShape(999.dp))
                            .background(Color(0xD90B0E14))
                            .border(1.dp, border, RoundedCornerShape(999.dp))
                            .padding(horizontal = 9.dp, vertical = 4.dp),
                    )
                }

                // Controls Row 1: Puck / Gretchen / 3-Turn Demo (single-line, zero wrap)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Button(
                        onClick = { selectAvatar("puck") },
                        enabled = ready && !busy,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (activeAvatar == "puck") Color(0xFF1F6FEB) else Color(0xFF212836),
                            contentColor = Color.White,
                            disabledContainerColor = Color(0xFF1A202C),
                            disabledContentColor = Color(0xFF8B949E),
                        ),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                        modifier = Modifier.height(38.dp),
                    ) { Text("Puck", fontSize = 12.sp, maxLines = 1, softWrap = false) }

                    Button(
                        onClick = { selectAvatar("gretchen") },
                        enabled = ready && !busy,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (activeAvatar == "gretchen") Color(0xFF1F6FEB) else Color(0xFF212836),
                            contentColor = Color.White,
                            disabledContainerColor = Color(0xFF1A202C),
                            disabledContentColor = Color(0xFF8B949E),
                        ),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                        modifier = Modifier.height(38.dp),
                    ) { Text("Gretchen", fontSize = 12.sp, maxLines = 1, softWrap = false) }

                    Button(
                        onClick = { runThreeTurnDemo() },
                        enabled = ready && !busy,
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF238636),
                            contentColor = Color.White,
                            disabledContainerColor = Color(0xFF1A202C),
                            disabledContentColor = Color(0xFF8B949E),
                        ),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        modifier = Modifier.weight(1f).height(38.dp),
                    ) { Text("3-Turn Demo", fontSize = 12.sp, maxLines = 1, softWrap = false) }

                    OutlinedButton(
                        onClick = { showTopicDialog = true },
                        enabled = !busy,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                        modifier = Modifier.height(38.dp),
                    ) {
                        Text(
                            "Topic",
                            fontSize = 11.sp,
                            maxLines = 1,
                            softWrap = false,
                            color = if (demoTopic != defaultDemoTopic) green else accent,
                        )
                    }
                }

                // Controls Row 2: Voice Cycle + Playback FPS Mode + Log + Accelerator
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(5.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    OutlinedButton(
                        onClick = {
                            voiceLabel = ttsEngine.cycleVoice(activeAvatar)
                            if (activeAvatar == "gretchen") gretchenVoiceShort = voiceLabel else puckVoiceShort = voiceLabel
                            ttsEngine.previewCurrentVoice(activeAvatar)
                        },
                        enabled = ready && !busy,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                        modifier = Modifier.weight(1.1f).height(34.dp),
                    ) {
                        Text(
                            "Voice: $voiceLabel",
                            fontSize = 11.sp,
                            maxLines = 1,
                            softWrap = false,
                            color = green,
                        )
                    }

                    val modeText = when (playbackMode) {
                        0 -> "12.5FPS"
                        1 -> "25FPS"
                        else -> "Live"
                    }
                    OutlinedButton(
                        onClick = { playbackMode = (playbackMode + 1) % 3 },
                        enabled = !busy,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp),
                    ) {
                        Text(
                            modeText,
                            fontSize = 11.sp,
                            maxLines = 1,
                            softWrap = false,
                            color = if (playbackMode != 2) green else accent,
                        )
                    }

                    OutlinedButton(
                        onClick = { showLog = !showLog },
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp),
                    ) { Text(if (showLog) "Hide" else "Log", fontSize = 11.sp, maxLines = 1, softWrap = false, color = Color(0xFF8B949E)) }

                    OutlinedButton(
                        onClick = {
                            if (!busy) {
                                preferGpu = !preferGpu
                                uiScope.launch { bootPipeline(preferGpu) }
                            }
                        },
                        enabled = !busy,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp),
                    ) { Text(if (preferGpu) "GPU" else "CPU", fontSize = 11.sp, maxLines = 1, softWrap = false, color = accent) }
                }

                // 4 Live Metrics
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    MetricBox("Nano LLM", metricLlm, Modifier.weight(1f))
                    MetricBox("TTS", metricTts, Modifier.weight(1f))
                    MetricBox("Motion", metricMotion, Modifier.weight(1f))
                    MetricBox("Rend ${pipeline.renderSide}", metricRend, Modifier.weight(1.3f))
                }

                // Shared Chat Box
                LazyColumn(
                    state = chatListState,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .clip(RoundedCornerShape(10.dp))
                        .background(card)
                        .border(1.dp, border, RoundedCornerShape(10.dp))
                        .padding(8.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    if (chatHistory.isEmpty()) {
                        item {
                            Text(
                                "100% offline Pixel 10 Studio: Gemini Nano (AICore) -> Android TTS -> LiteRT GPU IMTalker. Ask Puck or Gretchen anything!",
                                color = Color(0xFF8B949E),
                                fontSize = 12.sp,
                            )
                        }
                    }
                    items(chatHistory) { turn ->
                        val bubbleColor = when (turn.speaker) {
                            "You" -> Color(0xFF1F6FEB)
                            "System" -> Color.Transparent
                            else -> if (turn.avatarKey == "gretchen") Color(0xFF261D2E) else Color(0xFF1C2433)
                        }
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(bubbleColor)
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                        ) {
                            if (turn.speaker != "System") {
                                Text(turn.speaker.uppercase(), color = Color(0xB3E6EDF3), fontSize = 10.sp, fontWeight = FontWeight.Bold)
                            }
                            Text(
                                turn.text,
                                color = if (turn.speaker == "System") Color(0xFF8B949E) else Color(0xFFE6EDF3),
                                fontSize = 13.sp,
                            )
                        }
                    }
                }

                // Collapsible Engine Log
                if (showLog) {
                    Text(
                        text = logText.lines().takeLast(6).joinToString("\n"),
                        color = Color(0xFFC9D1D9),
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        lineHeight = 13.sp,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(Color(0xFF0F131C))
                            .border(1.dp, border, RoundedCornerShape(8.dp))
                            .clickable { showLog = !showLog }
                            .padding(6.dp),
                    )
                }

                // Input Row + Mic + Send + Stop
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { Text("Ask Puck or Gretchen…", fontSize = 13.sp) },
                        singleLine = true,
                        modifier = Modifier.weight(1f),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            focusedBorderColor = accent,
                            unfocusedBorderColor = border,
                        ),
                    )
                    Button(
                        onClick = { sendQuestion(inputText) },
                        enabled = ready && !busy && inputText.isNotBlank(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF238636),
                            contentColor = Color.White,
                            disabledContainerColor = Color(0xFF1A202C),
                            disabledContentColor = Color(0xFF8B949E),
                        ),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 12.dp),
                    ) { Text("Send", fontSize = 12.sp) }

                    Button(
                        onClick = {
                            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                                putExtra(RecognizerIntent.EXTRA_PROMPT, "Ask Puck or Gretchen…")
                            }
                            runCatching { voiceLauncher.launch(intent) }
                        },
                        enabled = ready && !busy,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF212836)),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 12.dp),
                    ) { Text("Mic", fontSize = 12.sp) }

                    if (busy) {
                        Button(
                            onClick = { stopActivePlayback() },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFB62324)),
                            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 12.dp),
                        ) { Text("Stop", fontSize = 12.sp) }
                    }
                }
            }

            if (showTopicDialog) {
                var draft by remember(demoTopic) { mutableStateOf(demoTopic) }
                val presets = listOf(
                    xShowcaseTopic,
                    defaultDemoTopic,
                    "Coffee vs tea — which is better?",
                    "Favorite movie of all time",
                    "Running AI 100% offline on Pixel 10",
                )
                AlertDialog(
                    onDismissRequest = { showTopicDialog = false },
                    containerColor = card,
                    titleContentColor = Color.White,
                    textContentColor = Color(0xFFC9D1D9),
                    title = { Text("3-Turn Demo Topic", fontSize = 16.sp, fontWeight = FontWeight.Bold) },
                    text = {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(
                                value = draft,
                                onValueChange = { draft = it },
                                label = { Text("Conversation prompt / topic", fontSize = 12.sp) },
                                singleLine = false,
                                maxLines = 3,
                                modifier = Modifier.fillMaxWidth(),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedTextColor = Color.White,
                                    unfocusedTextColor = Color.White,
                                    focusedBorderColor = accent,
                                    unfocusedBorderColor = border,
                                ),
                            )
                            Text("Quick Presets:", fontSize = 11.sp, color = Color(0xFF8B949E))
                            for (p in presets) {
                                Text(
                                    text = "• $p",
                                    color = if (draft == p) green else accent,
                                    fontSize = 12.sp,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(6.dp))
                                        .background(Color(0xFF1C2433))
                                        .clickable { draft = p }
                                        .padding(horizontal = 8.dp, vertical = 6.dp),
                                )
                            }
                        }
                    },
                    confirmButton = {
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            TextButton(onClick = {
                                draft = defaultDemoTopic
                                demoTopic = defaultDemoTopic
                            }) { Text("Reset Default", color = Color(0xFFD29922), fontSize = 12.sp) }
                            Button(
                                onClick = {
                                    demoTopic = draft.trim().ifBlank { defaultDemoTopic }
                                    showTopicDialog = false
                                    if (ready && !busy) runThreeTurnDemo()
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF238636)),
                                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                            ) { Text("Save & Run", fontSize = 12.sp) }
                        }
                    },
                    dismissButton = {
                        TextButton(onClick = {
                            demoTopic = draft.trim().ifBlank { defaultDemoTopic }
                            showTopicDialog = false
                        }) { Text("Save", color = accent, fontSize = 12.sp) }
                    },
                )
            }
        }
    }

    @Composable
    private fun MetricBox(label: String, value: String, modifier: Modifier = Modifier) {
        Column(
            modifier = modifier
                .clip(RoundedCornerShape(8.dp))
                .background(Color(0xFF0F131C))
                .border(1.dp, Color(0xFF232A36), RoundedCornerShape(8.dp))
                .padding(horizontal = 6.dp, vertical = 5.dp),
        ) {
            Text(label, color = Color(0xFF8B949E), fontSize = 9.sp, fontFamily = FontFamily.Monospace)
            Text(value, color = Color(0xFF79C0FF), fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, maxLines = 1)
        }
    }
}

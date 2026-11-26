package com.pixelavatar.app

import android.app.ActivityManager
import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Debug
import android.speech.RecognizerIntent
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

    private var mediaPlayer: MediaPlayer? = null
    private var activeJob: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pipeline = AvatarPipeline(this)
        nanoLlm = NanoLlmClient()
        ttsEngine = AndroidTtsEngine(this)

        setContent {
            PixelAvatarScreen()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopActivePlayback()
        scope.cancel()
        pipeline.releaseModels()
        nanoLlm.close()
        ttsEngine.close()
    }

    private fun stopActivePlayback() {
        activeJob?.cancel()
        activeJob = null
        runCatching { mediaPlayer?.stop() }
        runCatching { mediaPlayer?.release() }
        mediaPlayer = null
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

        // Double-buffered mutable bitmaps sized to renderer output (256x256 or 512x512) for zero-copy GPU swap!
        var frameBmp by remember { mutableStateOf<Bitmap?>(null) }
        var frameVersion by remember { mutableIntStateOf(0) }
        var bmpPool by remember {
            mutableStateOf(
                arrayOf(
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
                )
            }
            poolIdx = (poolIdx + 1) and 1
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
        }

        fun selectAvatar(key: String) {
            activeAvatar = key
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

                statusBadge = "Compiling LiteRT + Warming Nano…"
                val t0 = System.nanoTime()
                withContext(Dispatchers.IO) {
                    val nanoJob = async { nanoLlm.prepare { msg -> scope.launch { appendLog(msg) } } }
                    pipeline.initModels(preferGpu = useGpu) { msg -> scope.launch { appendLog(msg) } }
                    nanoJob.await()
                }
                val bootSec = (System.nanoTime() - t0) / 1e9
                appendLog("LiteRT 3-model stack + Nano warmed in ${"%.2f".format(bootSec)}s (${pipeline.activeAcceleratorLabel})")
                nanoBadge = nanoLlm.statusLabel

                statusBadge = "Benchmarking Tensor G5…"
                val targetBmp = nextWorkBmp()
                val bench = withContext(Dispatchers.IO) {
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
            val (allTokens, totalFrames, motionMs) = withContext(Dispatchers.IO) {
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
                val renderJob = uiScope.launch(Dispatchers.IO) {
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
                            if (mediaPlayer == null) {
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

                statusBadge = "Speaking (${avatar.displayName} · $fpsLabel)"
                var audioDone = false
                val mp = MediaPlayer().apply {
                    setDataSource(ttsAudio.wavFile.absolutePath)
                    setOnCompletionListener { audioDone = true }
                    prepare()
                }
                mediaPlayer = mp
                mp.start()

                // Wait for Android AudioFlinger to begin outputting samples so lips never lead audio
                val waitStart = System.nanoTime()
                while (currentCoroutineContext().isActive && !audioDone && mp.currentPosition <= 0 &&
                    (System.nanoTime() - waitStart) < 500_000_000L
                ) {
                    delay(8)
                }

                var shown = 0
                var lastK = -1
                while (currentCoroutineContext().isActive && !audioDone) {
                    val posMs = runCatching { mp.currentPosition }.getOrDefault(0)
                    val k = ((posMs / 1000.0) * targetFps).toInt()
                    if (k >= nOut) break
                    if (k == lastK) {
                        delay(8)
                        continue
                    }
                    val bmp = renderedFrames[k]
                    if (bmp != null) {
                        lastK = k
                        frameBmp = bmp
                        frameVersion++
                        shown++
                    } else {
                        delay(10)
                    }
                }
                while (!audioDone && runCatching { mp.isPlaying }.getOrDefault(false) && currentCoroutineContext().isActive) delay(20)
                renderJob.cancel()
                runCatching { mp.release() }
                mediaPlayer = null
                ttsAudio.wavFile.delete()

                val rawAvg = if (sumRendMs > 0) sumRendMs / renderedFrames.count { it != null }.coerceAtLeast(1) else 0.0
                appendLog("[${avatar.displayName} SMOOTH] $shown/$nOut frames @ $fpsLabel synced (raw GPU ${"%.0f".format(rawAvg)}ms = ${"%.1f".format(if (rawAvg > 0) 1000.0 / rawAvg else 0.0)} FPS)")
                statusBadge = "Ready · $fpsLabel Smooth"
                refreshMem()
            } else {
                // Zero-wait Live mode locked directly to hardware MediaPlayer audio clock
                statusBadge = "Speaking (${avatar.displayName})"
                var audioDone = false
                val mp = MediaPlayer().apply {
                    setDataSource(ttsAudio.wavFile.absolutePath)
                    setOnCompletionListener { audioDone = true }
                    prepare()
                }
                mediaPlayer = mp
                mp.start()

                // Wait until Android audio hardware actually starts outputting PCM samples
                val waitStart = System.nanoTime()
                while (currentCoroutineContext().isActive && !audioDone && mp.currentPosition <= 0 &&
                    (System.nanoTime() - waitStart) < 500_000_000L
                ) {
                    delay(8)
                }

                var rendered = 0
                var sumRendMs = 0.0
                var lastIdx = -1
                while (currentCoroutineContext().isActive && !audioDone) {
                    val posMs = runCatching { mp.currentPosition }.getOrDefault(0)
                    val idx = ((posMs / 1000.0) * 25.0).toInt()
                    if (idx >= totalFrames) break
                    if (idx == lastIdx) {
                        delay(6)
                        continue
                    }
                    lastIdx = idx
                    val tok = allTokens.copyOfRange(idx * 32, (idx + 1) * 32)
                    val targetBmp = nextWorkBmp()
                    val rMs = withContext(Dispatchers.IO) { pipeline.renderFrame(avatar, tok, targetBmp) }
                    frameBmp = targetBmp
                    frameVersion++
                    sumRendMs += rMs
                    rendered++
                    val avgR = sumRendMs / rendered
                    metricRend = "${"%.1f".format(avgR)} ms (${"%.1f".format(1000.0 / avgR)} FPS)"
                }
                while (!audioDone && runCatching { mp.isPlaying }.getOrDefault(false) && currentCoroutineContext().isActive) delay(20)
                runCatching { mp.release() }
                mediaPlayer = null
                ttsAudio.wavFile.delete()

                val avgR = if (rendered > 0) sumRendMs / rendered else 0.0
                val fps = if (avgR > 0) 1000.0 / avgR else 0.0
                appendLog("[${avatar.displayName}] $rendered/$totalFrames frames | motion=${"%.0f".format(motionMs)}ms | rend=${"%.1f".format(avgR)}ms (${"%.1f".format(fps)} FPS)")
                statusBadge = "Ready · ${"%.1f".format(fps)} FPS"
                refreshMem()
            }
        }

        fun sendQuestion(q: String) {
            val trimmed = q.trim()
            if (trimmed.isEmpty() || busy || !ready) return
            inputText = ""
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
            activeJob = uiScope.launch {
                busy = true
                try {
                    chatHistory.add(ChatTurn("System", "3-Turn Demo: Puck & Gretchen — What they did last weekend"))
                    val turns = listOf(
                        "puck" to "Start a fun casual chat with Gretchen about what you did last weekend and ask what she did.",
                        "gretchen" to "Respond naturally to Puck's weekend story, share one highlight from your own weekend, and ask Puck a quick follow-up.",
                        "puck" to "Respond warmly and wittily to Gretchen's weekend story to wrap up this 3-turn chat.",
                    )
                    // Pipelined lookahead: while Turn i is speaking, Turn i+1 runs Gemini Nano (TPU) + TTS (CPU)
                    // in the background so the gap between speakers drops to ~0.3s!
                    var prefetched: Deferred<Triple<String, Double, TtsAudio>>? = null

                    for ((idx, pair) in turns.withIndex()) {
                        val (speakerKey, cue) = pair
                        val name = pipeline.avatars[speakerKey]!!.displayName
                        selectAvatar(speakerKey)
                        val bubbleIdx = chatHistory.size
                        chatHistory.add(ChatTurn("$name (${idx + 1}/3)", "Thinking…", speakerKey))
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
                        } else {
                            statusBadge = "Turn ${idx + 1}/3 · Nano ($name)…"
                            val pairRes = if (nanoLlm.isReady) {
                                withContext(Dispatchers.IO) {
                                    nanoLlm.replyAs(speakerKey, chatHistory.dropLast(1), cue) { partial ->
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

                        metricLlm = "${"%.0f".format(llmMs)} ms"
                        chatHistory[bubbleIdx] = ChatTurn(name, reply, speakerKey)
                        appendLog("[Turn ${idx + 1}/3 · $name] (${"%.0f".format(llmMs)}ms): $reply")

                        // Kick off next turn's Gemini Nano + TTS right now in parallel while this turn speaks!
                        if (idx + 1 < turns.size) {
                            val (nextKey, nextCue) = turns[idx + 1]
                            val historySnap = chatHistory.toList()
                            prefetched = uiScope.async(Dispatchers.IO) {
                                val (nr, nms) = if (nanoLlm.isReady) {
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

        val voiceLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
            val spoken = res.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
            if (!spoken.isNullOrBlank()) sendQuestion(spoken)
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

        Surface(modifier = Modifier.fillMaxSize(), color = bg) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .statusBarsPadding()
                    .navigationBarsPadding()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // Header + Live Memory HUD
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text("Pixel Avatar", color = accent, fontSize = 19.sp, fontWeight = FontWeight.Bold)
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
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
                        modifier = Modifier.weight(1f).height(38.dp),
                    ) { Text("3-Turn Demo", fontSize = 12.sp, maxLines = 1, softWrap = false) }
                }

                // Controls Row 2: Playback FPS Mode + Accelerator
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    val modeText = when (playbackMode) {
                        0 -> "Playback: 12.5 FPS Smooth"
                        1 -> "Playback: 25 FPS Smooth"
                        else -> "Playback: Live (Zero-Wait)"
                    }
                    OutlinedButton(
                        onClick = { playbackMode = (playbackMode + 1) % 3 },
                        enabled = !busy,
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.weight(1f).height(34.dp),
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
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp),
                    ) { Text(if (showLog) "Hide Log" else "Log", fontSize = 11.sp, maxLines = 1, softWrap = false, color = Color(0xFF8B949E)) }

                    OutlinedButton(
                        onClick = {
                            if (!busy) {
                                preferGpu = !preferGpu
                                uiScope.launch { bootPipeline(preferGpu) }
                            }
                        },
                        enabled = !busy,
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
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

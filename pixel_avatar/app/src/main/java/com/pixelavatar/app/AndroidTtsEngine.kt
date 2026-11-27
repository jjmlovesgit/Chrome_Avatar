package com.pixelavatar.app

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.texttospeech.SpeechSynthesis
import com.google.mlkit.genai.texttospeech.SpeechSynthesizer
import com.google.mlkit.genai.texttospeech.SpeechSynthesizerOptions
import com.google.mlkit.genai.texttospeech.VoiceInfo
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.UUID

data class TtsAudio(
    val wavFile: File,
    val pcm16k: FloatArray,
    val durationSec: Double,
    val synthMs: Double,
    val pcmRawBytes: ByteArray,
    val rawSampleRate: Int,
    val rawTotalFrames: Int,
)

class AndroidTtsEngine(private val context: Context) {

    private var tts: TextToSpeech? = null
    private var genaiClient: SpeechSynthesizer? = null

    private var puckVoice: Voice? = null
    private var gretchenVoice: Voice? = null
    private var availableVoices: List<Voice> = emptyList()
    private var usingGenAiVoices: Boolean = false

    var puckVoiceName: String = "default"
        private set
    var gretchenVoiceName: String = "default"
        private set
    var statusLabel: String = "Checking Voices…"
        private set

    fun shortVoiceName(avatarKey: String): String {
        val raw = if (avatarKey == "gretchen") gretchenVoiceName else puckVoiceName
        return raw.removePrefix("en-us-x-").removeSuffix("-local").ifBlank { raw }
    }

    private val prefs by lazy { context.getSharedPreferences("pixel_avatar_voices", Context.MODE_PRIVATE) }

    fun cycleVoice(avatarKey: String): String {
        if (availableVoices.isEmpty()) return shortVoiceName(avatarKey)
        val isGretchen = avatarKey == "gretchen"
        val cur = if (isGretchen) gretchenVoice else puckVoice
        val curIdx = availableVoices.indexOfFirst { it.name == cur?.name }
        val next = availableVoices[(curIdx + 1) % availableVoices.size]
        if (isGretchen) {
            gretchenVoice = next
            gretchenVoiceName = next.name
            prefs.edit().putString("voice_gretchen", next.name).apply()
        } else {
            puckVoice = next
            puckVoiceName = next.name
            prefs.edit().putString("voice_puck", next.name).apply()
        }
        android.util.Log.i("PixelAvatar", "Voice switched for $avatarKey -> ${next.name}")
        return shortVoiceName(avatarKey)
    }

    fun previewCurrentVoice(avatarKey: String) {
        val engine = tts ?: return
        val isGretchen = avatarKey == "gretchen"
        val voice = if (isGretchen) gretchenVoice else puckVoice
        if (voice != null) runCatching { engine.voice = voice }
        engine.setPitch(1.0f)
        engine.setSpeechRate(1.0f)
        val line = if (isGretchen) "Hi, I'm Gretchen." else "Hey, I'm Puck."
        runCatching { engine.speak(line, TextToSpeech.QUEUE_FLUSH, null, "preview_${System.currentTimeMillis()}") }
    }

    private val readyDeferred = CompletableDeferred<Boolean>()
    private val pending = mutableMapOf<String, CompletableDeferred<Boolean>>()
    private val synthMutex = Mutex()

    init {
        // Explicitly bind to Google Speech Services (com.google.android.tts) where ML Kit GenAI voices live
        tts = TextToSpeech(context, { status ->
            if (status == TextToSpeech.SUCCESS) {
                val engine = tts
                engine?.language = Locale.US
                selectFallbackVoices(engine)
                engine?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        synchronized(pending) { pending.remove(utteranceId) }?.complete(true)
                    }
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        synchronized(pending) { pending.remove(utteranceId) }?.complete(false)
                    }
                    override fun onError(utteranceId: String?, errorCode: Int) {
                        synchronized(pending) { pending.remove(utteranceId) }?.complete(false)
                    }
                })
                readyDeferred.complete(true)
            } else {
                readyDeferred.complete(false)
            }
        }, "com.google.android.tts")
    }

    /**
     * Discover, download (if needed), and bind Google's on-device ML Kit GenAI 24 kHz
     * generative diffusion voices (star*) for Puck (male) and Gretchen (female).
     */
    suspend fun prepareGenAiVoices(onStatus: (String) -> Unit): Boolean {
        readyDeferred.await()
        return try {
            val opts = SpeechSynthesizerOptions.builder(Locale.US)
                .setVoiceFallbackEnabled(true)
                .build()
            val client = SpeechSynthesis.getClient(opts)
            genaiClient = client

            val supported = client.getSupportedVoices(Locale.US)
            onStatus("ML Kit GenAI TTS supported en-US voices (${supported.size}): ${supported.joinToString { "${it.name}(${it.gender})" }}")
            if (supported.isEmpty()) {
                statusLabel = "Android Neural TTS"
                return false
            }

            // Pick natural distinct male (Puck) & female (Gretchen) generative diffusion voices
            val males = supported.filter { it.gender == VoiceInfo.Gender.MALE }
            val females = supported.filter { it.gender == VoiceInfo.Gender.FEMALE }

            val puckInfo = males.firstOrNull { it.name.contains("star11", ignoreCase = true) }
                ?: males.firstOrNull { it.name.contains("star13", ignoreCase = true) }
                ?: males.firstOrNull { it.name.contains("star04", ignoreCase = true) }
                ?: males.firstOrNull()
                ?: supported.first()

            val gretchenInfo = females.firstOrNull { it.name.contains("star02", ignoreCase = true) }
                ?: females.firstOrNull { it.name.contains("star12", ignoreCase = true) }
                ?: females.firstOrNull { it.name.contains("star16", ignoreCase = true) }
                ?: females.firstOrNull { it.name != puckInfo.name }
                ?: supported.firstOrNull { it.name != puckInfo.name }
                ?: puckInfo

            suspend fun ensureDownloaded(label: String, info: VoiceInfo): Boolean {
                val st = client.checkStatus(info)
                onStatus("GenAI Voice [$label=${info.name}]: status=$st")
                return when (st) {
                    FeatureStatus.AVAILABLE -> true
                    FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> {
                        onStatus("Downloading GenAI Voice $label (${info.name})…")
                        var ok = false
                        client.download(info).collect { dl ->
                            when (dl) {
                                is DownloadStatus.DownloadStarted ->
                                    onStatus("GenAI Voice ${info.name} download started…")
                                is DownloadStatus.DownloadProgress -> {
                                    val mb = dl.totalBytesDownloaded / (1024.0 * 1024.0)
                                    statusLabel = "Voice ${info.name} ${"%.0f".format(mb)} MB"
                                }
                                is DownloadStatus.DownloadCompleted -> {
                                    onStatus("GenAI Voice ${info.name} downloaded!")
                                    ok = true
                                }
                                is DownloadStatus.DownloadFailed -> {
                                    onStatus("GenAI Voice ${info.name} download failed: ${dl.e.message}")
                                }
                            }
                        }
                        ok || (client.checkStatus(info) == FeatureStatus.AVAILABLE)
                    }
                    else -> false
                }
            }

            val puckOk = ensureDownloaded("Puck", puckInfo)
            val gretOk = ensureDownloaded("Gretchen", gretchenInfo)

            // Match the exact GenAI voice inside com.google.android.tts (same matching ML Kit uses internally)
            val engine = tts
            val allVoices = runCatching { engine?.voices }.getOrNull().orEmpty()

            fun findTtsVoice(info: VoiceInfo): Voice? {
                return allVoices.firstOrNull {
                    it.name.removeSuffix("-local").equals(info.name, ignoreCase = true)
                } ?: allVoices.firstOrNull {
                    it.name.contains(info.name, ignoreCase = true)
                }
            }

            val pv = if (puckOk) findTtsVoice(puckInfo) else null
            val gv = if (gretOk) findTtsVoice(gretchenInfo) else null

            if (pv != null) {
                puckVoice = pv
                puckVoiceName = pv.name
            }
            if (gv != null) {
                gretchenVoice = gv
                gretchenVoiceName = gv.name
            }

            usingGenAiVoices = (pv != null && gv != null)
            if (usingGenAiVoices) {
                statusLabel = "GenAI 24kHz Voices"
                onStatus("ML Kit GenAI 24kHz Diffusion Voices ACTIVE: Puck=$puckVoiceName | Gretchen=$gretchenVoiceName")
            } else {
                // Also log any star* voices already visible in TextToSpeech.voices
                val starVoices = allVoices.filter { it.name.contains("star", ignoreCase = true) }
                onStatus("Visible star* voices in Android TTS (${starVoices.size}): ${starVoices.joinToString { it.name }}")
                statusLabel = if (pv != null || gv != null) "GenAI Voice Partial" else "Android Neural TTS"
            }
            usingGenAiVoices
        } catch (e: Throwable) {
            onStatus("ML Kit GenAI TTS fallback to Android TTS: ${e.message}")
            statusLabel = "Android Neural TTS"
            false
        }
    }

    private fun selectFallbackVoices(engine: TextToSpeech?) {
        val voices = runCatching { engine?.voices }.getOrNull() ?: return
        val enUsLocal = voices.filter {
            it.locale?.language == "en" && it.locale?.country == "US" &&
                !it.isNetworkConnectionRequired && !it.name.equals("en-US-language", ignoreCase = true)
        }.sortedWith(compareByDescending<Voice> { it.name.contains("star", ignoreCase = true) }
            .thenByDescending { it.quality }
            .thenBy { it.name })
        availableVoices = enUsLocal

        android.util.Log.i(
            "PixelAvatar",
            "Local en-US voices (${enUsLocal.size}): " + enUsLocal.joinToString { "${it.name}(q=${it.quality})" },
        )

        // Prefer any already-downloaded GenAI star* voices first!
        val starEnUs = enUsLocal.filter { it.name.contains("star", ignoreCase = true) }
        if (starEnUs.size >= 2) {
            puckVoice = starEnUs.firstOrNull { it.name.contains("star11") || it.name.contains("star13") || it.name.contains("star04") }
                ?: starEnUs.first()
            gretchenVoice = starEnUs.firstOrNull { it != puckVoice && (it.name.contains("star02") || it.name.contains("star12") || it.name.contains("star16")) }
                ?: starEnUs.firstOrNull { it != puckVoice }
                ?: puckVoice
            puckVoiceName = puckVoice?.name ?: "star-male"
            gretchenVoiceName = gretchenVoice?.name ?: "star-female"
            usingGenAiVoices = true
            return
        }

        // Pick user's saved or preferred Google SEANet Neural local voices (default: Puck=iom, Gretchen=tpc)
        val savedPuck = prefs.getString("voice_puck", null)
        val savedGretchen = prefs.getString("voice_gretchen", null)
        puckVoice = savedPuck?.let { name -> enUsLocal.firstOrNull { it.name.equals(name, ignoreCase = true) } }
            ?: enUsLocal.firstOrNull { it.name.contains("iom", ignoreCase = true) }
            ?: enUsLocal.firstOrNull { it.name.contains("tpd", ignoreCase = true) || it.name.contains("iol", ignoreCase = true) }
            ?: enUsLocal.firstOrNull()
        gretchenVoice = savedGretchen?.let { name -> enUsLocal.firstOrNull { it.name.equals(name, ignoreCase = true) } }
            ?: enUsLocal.firstOrNull { it != puckVoice && it.name.contains("tpc", ignoreCase = true) }
            ?: enUsLocal.firstOrNull { it != puckVoice && (it.name.contains("iog", ignoreCase = true) || it.name.contains("ioj", ignoreCase = true) || it.name.contains("tpf", ignoreCase = true) || it.name.contains("iob", ignoreCase = true)) }
            ?: enUsLocal.firstOrNull { it != puckVoice }
            ?: puckVoice
        puckVoiceName = puckVoice?.name ?: "default"
        gretchenVoiceName = gretchenVoice?.name ?: "default"
        android.util.Log.i("PixelAvatar", "Selected local voices: Puck=$puckVoiceName | Gretchen=$gretchenVoiceName")
    }

    suspend fun synthesize(text: String, avatarKey: String): TtsAudio = synthMutex.withLock {
        val ok = readyDeferred.await()
        val engine = requireNotNull(tts) { "Android TTS not initialized" }
        require(ok) { "Android TTS initialization failed" }

        val isGretchen = avatarKey == "gretchen"
        val voice = if (isGretchen) gretchenVoice else puckVoice
        if (voice != null) {
            runCatching { engine.voice = voice }
        }

        // Keep natural 1.0 pitch for Google SEANet & GenAI voices so prosody stays clean and unwarped!
        engine.setPitch(1.0f)
        engine.setSpeechRate(1.0f)

        val outFile = File(context.cacheDir, "tts_${System.currentTimeMillis()}_${avatarKey}.wav")
        val uttId = UUID.randomUUID().toString()
        val done = CompletableDeferred<Boolean>()
        synchronized(pending) { pending[uttId] = done }

        val params = Bundle().apply {
            // Match ML Kit GenAI TTS bundle hint for high-quality local voice without silent network fallback
            putString("com.google.android.tts:DisableVoiceFallback", "false")
        }

        android.util.Log.i("PixelAvatar", "TTS synth [$avatarKey] voice=${voice?.name ?: "default"}")
        val t0 = System.nanoTime()
        val res = engine.synthesizeToFile(text, params, outFile, uttId)
        require(res == TextToSpeech.SUCCESS) { "TTS synthesizeToFile returned $res" }
        val finished = done.await()
        val synthMs = (System.nanoTime() - t0) / 1e6
        require(finished && outFile.exists() && outFile.length() > 44) { "TTS synthesis failed" }

        val decoded = decodeWavTo16kMono(outFile)
        outFile.delete()
        return@withLock TtsAudio(
            wavFile = outFile,
            pcm16k = decoded.pcm16k,
            durationSec = decoded.durationSec,
            synthMs = synthMs,
            pcmRawBytes = decoded.pcmRawBytes,
            rawSampleRate = decoded.rawSampleRate,
            rawTotalFrames = decoded.rawTotalFrames,
        )
    }

    private data class DecodedWav(
        val pcm16k: FloatArray,
        val durationSec: Double,
        val pcmRawBytes: ByteArray,
        val rawSampleRate: Int,
        val rawTotalFrames: Int,
    )

    private fun decodeWavTo16kMono(file: File): DecodedWav {
        val bytes = file.readBytes()
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12
        var channels = 1
        var sampleRate = 24000
        var bitsPerSample = 16
        var dataOffset = 44
        var dataSize = bytes.size - 44

        while (pos + 8 <= bytes.size) {
            val chunkId = String(bytes, pos, 4, Charsets.US_ASCII)
            val chunkSize = bb.getInt(pos + 4)
            if (chunkId == "fmt " && pos + 24 <= bytes.size) {
                channels = bb.getShort(pos + 10).toInt()
                sampleRate = bb.getInt(pos + 12)
                bitsPerSample = bb.getShort(pos + 22).toInt()
            } else if (chunkId == "data") {
                dataOffset = pos + 8
                dataSize = chunkSize.coerceAtMost(bytes.size - dataOffset)
                break
            }
            pos += 8 + chunkSize + (chunkSize and 1)
        }

        val bytesPerSample = (bitsPerSample / 8).coerceAtLeast(2)
        val totalFrames = dataSize / (channels * bytesPerSample)
        val mono = FloatArray(totalFrames)
        val monoPcm16Bytes = ByteArray(totalFrames * 2)
        val outBb = ByteBuffer.wrap(monoPcm16Bytes).order(ByteOrder.LITTLE_ENDIAN)
        // Soft 6ms fade-in/out + -1.4 dB headroom prevents digital clipping in 24kHz->48kHz resamplers & clicks
        val fadeFrames = (sampleRate * 0.006).toInt().coerceIn(1, (totalFrames / 4).coerceAtLeast(1))
        val headroom = 0.85f
        for (i in 0 until totalFrames) {
            var acc = 0f
            for (c in 0 until channels) {
                val off = dataOffset + (i * channels + c) * bytesPerSample
                val s = bb.getShort(off).toInt()
                acc += s / 32768f
            }
            val avg = acc / channels
            mono[i] = avg
            val env = when {
                i < fadeFrames -> i.toFloat() / fadeFrames
                i >= totalFrames - fadeFrames -> (totalFrames - 1 - i).toFloat() / fadeFrames
                else -> 1f
            }
            val s16 = (avg * headroom * env * 32767f).toInt().coerceIn(-32768, 32767).toShort()
            outBb.putShort(s16)
        }
        val durationSec = totalFrames.toDouble() / sampleRate.toDouble()
        val pcm16k = if (sampleRate == 16000) {
            mono
        } else {
            val targetLen = (durationSec * 16000.0).toInt().coerceAtLeast(1)
            val resampled = FloatArray(targetLen)
            val ratio = sampleRate.toDouble() / 16000.0
            for (i in 0 until targetLen) {
                val srcPos = i * ratio
                val i0 = srcPos.toInt().coerceIn(0, totalFrames - 1)
                val i1 = (i0 + 1).coerceIn(0, totalFrames - 1)
                val frac = (srcPos - i0).toFloat()
                resampled[i] = mono[i0] * (1f - frac) + mono[i1] * frac
            }
            resampled
        }
        return DecodedWav(
            pcm16k = pcm16k,
            durationSec = durationSec,
            pcmRawBytes = monoPcm16Bytes,
            rawSampleRate = sampleRate,
            rawTotalFrames = totalFrames,
        )
    }

    fun close() {
        runCatching { genaiClient?.close() }
        genaiClient = null
        runCatching { tts?.stop() }
        runCatching { tts?.shutdown() }
        tts = null
    }
}

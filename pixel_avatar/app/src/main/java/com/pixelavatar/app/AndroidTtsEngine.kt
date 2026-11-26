package com.pixelavatar.app

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import kotlinx.coroutines.CompletableDeferred
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
)

class AndroidTtsEngine(private val context: Context) {

    private var tts: TextToSpeech? = null
    private var puckVoice: Voice? = null
    private var gretchenVoice: Voice? = null
    private val readyDeferred = CompletableDeferred<Boolean>()
    private val pending = mutableMapOf<String, CompletableDeferred<Boolean>>()

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val engine = tts
                engine?.language = Locale.US
                selectDistinctVoices(engine)
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
        }
    }

    private fun selectDistinctVoices(engine: TextToSpeech?) {
        val voices = runCatching { engine?.voices }.getOrNull() ?: return
        val enUsLocal = voices.filter {
            it.locale?.language == "en" && it.locale?.country == "US" && !it.isNetworkConnectionRequired
        }.sortedBy { it.name }
        // Google TTS voice IDs on Pixel typically include -tpd/-tpf/-iol/-iob/-sfg/-iom etc.
        puckVoice = enUsLocal.firstOrNull { it.name.contains("tpd") || it.name.contains("iom") || it.name.contains("iol") }
            ?: enUsLocal.firstOrNull()
        gretchenVoice = enUsLocal.firstOrNull { it != puckVoice && (it.name.contains("tpf") || it.name.contains("iob") || it.name.contains("sfg")) }
            ?: enUsLocal.firstOrNull { it != puckVoice }
            ?: puckVoice
    }

    suspend fun synthesize(text: String, avatarKey: String): TtsAudio {
        val ok = readyDeferred.await()
        val engine = requireNotNull(tts) { "Android TTS not initialized" }
        require(ok) { "Android TTS initialization failed" }

        val isGretchen = avatarKey == "gretchen"
        val voice = if (isGretchen) gretchenVoice else puckVoice
        if (voice != null) runCatching { engine.voice = voice }
        engine.setPitch(if (isGretchen) 1.06f else 0.94f)
        engine.setSpeechRate(1.02f)

        val outFile = File(context.cacheDir, "tts_${System.currentTimeMillis()}.wav")
        val uttId = UUID.randomUUID().toString()
        val done = CompletableDeferred<Boolean>()
        synchronized(pending) { pending[uttId] = done }

        val t0 = System.nanoTime()
        val res = engine.synthesizeToFile(text, Bundle(), outFile, uttId)
        require(res == TextToSpeech.SUCCESS) { "TTS synthesizeToFile returned $res" }
        val finished = done.await()
        val synthMs = (System.nanoTime() - t0) / 1e6
        require(finished && outFile.exists() && outFile.length() > 44) { "TTS synthesis failed" }

        val (pcm16k, durationSec) = decodeWavTo16kMono(outFile)
        return TtsAudio(outFile, pcm16k, durationSec, synthMs)
    }

    private fun decodeWavTo16kMono(file: File): Pair<FloatArray, Double> {
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
        for (i in 0 until totalFrames) {
            var acc = 0f
            for (c in 0 until channels) {
                val off = dataOffset + (i * channels + c) * bytesPerSample
                val s = bb.getShort(off).toInt()
                acc += s / 32768f
            }
            mono[i] = acc / channels
        }
        val durationSec = totalFrames.toDouble() / sampleRate.toDouble()
        if (sampleRate == 16000) return mono to durationSec

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
        return resampled to durationSec
    }

    fun close() {
        runCatching { tts?.stop() }
        runCatching { tts?.shutdown() }
        tts = null
    }
}

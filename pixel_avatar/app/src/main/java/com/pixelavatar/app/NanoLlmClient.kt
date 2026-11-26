package com.pixelavatar.app

import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import com.google.mlkit.genai.prompt.TextPart

data class ChatTurn(
    val speaker: String, // "You", "Puck", "Gretchen", or "System"
    val text: String,
    val avatarKey: String? = null,
)

class NanoLlmClient {

    private val model: GenerativeModel = Generation.getClient()
    var statusLabel: String = "Checking AICore…"
        private set
    var isReady: Boolean = false
        private set

    suspend fun prepare(onStatus: (String) -> Unit): Boolean {
        return try {
            when (val st = model.checkStatus()) {
                FeatureStatus.AVAILABLE -> {
                    isReady = true
                    statusLabel = "Warming Nano…"
                    val t0 = System.nanoTime()
                    runCatching { model.warmup() }
                    val wMs = (System.nanoTime() - t0) / 1e6
                    statusLabel = "Gemini Nano Ready"
                    onStatus("AICore Gemini Nano: AVAILABLE & warmed on Tensor G5 (${"%.0f".format(wMs)} ms)")
                    true
                }
                FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> {
                    statusLabel = "Downloading Nano…"
                    onStatus("AICore Gemini Nano downloadable — starting on-device download…")
                    model.download().collect { dl ->
                        when (dl) {
                            is DownloadStatus.DownloadStarted -> onStatus("Gemini Nano download started…")
                            is DownloadStatus.DownloadProgress -> {
                                val mb = dl.totalBytesDownloaded / (1024.0 * 1024.0)
                                statusLabel = "Nano ${"%.0f".format(mb)} MB"
                            }
                            is DownloadStatus.DownloadCompleted -> {
                                isReady = true
                                runCatching { model.warmup() }
                                statusLabel = "Gemini Nano Ready"
                                onStatus("AICore Gemini Nano download complete & warmed!")
                            }
                            is DownloadStatus.DownloadFailed -> {
                                statusLabel = "Nano Download Failed"
                                onStatus("AICore Gemini Nano download failed: ${dl.e.message}")
                            }
                        }
                    }
                    isReady
                }
                else -> {
                    isReady = false
                    statusLabel = "Nano Unavailable ($st)"
                    onStatus("AICore Gemini Nano status=$st")
                    false
                }
            }
        } catch (e: Throwable) {
            isReady = false
            statusLabel = "Nano Error"
            onStatus("AICore Gemini Nano error: ${e.message}")
            false
        }
    }

    private fun cleanReply(raw: String, personaName: String): String {
        var reply = raw.replace(Regex("[*_`#]"), "").trim()
        val ownTag = Regex("(?:^|\\n)\\s*(?:\\[?$personaName\\]?)\\s*:\\s*([\\s\\S]+)$", RegexOption.IGNORE_CASE)
        ownTag.find(reply)?.groupValues?.getOrNull(1)?.trim()?.takeIf { it.length > 8 }?.let { reply = it }
        reply = reply
            .replace(Regex("^\\s*\\[[^\\]]{1,24}]\\s*:?\\s*"), "")
            .replace(Regex("^\\s*(?:Puck|Gretchen)\\s*:\\s*", RegexOption.IGNORE_CASE), "")
            .trim()
        return reply
    }

    suspend fun replyAs(
        avatarKey: String,
        history: List<ChatTurn>,
        promptText: String,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"
        val personaDesc = if (avatarKey == "gretchen") {
            "You are Gretchen, a friendly, sharp AI companion with Puck."
        } else {
            "You are Puck, a warm, witty AI companion with Gretchen."
        }

        // Ultra-compact prompt (<=4 recent turns) minimizes prefill latency on Tensor G5!
        val recent = history.filter { it.speaker != "System" }.takeLast(4)
        val sb = StringBuilder()
        sb.append(personaDesc)
        sb.append(" Reply in 1-2 short spoken sentences (under 25 words), no speaker labels or markdown.\n")
        if (recent.isNotEmpty()) {
            for (m in recent) {
                sb.append(m.speaker).append(": ").append(m.text).append('\n')
            }
        }
        sb.append("Task: ").append(promptText).append('\n')
        sb.append(personaName).append(':')

        val request = GenerateContentRequest.Builder(TextPart(sb.toString())).apply {
            temperature = 0.7f
            topK = 32
            candidateCount = 1
            maxOutputTokens = 56
            enableThinking = false
        }.build()

        val t0 = System.nanoTime()
        val acc = StringBuilder()
        if (onPartial != null) {
            try {
                model.generateContentStream(request).collect { chunk ->
                    val piece = chunk.candidates.firstOrNull()?.text ?: ""
                    if (piece.isNotEmpty()) {
                        acc.append(piece)
                        val cleaned = cleanReply(acc.toString(), personaName)
                        if (cleaned.isNotEmpty()) onPartial(cleaned)
                    }
                }
            } catch (_: Throwable) {
                // Fallback to non-streaming if stream fails
                val resp = model.generateContent(request)
                acc.clear()
                acc.append(resp.candidates.firstOrNull()?.text ?: "")
            }
        } else {
            val resp = model.generateContent(request)
            acc.append(resp.candidates.firstOrNull()?.text ?: "")
        }
        val ms = (System.nanoTime() - t0) / 1e6

        var reply = cleanReply(acc.toString(), personaName)
        if (reply.isEmpty()) reply = "Hey there! Great chatting with you offline on Pixel 10."
        return reply to ms
    }

    fun close() {
        runCatching { model.close() }
    }
}

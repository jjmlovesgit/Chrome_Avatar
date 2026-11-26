package com.pixelavatar.app

import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel

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
                    statusLabel = "Gemini Nano Ready"
                    onStatus("AICore Gemini Nano: AVAILABLE (on-device Tensor G5)")
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
                                statusLabel = "Gemini Nano Ready"
                                onStatus("AICore Gemini Nano download complete!")
                            }
                            is DownloadStatus.DownloadFailed -> {
                                statusLabel = "Nano Download Failed"
                                onStatus("Gemini Nano download failed: ${dl.e.message}")
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

    suspend fun replyAs(
        avatarKey: String,
        history: List<ChatTurn>,
        promptText: String,
    ): Pair<String, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"
        val personaDesc = if (avatarKey == "gretchen") {
            "You are Gretchen, an articulate, friendly, thoughtful AI companion sharing a room with Puck."
        } else {
            "You are Puck, a warm, witty, sharp AI companion sharing a room with Gretchen."
        }

        val recent = history.filter { it.speaker != "System" }.takeLast(8)
        val sb = StringBuilder()
        sb.append(personaDesc).append('\n')
        sb.append("Reply naturally in 1 to 2 concise spoken sentences as $personaName only. ")
        sb.append("Never repeat what someone else just said, never write any speaker name prefix or colon, and never use markdown or emojis.\n")
        if (recent.isNotEmpty()) {
            sb.append("\nContext from earlier in the room (for reference only — do NOT repeat or quote it):\n")
            for (m in recent) {
                sb.append("(${m.speaker} said) ${m.text}\n")
            }
        }
        sb.append("\nNow respond in 1 to 2 spoken sentences as $personaName:\n")
        sb.append(promptText)

        val t0 = System.nanoTime()
        val resp = model.generateContent(sb.toString())
        val ms = (System.nanoTime() - t0) / 1e6

        var reply = (resp.candidates.firstOrNull()?.text ?: "").replace(Regex("[*_`#]"), "").trim()
        val ownTag = Regex("(?:^|\\n)\\s*(?:\\[?$personaName\\]?)\\s*:\\s*([\\s\\S]+)$", RegexOption.IGNORE_CASE)
        ownTag.find(reply)?.groupValues?.getOrNull(1)?.trim()?.takeIf { it.length > 10 }?.let { reply = it }
        reply = reply
            .replace(Regex("^\\s*\\[[^\\]]{1,24}]\\s*:?\\s*"), "")
            .replace(Regex("^\\s*(?:Puck|Gretchen)\\s*:\\s*", RegexOption.IGNORE_CASE), "")
            .trim()
        if (reply.isEmpty()) reply = "Hey there! It is great chatting with you right here on your Pixel 10."
        return reply to ms
    }

    fun close() {
        runCatching { model.close() }
    }
}

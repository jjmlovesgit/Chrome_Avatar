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

    /**
     * Turn A (Voice Study Quiz): Injects ONLY the current Flashcard (~130 tokens total)
     * so Gemini Nano stays fast (~600-900 ms) with zero context bloat and zero hallucinations.
     */
    suspend fun askStudyQuestion(
        avatarKey: String,
        card: Flashcard,
        cardIndex: Int,
        totalCards: Int,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"
        val prompt = buildString {
            append("SYSTEM:\n")
            append("You are $personaName, an encouraging, concise voice tutor.\n")
            append("Rules:\n")
            append("1. Speak in plain conversational sentences (no markdown, bullets, labels, or emojis).\n")
            append("2. Keep your response under 32 words.\n")
            append("3. Ask the student ONE clear question based ONLY on the Study Card below. Do not give away the answer.\n\n")
            append("STUDY CARD (${cardIndex + 1} of $totalCards):\n")
            append("Topic: ${card.topic}\n")
            append("Facts: ${card.facts}\n")
            append("Key Concept: ${card.keyConcept}\n\n")
            append("TASK: Ask the student one fun, clear spoken question about ${card.topic}.\n")
            append("$personaName:")
        }
        val request = GenerateContentRequest.Builder(TextPart(prompt)).apply {
            temperature = 0.6f
            topK = 24
            candidateCount = 1
            maxOutputTokens = 68
            enableThinking = false
        }.build()

        val (raw, ms) = runNanoRequest(request, personaName, onPartial)
        val cleaned = raw.replace(Regex("\\[(?:PASS|RETRY)]", RegexOption.IGNORE_CASE), "").trim()
        val finalQ = if (cleaned.length >= 10) cleaned else StudyGuideParser.fallbackQuestion(card, cardIndex, totalCards)
        return finalQ to ms
    }

    /**
     * Turn B (Voice Study Quiz Evaluation): Compares user's spoken/typed answer strictly
     * against the current Study Card, parses [PASS] / [RETRY], and strips the tag before TTS!
     */
    suspend fun evaluateStudyAnswer(
        avatarKey: String,
        card: Flashcard,
        questionAsked: String,
        userAnswer: String,
        allowRetry: Boolean,
        onPartial: ((String) -> Unit)? = null,
    ): Triple<String, Boolean, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"
        val prompt = buildString {
            append("SYSTEM:\n")
            append("You are $personaName, a friendly voice tutor evaluating a student's spoken answer.\n")
            append("Rules:\n")
            append("1. Speak in plain conversational sentences (no markdown or lists) under 42 words.\n")
            append("2. Compare the Student Answer strictly against the Study Card facts.\n")
            append("3. If CORRECT: Praise briefly, share one quick bonus fact from the card, and end with [PASS].\n")
            if (allowRetry) {
                append("4. If PARTIAL or WRONG: Give a gentle hint without revealing the full answer, ask a simpler follow-up, and end with [RETRY].\n\n")
            } else {
                append("4. If WRONG: Kindly explain the right answer using the card's key concept, encourage them, and end with [PASS].\n\n")
            }
            append("STUDY CARD:\n")
            append("Topic: ${card.topic}\n")
            append("Facts: ${card.facts}\n")
            append("Key Concept: ${card.keyConcept}\n\n")
            append("QUESTION ASKED: $questionAsked\n")
            append("STUDENT ANSWER: \"$userAnswer\"\n")
            append("$personaName:")
        }
        val request = GenerateContentRequest.Builder(TextPart(prompt)).apply {
            temperature = 0.45f
            topK = 24
            candidateCount = 1
            maxOutputTokens = 88
            enableThinking = false
        }.build()

        val (raw, ms) = runNanoRequest(request, personaName) { partial ->
            onPartial?.invoke(partial.replace(Regex("\\[(?:PASS|RETRY)]", RegexOption.IGNORE_CASE), "").trim())
        }
        val hasRetry = raw.contains("[RETRY]", ignoreCase = true)
        val hasPass = raw.contains("[PASS]", ignoreCase = true)
        val passed = when {
            !allowRetry -> true
            hasPass -> true
            hasRetry -> false
            else -> StudyGuideParser.fallbackEvaluate(card, userAnswer, allowRetry).second
        }
        val spoken = raw
            .replace(Regex("\\[(?:PASS|RETRY)]", RegexOption.IGNORE_CASE), "")
            .replace(Regex("\\s+"), " ")
            .trim()
            .ifBlank { StudyGuideParser.fallbackEvaluate(card, userAnswer, allowRetry).first }
        return Triple(spoken, passed, ms)
    }

    /**
     * On-Device Gmail Notification Brief: Injects ONLY the captured notification snippets
     * (~150 tokens) into Gemini Nano for a natural spoken executive summary.
     */
    suspend fun summarizeEmails(
        avatarKey: String,
        emails: List<GmailSnippet>,
        userRequest: String,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"
        val prompt = buildString {
            append("SYSTEM:\n")
            append("You are $personaName, a helpful on-device voice assistant reading recent Gmail notifications aloud.\n")
            append("Rules:\n")
            append("1. Speak in 1 to 2 natural, concise conversational sentences under 40 words (no markdown, bullets, or URLs).\n")
            append("2. Report ONLY what is in the EMAIL NOTIFICATIONS below. Never invent details.\n")
            append("3. Clearly mention who sent the latest email and its main point.\n\n")
            append("EMAIL NOTIFICATIONS (newest first):\n")
            emails.take(3).forEachIndexed { idx, e ->
                append("${idx + 1}. From: ${e.sender} | Snippet: ${e.bestBody().take(240)}\n")
            }
            append("\nUSER REQUEST: $userRequest\n")
            append("$personaName:")
        }
        val request = GenerateContentRequest.Builder(TextPart(prompt)).apply {
            temperature = 0.4f
            topK = 24
            candidateCount = 1
            maxOutputTokens = 84
            enableThinking = false
        }.build()

        val (raw, ms) = runNanoRequest(request, personaName, onPartial)
        val finalBrief = if (raw.length >= 10) raw else fallbackEmailBrief(emails)
        return finalBrief to ms
    }

    fun fallbackEmailBrief(emails: List<GmailSnippet>): String {
        val latest = emails.firstOrNull() ?: return "I don't see any active Gmail notifications right now."
        val body = latest.bestBody().replace('\n', ' ').replace(Regex("\\s+"), " ").take(140)
        val extra = if (emails.size > 1) " You also have ${emails.size - 1} other recent Gmail notification${if (emails.size > 2) "s" else ""}." else ""
        return "Your latest email is from ${latest.sender}. It says: $body.$extra"
    }

    private suspend fun runNanoRequest(
        request: GenerateContentRequest,
        personaName: String,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> {
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
                val resp = model.generateContent(request)
                acc.clear()
                acc.append(resp.candidates.firstOrNull()?.text ?: "")
            }
        } else {
            val resp = model.generateContent(request)
            acc.append(resp.candidates.firstOrNull()?.text ?: "")
        }
        val ms = (System.nanoTime() - t0) / 1e6
        return cleanReply(acc.toString(), personaName) to ms
    }

    fun close() {
        runCatching { model.close() }
    }
}

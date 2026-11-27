package com.pixelavatar.app

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import com.google.mlkit.genai.prompt.TextPart
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

data class ChatTurn(
    val speaker: String, // "You", "Puck", "Gretchen", or "System"
    val text: String,
    val avatarKey: String? = null,
)

/**
 * Unified 100% On-Device LLM Client with automatic 2-tier local execution:
 *   Tier 1: Pixel AICore Gemini Nano (ML Kit GenAI Prompt API — Pixel 10 Tensor G5)
 *   Tier 2: Google AI Edge LiteRT-LM (Qwen3-0.6B.litertlm — AAOS, Pixel 7a, any Android device)
 */
class NanoLlmClient(private val context: Context) {

    companion object {
        const val FALLBACK_MODEL_NAME = "Qwen3-0.6B.litertlm"
        const val FALLBACK_MODEL_URL =
            "https://huggingface.co/litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm"
        const val FALLBACK_MIN_BYTES = 500_000_000L // ~585 MB
    }

    private val model: GenerativeModel = Generation.getClient()
    private var litertEngine: Engine? = null
    private val lmMutex = Mutex()

    var statusLabel: String = "Checking AICore…"
        private set
    var activeEngineShort: String = "Offline Fallback"
        private set
    var isReady: Boolean = false
        private set
    var usingLiteRtLm: Boolean = false
        private set
    var usingCpuLlm: Boolean = false
        private set

    suspend fun prepare(
        preferGpu: Boolean = true,
        onProgress: ((String) -> Unit)? = null,
        onStatus: (String) -> Unit,
    ): Boolean {
        // Tier 1: Try Pixel AICore Gemini Nano first
        val aicoreOk = try {
            when (val st = model.checkStatus()) {
                FeatureStatus.AVAILABLE -> {
                    isReady = true
                    usingLiteRtLm = false
                    activeEngineShort = "Gemini Nano"
                    statusLabel = "Warming Nano…"
                    val t0 = System.nanoTime()
                    runCatching { model.warmup() }
                    val wMs = (System.nanoTime() - t0) / 1e6
                    statusLabel = "Gemini Nano Ready"
                    onStatus("AICore Gemini Nano: AVAILABLE & warmed (${"%.0f".format(wMs)} ms)")
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
                                onProgress?.invoke("Downloading Nano ${"%.0f".format(mb)} MB…")
                            }
                            is DownloadStatus.DownloadCompleted -> {
                                isReady = true
                                usingLiteRtLm = false
                                activeEngineShort = "Gemini Nano"
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
                    onStatus("AICore Gemini Nano unavailable (status=$st) — activating LiteRT-LM on-device fallback…")
                    false
                }
            }
        } catch (e: Throwable) {
            onStatus("AICore Gemini Nano check failed (${e.message}) — activating LiteRT-LM on-device fallback…")
            false
        }

        if (aicoreOk) return true

        // Tier 2: Google AI Edge LiteRT-LM (Qwen3-0.6B, 585 MB, 100% on-device)
        return prepareLiteRtLmFallback(preferGpu = preferGpu, onProgress = onProgress, onStatus = onStatus)
    }

    private fun prepareLiteRtLmFallback(
        preferGpu: Boolean,
        onProgress: ((String) -> Unit)?,
        onStatus: (String) -> Unit,
    ): Boolean {
        val modelDir = File(context.filesDir, "models").apply { mkdirs() }
        val modelFile = File(modelDir, FALLBACK_MODEL_NAME)

        if (!modelFile.exists() || modelFile.length() < FALLBACK_MIN_BYTES) {
            statusLabel = "Downloading Qwen3-0.6B…"
            onProgress?.invoke("Downloading Qwen3-0.6B (585 MB)…")
            onStatus("Downloading on-device LiteRT-LM fallback: $FALLBACK_MODEL_NAME (585 MB, 1-time)…")
            val dlOk = downloadFileWithProgress(
                urlStr = FALLBACK_MODEL_URL,
                dst = modelFile,
                onProgress = { mb, totalMb ->
                    val pct = if (totalMb > 0) (mb * 100 / totalMb) else 0
                    statusLabel = "Qwen3 $pct% (${mb}MB)"
                    onProgress?.invoke("Downloading Qwen3-0.6B $pct% (${mb}/${totalMb} MB)…")
                },
                onStatus = onStatus,
            )
            if (!dlOk) {
                isReady = false
                usingLiteRtLm = false
                activeEngineShort = "Offline Fallback"
                statusLabel = "LiteRT-LM Offline"
                return false
            }
        } else {
            onStatus("Cached LiteRT-LM model found: ${modelFile.name} (${modelFile.length() / 1048576} MB)")
        }

        // Release any previous engine instance cleanly
        runCatching { litertEngine?.close() }
        litertEngine = null

        val cacheDir = File(context.cacheDir, "litertlm_cache").apply { mkdirs() }
        // Leave at least 1 core free for AudioFlinger & UI so audio never underruns
        val cpuThreads = (Runtime.getRuntime().availableProcessors() - 1).coerceIn(2, 4)
        val candidates: List<Pair<String, Backend>> = if (preferGpu) {
            listOf("GPU" to Backend.GPU(), "CPU" to Backend.CPU(threadCount = cpuThreads))
        } else {
            listOf("CPU" to Backend.CPU(threadCount = cpuThreads))
        }

        for ((label, backend) in candidates) {
            try {
                statusLabel = "Loading Qwen3 ($label)…"
                onProgress?.invoke("Initializing LiteRT-LM ($label)…")
                val t0 = System.nanoTime()
                val cfg = EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = backend,
                    maxNumTokens = 1024,
                    cacheDir = cacheDir.absolutePath,
                )
                val eng = Engine(cfg)
                eng.initialize()
                val ms = (System.nanoTime() - t0) / 1e6
                litertEngine = eng
                isReady = true
                usingLiteRtLm = true
                usingCpuLlm = (label == "CPU")
                activeEngineShort = "LiteRT-LM Qwen3 ($label)"
                statusLabel = "Qwen3-0.6B ($label)"
                onStatus("LiteRT-LM Qwen3-0.6B initialized on $label (${"%.0f".format(ms)} ms) — 100% offline LLM ready!")
                return true
            } catch (e: Throwable) {
                onStatus("LiteRT-LM [$label] init fallback: ${e.message?.take(140)}")
            }
        }

        isReady = false
        usingLiteRtLm = false
        activeEngineShort = "Offline Fallback"
        statusLabel = "LiteRT-LM Failed"
        return false
    }

    private fun downloadFileWithProgress(
        urlStr: String,
        dst: File,
        onProgress: (Long, Long) -> Unit,
        onStatus: (String) -> Unit,
    ): Boolean {
        val tmp = File(dst.parentFile, "${dst.name}.part")
        return try {
            var currentUrl = urlStr
            var conn: HttpURLConnection? = null
            for (hop in 0..5) {
                val c = (URL(currentUrl).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 20_000
                    readTimeout = 60_000
                    instanceFollowRedirects = false
                    setRequestProperty("User-Agent", "PixelAvatar/1.0")
                }
                val code = c.responseCode
                if (code in 301..308) {
                    val loc = c.getHeaderField("Location") ?: break
                    c.disconnect()
                    currentUrl = URL(URL(currentUrl), loc).toString()
                } else {
                    conn = c
                    break
                }
            }
            val activeConn = conn ?: throw IllegalStateException("Redirect loop downloading $urlStr")
            if (activeConn.responseCode !in 200..299) {
                throw IllegalStateException("HTTP ${activeConn.responseCode} downloading $urlStr")
            }
            val totalBytes = activeConn.contentLengthLong.coerceAtLeast(1L)
            val totalMb = totalBytes / 1_048_576L
            var copied = 0L
            var lastReportedMb = -10L
            activeConn.inputStream.use { inp ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(131_072)
                    while (true) {
                        val n = inp.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        copied += n
                        val mb = copied / 1_048_576L
                        if (mb - lastReportedMb >= 10L) {
                            lastReportedMb = mb
                            onProgress(mb, totalMb)
                        }
                    }
                }
            }
            activeConn.disconnect()
            if (dst.exists()) dst.delete()
            tmp.renameTo(dst)
            onStatus("Downloaded ${dst.name} (${dst.length() / 1_048_576L} MB)")
            true
        } catch (e: Throwable) {
            runCatching { tmp.delete() }
            onStatus("LiteRT-LM model download failed: ${e.message}")
            false
        }
    }

    private fun stripNonTextChars(s: String): String {
        val sb = StringBuilder(s.length)
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            val charCount = Character.charCount(cp)
            val keep = when {
                // Standard ASCII letters, digits, space, and clean spoken punctuation (no *, _, `, #, ~, emojis)
                cp in 0x20..0x7E -> {
                    val ch = cp.toChar()
                    ch != '*' && ch != '_' && ch != '`' && ch != '#' && ch != '~' && ch != '^'
                }
                // Smart quotes / dashes / ellipsis / degree / basic accented Latin letters
                cp == 0x2018 || cp == 0x2019 -> true // ‘ ’
                cp == 0x201C || cp == 0x201D -> true // “ ”
                cp == 0x2013 || cp == 0x2014 -> true // – —
                cp == 0x2026 || cp == 0x00B0 -> true // … °
                cp in 0x00C0..0x024F -> Character.isLetter(cp) // accented Latin letters
                else -> false // strips ALL emojis, pictographs, dingbats, variation selectors, ZWJ, etc.
            }
            if (keep) sb.appendCodePoint(cp)
            i += charCount
        }
        return sb.toString()
    }

    private fun cleanReply(raw: String, personaName: String): String {
        // Strip any Qwen3 <think>...</think> blocks and ALL emojis/non-text symbols cleanly before UI & TTS
        var reply = raw
            .replace(Regex("<think>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE), "")
            .replace(Regex("<think>[\\s\\S]*$", RegexOption.IGNORE_CASE), "")
        reply = stripNonTextChars(reply).trim()
        val ownTag = Regex("(?:^|\\n)\\s*(?:\\[?$personaName\\]?)\\s*:\\s*([\\s\\S]+)$", RegexOption.IGNORE_CASE)
        ownTag.find(reply)?.groupValues?.getOrNull(1)?.trim()?.takeIf { it.length > 8 }?.let { reply = it }
        reply = reply
            .replace(Regex("^\\s*\\[[^\\]]{1,24}]\\s*:?\\s*"), "")
            .replace(Regex("^\\s*(?:Puck|Gretchen)\\s*:\\s*", RegexOption.IGNORE_CASE), "")
            .replace(Regex("\\s+"), " ")
            .trim()
        // Strip outer wrapping quotes if model wrapped the entire spoken reply in quotes
        if (reply.length >= 2 &&
            ((reply.startsWith("\"") && reply.endsWith("\"")) ||
                (reply.startsWith("\u201C") && reply.endsWith("\u201D")))
        ) {
            reply = reply.substring(1, reply.length - 1).trim()
        }
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

        val recent = history.filter { it.speaker != "System" }.takeLast(4)

        if (usingLiteRtLm && litertEngine != null) {
            val sysInst = "$personaDesc Reply in 1 to 2 short spoken sentences (under 28 words). Use plain ASCII text characters only—never use emojis, icons, quotation marks, bullet points, or markdown."
            val userMsg = buildString {
                if (recent.isNotEmpty()) {
                    append("Recent conversation:\n")
                    for (m in recent) {
                        append(m.speaker).append(": ").append(m.text).append('\n')
                    }
                    append('\n')
                }
                append(promptText)
            }
            val (reply, ms) = runLiteRtLmRequest(
                systemInstruction = sysInst,
                userPrompt = userMsg,
                personaName = personaName,
                temperature = 0.7,
                maxOutputTokens = 72,
                onPartial = onPartial,
            )
            return (reply.ifBlank { "Hey there! Great chatting with you 100% offline." }) to ms
        }

        // Tier 1: AICore Gemini Nano
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

        val (raw, ms) = runNanoRequest(request, personaName, onPartial)
        val reply = raw.ifBlank { "Hey there! Great chatting with you offline." }
        return reply to ms
    }

    /**
     * Turn A (Voice Study Quiz): Injects ONLY the current Flashcard (~130 tokens total)
     * so on-device Nano / LiteRT-LM stays fast with zero context bloat and zero hallucinations.
     */
    suspend fun askStudyQuestion(
        avatarKey: String,
        card: Flashcard,
        cardIndex: Int,
        totalCards: Int,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"

        val (raw, ms) = if (usingLiteRtLm && litertEngine != null) {
            val sysInst = buildString {
                append("You are $personaName, an encouraging, concise voice tutor. ")
                append("Speak in plain conversational sentences (no markdown, bullets, or emojis) under 32 words. ")
                append("Ask ONE clear question based ONLY on the Study Card. Never give away the answer.")
            }
            val userMsg = buildString {
                append("STUDY CARD (${cardIndex + 1} of $totalCards):\n")
                append("Topic: ${card.topic}\n")
                append("Facts: ${card.facts}\n")
                append("Key Concept: ${card.keyConcept}\n\n")
                append("Ask the student one fun, clear spoken question about ${card.topic}.")
            }
            runLiteRtLmRequest(sysInst, userMsg, personaName, temperature = 0.6, maxOutputTokens = 76, onPartial = onPartial)
        } else {
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
            runNanoRequest(request, personaName, onPartial)
        }

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

        val (raw, ms) = if (usingLiteRtLm && litertEngine != null) {
            val sysInst = buildString {
                append("You are $personaName, a friendly voice tutor evaluating a student's spoken answer. ")
                append("Speak in plain conversational sentences (no markdown or lists) under 42 words. ")
                append("Compare strictly against the Study Card. ")
                append("If CORRECT: praise briefly, share one bonus fact from the card, and end with [PASS]. ")
                if (allowRetry) {
                    append("If PARTIAL or WRONG: give a gentle hint without revealing the full answer, ask a simpler follow-up, and end with [RETRY].")
                } else {
                    append("If WRONG: kindly explain the right answer using the card's key concept and end with [PASS].")
                }
            }
            val userMsg = buildString {
                append("STUDY CARD:\nTopic: ${card.topic}\nFacts: ${card.facts}\nKey Concept: ${card.keyConcept}\n\n")
                append("QUESTION ASKED: $questionAsked\nSTUDENT ANSWER: \"$userAnswer\"")
            }
            runLiteRtLmRequest(sysInst, userMsg, personaName, temperature = 0.45, maxOutputTokens = 96) { partial ->
                onPartial?.invoke(partial.replace(Regex("\\[(?:PASS|RETRY)]", RegexOption.IGNORE_CASE), "").trim())
            }
        } else {
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
            runNanoRequest(request, personaName) { partial ->
                onPartial?.invoke(partial.replace(Regex("\\[(?:PASS|RETRY)]", RegexOption.IGNORE_CASE), "").trim())
            }
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
     * (~150 tokens) into Nano / LiteRT-LM for a natural spoken executive summary.
     */
    suspend fun summarizeEmails(
        avatarKey: String,
        emails: List<GmailSnippet>,
        userRequest: String,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> {
        val personaName = if (avatarKey == "gretchen") "Gretchen" else "Puck"

        val (raw, ms) = if (usingLiteRtLm && litertEngine != null) {
            val sysInst = "You are $personaName, a helpful on-device voice assistant reading recent Gmail notifications aloud. " +
                "Speak in 1 to 2 concise spoken sentences under 40 words (no markdown, bullets, or URLs). " +
                "Report ONLY what is in the notifications and mention who sent the latest email."
            val userMsg = buildString {
                append("EMAIL NOTIFICATIONS (newest first):\n")
                emails.take(3).forEachIndexed { idx, e ->
                    append("${idx + 1}. From: ${e.sender} | Snippet: ${e.bestBody().take(240)}\n")
                }
                append("\nUSER REQUEST: $userRequest")
            }
            runLiteRtLmRequest(sysInst, userMsg, personaName, temperature = 0.4, maxOutputTokens = 88, onPartial = onPartial)
        } else {
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
            runNanoRequest(request, personaName, onPartial)
        }

        val finalBrief = if (raw.length >= 10) raw else fallbackEmailBrief(emails)
        return finalBrief to ms
    }

    fun fallbackEmailBrief(emails: List<GmailSnippet>): String {
        val latest = emails.firstOrNull() ?: return "I don't see any active Gmail notifications right now."
        val body = latest.bestBody().replace('\n', ' ').replace(Regex("\\s+"), " ").take(140)
        val extra = if (emails.size > 1) " You also have ${emails.size - 1} other recent Gmail notification${if (emails.size > 2) "s" else ""}." else ""
        return "Your latest email is from ${latest.sender}. It says: $body.$extra"
    }

    private suspend fun runLiteRtLmRequest(
        systemInstruction: String,
        userPrompt: String,
        personaName: String,
        temperature: Double,
        maxOutputTokens: Int,
        onPartial: ((String) -> Unit)? = null,
    ): Pair<String, Double> = lmMutex.withLock {
        val eng = litertEngine ?: return@withLock "" to 0.0
        val t0 = System.nanoTime()
        val acc = StringBuilder()
        val convCfg = ConversationConfig(
            systemInstruction = Contents.of(systemInstruction),
            samplerConfig = SamplerConfig(topK = 28, topP = 0.9, temperature = temperature),
            maxOutputToken = maxOutputTokens,
            thinkingConfig = ThinkingConfig(enableThinking = false),
        )
        eng.createConversation(convCfg).use { conv ->
            if (onPartial != null) {
                try {
                    conv.sendMessageAsync(userPrompt).collect { chunk ->
                        val piece = chunk.contents.contents
                            .filterIsInstance<Content.Text>()
                            .joinToString("") { it.text }
                        if (piece.isNotEmpty()) {
                            acc.append(piece)
                            val cleaned = cleanReply(acc.toString(), personaName)
                            if (cleaned.isNotEmpty()) onPartial(cleaned)
                        }
                    }
                } catch (_: Throwable) {
                    val msg = conv.sendMessage(userPrompt)
                    acc.clear()
                    acc.append(msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text })
                }
            } else {
                val msg = conv.sendMessage(userPrompt)
                acc.append(msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text })
            }
        }
        val ms = (System.nanoTime() - t0) / 1e6
        cleanReply(acc.toString(), personaName) to ms
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
        runCatching { litertEngine?.close() }
        litertEngine = null
        runCatching { model.close() }
    }
}

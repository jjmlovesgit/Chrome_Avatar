package com.pixelavatar.app

import android.content.Context

data class Flashcard(
    val number: Int,
    val topic: String,
    val facts: String,
    val keyConcept: String,
)

data class StudyDeck(
    val title: String,
    val cards: List<Flashcard>,
)

data class QuizUiSnapshot(
    val active: Boolean = false,
    val deckTitle: String = "Solar System Study Guide",
    val cardIndex: Int = 0,
    val totalCards: Int = 8,
    val currentTopic: String = "",
    val masteredCount: Int = 0,
    val awaitingAnswer: Boolean = false,
    val retryCount: Int = 0,
    val completed: Boolean = false,
)

object StudyGuideParser {

    fun loadBundledSolarSystem(context: Context): StudyDeck {
        val raw = runCatching {
            context.assets.open("study_guides/solar_system.txt").bufferedReader().use { it.readText() }
        }.getOrDefault(FALLBACK_SOLAR_TEXT)
        return parse(raw, "Solar System Study Guide")
    }

    fun parse(rawText: String, fallbackTitle: String = "Custom Study Guide"): StudyDeck {
        var title = fallbackTitle
        val cards = mutableListOf<Flashcard>()

        var curNum = 0
        var curTopic = ""
        var curFacts = StringBuilder()
        var curConcept = StringBuilder()
        var activeField = ""

        fun flushCard() {
            if (curTopic.isNotBlank() && curFacts.isNotBlank()) {
                cards.add(
                    Flashcard(
                        number = if (curNum > 0) curNum else cards.size + 1,
                        topic = curTopic.trim(),
                        facts = curFacts.toString().trim(),
                        keyConcept = curConcept.toString().trim().ifBlank { curFacts.toString().trim() },
                    )
                )
            }
            curNum = 0
            curTopic = ""
            curFacts = StringBuilder()
            curConcept = StringBuilder()
            activeField = ""
        }

        val cardHeaderRe = Regex("^\\[CARD\\s*(\\d+)?\\]$", RegexOption.IGNORE_CASE)
        for (rawLine in rawText.lines()) {
            val line = rawLine.trim()
            if (line.isEmpty()) continue
            if (line.startsWith("#") && cards.isEmpty() && curTopic.isEmpty()) {
                val cleaned = line.trimStart('#').trim()
                if (cleaned.isNotEmpty()) title = cleaned
                continue
            }
            val cardMatch = cardHeaderRe.matchEntire(line)
            if (cardMatch != null) {
                flushCard()
                curNum = cardMatch.groupValues.getOrNull(1)?.toIntOrNull() ?: (cards.size + 1)
                continue
            }
            when {
                line.startsWith("TOPIC:", ignoreCase = true) -> {
                    curTopic = line.substringAfter(':').trim()
                    activeField = "TOPIC"
                }
                line.startsWith("FACTS:", ignoreCase = true) -> {
                    curFacts.append(line.substringAfter(':').trim())
                    activeField = "FACTS"
                }
                line.startsWith("KEY_CONCEPT:", ignoreCase = true) -> {
                    curConcept.append(line.substringAfter(':').trim())
                    activeField = "KEY_CONCEPT"
                }
                activeField == "FACTS" -> {
                    if (curFacts.isNotEmpty()) curFacts.append(' ')
                    curFacts.append(line)
                }
                activeField == "KEY_CONCEPT" -> {
                    if (curConcept.isNotEmpty()) curConcept.append(' ')
                    curConcept.append(line)
                }
            }
        }
        flushCard()

        // If a user loads a plain bulleted .txt/.md file without [CARD] blocks, split by lines gracefully
        if (cards.isEmpty()) {
            rawText.lines().map { it.trim().trimStart('-', '*', '•').trim() }
                .filter { it.length >= 12 && !it.startsWith("#") }
                .take(12)
                .forEachIndexed { idx, line ->
                    val parts = line.split(":", "—", " - ", limit = 2)
                    val topic = if (parts.size == 2) parts[0].trim() else "Concept ${idx + 1}"
                    val facts = if (parts.size == 2) parts[1].trim() else line
                    cards.add(Flashcard(idx + 1, topic, facts, facts))
                }
        }

        return StudyDeck(title = title, cards = cards)
    }

    /**
     * Deterministic offline fallback tutor for devices without AICore Gemini Nano (e.g., Pixel 7a).
     */
    fun fallbackQuestion(card: Flashcard, cardIdx: Int, total: Int): String {
        val intro = if (cardIdx == 0) "Let's start your quiz! " else "Next up, card ${cardIdx + 1} of $total: "
        return when (card.topic.lowercase()) {
            "mercury" -> "${intro}Mercury is the closest planet to the Sun, yet it gets freezing cold at night. Why do its temperatures swing so wildly?"
            "venus" -> "${intro}Venus isn't the closest planet to the Sun, yet it's the hottest planet in our solar system. Do you know why?"
            "earth" -> "${intro}What makes Earth unique compared to every other recognized planet in our solar system?"
            "mars" -> "${intro}Mars is famous as the Red Planet. What gives its surface that rusty red color, or what giant volcano lives there?"
            "jupiter" -> "${intro}Jupiter is the largest planet in our solar system. What is its most famous giant storm called?"
            "saturn" -> "${intro}Saturn is famous for its bright icy rings. What surprising fact makes its density stand out from every other planet?"
            "uranus" -> "${intro}Uranus is a pale cyan ice giant. What is unusual about the way it spins around its axis?"
            "neptune" -> "${intro}Neptune is the farthest recognized planet from the Sun. What extreme weather is it famous for?"
            else -> "${intro}What is the key concept to remember about ${card.topic}?"
        }
    }

    fun fallbackEvaluate(card: Flashcard, userAnswer: String, allowRetry: Boolean): Pair<String, Boolean> {
        val stopWords = setOf(
            "the", "and", "for", "with", "that", "this", "from", "into", "about", "its", "has",
            "are", "was", "have", "most", "very", "planet", "solar", "system", "made", "mostly",
            "due", "called", "known", "largest", "smallest", "from", "sun", "because",
        )
        val keywords = (card.keyConcept + " " + card.facts)
            .lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length >= 4 && it !in stopWords }
            .toSet()
        val answerTokens = userAnswer.lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length >= 3 }
            .toSet()

        val hits = answerTokens.count { tok -> keywords.any { kw -> kw.startsWith(tok) || tok.startsWith(kw) } }
        val passed = hits >= 1
        return if (passed) {
            "Nice job! You nailed it on ${card.topic}. Remember: ${card.keyConcept}" to true
        } else if (allowRetry) {
            val hintWord = keywords.firstOrNull() ?: card.topic
            "Close! Here's a hint for ${card.topic}: think about $hintWord. Want to give it one more try?" to false
        } else {
            "Good effort! For ${card.topic}, the key idea is: ${card.keyConcept} Let's move on to the next one." to true
        }
    }

    private const val FALLBACK_SOLAR_TEXT = """
# SOLAR SYSTEM STUDY GUIDE
[CARD 1]
TOPIC: Mercury
FACTS: Closest planet to the Sun; smallest planet; extreme temperature swings; almost no atmosphere to retain heat.
KEY_CONCEPT: Proximity to Sun and lack of atmosphere.
[CARD 2]
TOPIC: Venus
FACTS: Second from the Sun; hottest planet due to thick CO2 greenhouse atmosphere; rotates backward (retrograde).
KEY_CONCEPT: Runaway greenhouse effect and retrograde rotation.
"""
}

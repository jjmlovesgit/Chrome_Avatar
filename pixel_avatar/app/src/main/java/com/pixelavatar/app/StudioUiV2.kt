package com.pixelavatar.app

import android.graphics.Bitmap
import androidx.compose.animation.CapturedAnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.input.TextFieldLineLimits
import androidx.compose.foundation.text.input.rememberTextFieldState
import androidx.compose.foundation.text.input.setTextAndPlaceCursorAtEnd
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.blur.BlurRadiusSpec
import androidx.compose.ui.graphics.blur.BlurStop
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

data class StudioUiState(
    val ready: Boolean,
    val busy: Boolean,
    val preferGpu: Boolean,
    val playbackMode: Int,
    val activeAvatar: String,
    val puckVoiceShort: String,
    val gretchenVoiceShort: String,
    val demoTopic: String,
    val defaultDemoTopic: String,
    val xShowcaseTopic: String,
    val statusBadge: String,
    val nanoReady: Boolean,
    val nanoBadge: String,
    val inputText: String,
    val logText: String,
    val memMain: String,
    val memSub: String,
    val metricLlm: String,
    val metricTts: String,
    val metricMotion: String,
    val metricRend: String,
    val renderSide: Int,
    val acceleratorLabel: String,
    val frameBmp: Bitmap?,
    val frameVersion: Int,
    val chatHistory: List<ChatTurn>,
    val chatListState: LazyListState,
    val quiz: QuizUiSnapshot = QuizUiSnapshot(),
    val gmailAccessGranted: Boolean = false,
    val gmailCachedCount: Int = 0,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StudioUiV2(
    state: StudioUiState,
    onSelectAvatar: (String) -> Unit,
    onRunThreeTurnDemo: () -> Unit,
    onStartStudyQuiz: () -> Unit,
    onSkipQuizCard: () -> Unit,
    onEndStudyQuiz: () -> Unit,
    onLoadStudyGuideFile: () -> Unit,
    onResetStudyGuide: () -> Unit,
    onOpenGmailAccessSettings: () -> Unit,
    onReadLastEmail: () -> Unit,
    onStop: () -> Unit,
    onInputChange: (String) -> Unit,
    onSendQuestion: (String) -> Unit,
    onMicClick: () -> Unit,
    onCycleVoice: (String) -> Unit,
    onPlaybackModeChange: (Int) -> Unit,
    onPreferGpuChange: (Boolean) -> Unit,
    onDemoTopicSave: (String, Boolean) -> Unit,
    onRefreshMem: () -> Unit,
    onSwitchToClassic: () -> Unit,
) {
    // Flagship surface palette: deep obsidian canvas + elevated dark slate surfaces, zero 1px borders
    val bgCanvas = Color(0xFF0D1117)
    val surfaceElevated = Color(0xFF161B22)
    val surfaceHigher = Color(0xFF1F2630)
    val surfaceActive = Color(0xFF2A3442)
    val primaryBlue = Color(0xFF2F81F7)
    val dangerRed = Color(0xFFDA3633)
    val emerald = Color(0xFF3FB950)
    val amber = Color(0xFFD29922)
    val textPrimary = Color(0xFFF0F6FC)
    val textSecondary = Color(0xFF8B949E)

    var showStatsSheet by remember { mutableStateOf(false) }
    var showSettingsSheet by remember { mutableStateOf(false) }

    val statusDotColor = when {
        !state.ready -> amber
        state.busy -> primaryBlue
        else -> emerald
    }

    Surface(modifier = Modifier.fillMaxSize(), color = bgCanvas) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .imePadding()
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // 1. Minimal Top Bar: Clean Sans-Serif Brand + Compact Status Pill (opens Stats) + Settings Gear
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Pixel Avatar",
                        color = textPrimary,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.SemiBold,
                        letterSpacing = (-0.4).sp,
                        maxLines = 1,
                    )
                    Text(
                        text = if (state.nanoReady) "Gemini Nano · Offline" else "On-Device Studio",
                        color = textSecondary,
                        fontSize = 11.sp,
                        maxLines = 1,
                    )
                }

                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    // Single compact progressive-disclosure status pill -> opens Stats & Telemetry sheet
                    Row(
                        modifier = Modifier
                            .clip(RoundedCornerShape(999.dp))
                            .background(surfaceElevated)
                            .clickable {
                                onRefreshMem()
                                showStatsSheet = true
                            }
                            .padding(horizontal = 12.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Box(
                            modifier = Modifier
                                .size(7.dp)
                                .clip(CircleShape)
                                .background(statusDotColor),
                        )
                        Text(
                            text = state.statusBadge,
                            color = textPrimary,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.widthIn(max = 155.dp),
                        )
                    }

                    // Clean circular Settings button -> opens Settings sheet (Topic, Voices, Mode, GPU, Classic UI)
                    Box(
                        modifier = Modifier
                            .size(34.dp)
                            .clip(CircleShape)
                            .background(surfaceElevated)
                            .clickable { showSettingsSheet = true },
                        contentAlignment = Alignment.Center,
                    ) {
                        GearIcon(color = textSecondary, modifier = Modifier.size(18.dp))
                    }
                }
            }

            // 2. Generous Hero Avatar Viewport (24dp soft corners, elevated surface, zero 1px border, subtle scrims)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(325.dp)
                    .clip(RoundedCornerShape(24.dp))
                    .background(surfaceElevated),
                contentAlignment = Alignment.Center,
            ) {
                val bmp = state.frameBmp
                if (bmp != null) {
                    key(state.frameVersion) {
                        // Compose 1.13 Progressive Blur: razor-sharp top 75% (face/eyes/mouth),
                        // smoothly transitioning to 14.dp frosted glass behind the bottom speaker label!
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "Live 512x512 HD Avatar",
                            modifier = Modifier
                                .fillMaxSize()
                                .blur {
                                    blurRadiusSpec = BlurRadiusSpec.verticalGradient(
                                        listOf(
                                            BlurStop(0f, 0.dp),
                                            BlurStop(0.76f, 0.dp),
                                            BlurStop(1f, 14.dp),
                                        )
                                    )
                                },
                            contentScale = ContentScale.Crop,
                        )
                    }
                }

                // Compose 1.13 CapturedAnimatedVisibility + M3 1.5 Expressive Wavy indicator while preparing
                val isPreparing = !state.ready || (state.busy && !state.statusBadge.startsWith("Speaking"))
                CapturedAnimatedVisibility(
                    visible = isPreparing,
                    enter = fadeIn(),
                    exit = fadeOut(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.TopCenter)
                        .padding(horizontal = 16.dp, vertical = 10.dp),
                ) {
                    LinearWavyProgressIndicator(
                        color = primaryBlue,
                        trackColor = Color(0x440D1117),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                // Subtle top scrim for depth
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                        .align(Alignment.TopCenter)
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color(0x550D1117), Color.Transparent),
                            )
                        ),
                )

                // Subtle bottom gradient scrim with clean floating speaker identity
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.BottomCenter)
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(Color.Transparent, Color(0xCC0D1117)),
                            )
                        )
                        .padding(horizontal = 18.dp, vertical = 14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Column {
                        Text(
                            text = if (state.activeAvatar == "gretchen") "Gretchen" else "Puck",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.SemiBold,
                            letterSpacing = (-0.3).sp,
                        )
                        Text(
                            text = if (state.busy) state.statusBadge else "${state.renderSide}² HD · LiteRT",
                            color = Color(0xB3F0F6FC),
                            fontSize = 11.sp,
                        )
                    }

                    // Active topic tag when X Showcase or custom topic is armed
                    if (state.demoTopic != state.defaultDemoTopic) {
                        Text(
                            text = if (state.demoTopic == state.xShowcaseTopic) "★ X Showcase" else "Custom Topic",
                            color = Color(0xFFE6EDF3),
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier
                                .clip(RoundedCornerShape(999.dp))
                                .background(Color(0x99161B22))
                                .clickable { showSettingsSheet = true }
                                .padding(horizontal = 10.dp, vertical = 4.dp),
                        )
                    }
                }
            }

            // 3. Clear Visual Hierarchy Control Bar: Segmented Avatar Toggle + Demo + Primary Voice Quiz CTA
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Sleek Segmented Control (Puck | Gretchen)
                Row(
                    modifier = Modifier
                        .weight(1f)
                        .height(44.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(surfaceElevated)
                        .padding(4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    val enabled = state.ready && !state.busy
                    listOf("puck" to "Puck", "gretchen" to "Gretchen").forEach { (key, label) ->
                        val selected = state.activeAvatar == key
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (selected) surfaceActive else Color.Transparent)
                                .clickable(enabled = enabled) { onSelectAvatar(key) },
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = label,
                                color = when {
                                    selected -> textPrimary
                                    enabled -> textSecondary
                                    else -> Color(0xFF484F58)
                                },
                                fontSize = 13.sp,
                                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
                                maxLines = 1,
                                softWrap = false,
                            )
                        }
                    }
                }

                if (state.busy) {
                    Button(
                        onClick = onStop,
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = dangerRed,
                            contentColor = Color.White,
                        ),
                        contentPadding = PaddingValues(horizontal = 18.dp, vertical = 0.dp),
                        modifier = Modifier.height(44.dp),
                    ) {
                        Text("■  Stop", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, softWrap = false)
                    }
                } else {
                    // Secondary compact 3-Turn Demo button
                    Button(
                        onClick = onRunThreeTurnDemo,
                        enabled = state.ready,
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = surfaceHigher,
                            contentColor = textPrimary,
                            disabledContainerColor = surfaceElevated,
                            disabledContentColor = textSecondary,
                        ),
                        contentPadding = PaddingValues(horizontal = 13.dp, vertical = 0.dp),
                        modifier = Modifier.height(44.dp),
                    ) {
                        Text("▶ Demo", fontSize = 12.sp, fontWeight = FontWeight.Medium, maxLines = 1, softWrap = false)
                    }

                    // Single Solid Primary CTA: Voice Study Quiz (or Restart when active)
                    Button(
                        onClick = { if (state.quiz.active) onEndStudyQuiz() else onStartStudyQuiz() },
                        enabled = state.ready,
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (state.quiz.active) surfaceActive else primaryBlue,
                            contentColor = Color.White,
                            disabledContainerColor = surfaceElevated,
                            disabledContentColor = textSecondary,
                        ),
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 0.dp),
                        modifier = Modifier.height(44.dp),
                    ) {
                        Text(
                            text = if (state.quiz.active) "End Quiz" else "★ Quiz",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            softWrap = false,
                        )
                    }
                }
            }

            // 3b. Active Study Quiz Progress Strip (Compose 1.13 CapturedAnimatedVisibility + M3 1.5 LinearWavyProgressIndicator!)
            CapturedAnimatedVisibility(
                visible = state.quiz.active,
                enter = fadeIn() + expandVertically(),
                exit = fadeOut() + shrinkVertically(),
            ) {
                val progress = if (state.quiz.totalCards > 0) {
                    (state.quiz.masteredCount.toFloat() / state.quiz.totalCards.toFloat()).coerceIn(0f, 1f)
                } else 0f
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(surfaceElevated)
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(
                                text = if (state.quiz.completed) "Quiz Complete"
                                else "Card ${state.quiz.cardIndex + 1}/${state.quiz.totalCards} · ${state.quiz.currentTopic}",
                                color = textPrimary,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                            Text(
                                text = "${state.quiz.masteredCount}/${state.quiz.totalCards} mastered",
                                color = emerald,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                maxLines = 1,
                            )
                        }
                        if (!state.quiz.completed && !state.busy) {
                            Text(
                                text = "Skip →",
                                color = primaryBlue,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .clickable { onSkipQuizCard() }
                                    .padding(horizontal = 6.dp, vertical = 2.dp),
                            )
                        }
                    }
                    // Material 3 1.5.0 Stable Expressive Wavy Progress Indicator
                    LinearWavyProgressIndicator(
                        progress = { progress.coerceAtLeast(0.05f) },
                        color = emerald,
                        trackColor = surfaceHigher,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }

            // 4. Borderless Modern Chat Stream (no outer box border, clean left/right elevated bubbles)
            LazyColumn(
                state = state.chatListState,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(vertical = 4.dp),
            ) {
                if (state.chatHistory.isEmpty()) {
                    item {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 24.dp, bottom = 12.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = "Ask Puck or Gretchen anything,\nor tap ★ Quiz for an offline voice study session.",
                                color = textSecondary,
                                fontSize = 13.sp,
                                lineHeight = 19.sp,
                                textAlign = TextAlign.Center,
                            )
                        }
                    }
                }
                items(state.chatHistory) { turn ->
                    when (turn.speaker) {
                        "System" -> {
                            Text(
                                text = turn.text,
                                color = textSecondary,
                                fontSize = 11.sp,
                                textAlign = TextAlign.Center,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp, horizontal = 12.dp),
                            )
                        }
                        "You" -> {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End,
                            ) {
                                Box(
                                    modifier = Modifier
                                        .widthIn(max = 285.dp)
                                        .clip(RoundedCornerShape(18.dp, 18.dp, 4.dp, 18.dp))
                                        .background(primaryBlue)
                                        .padding(horizontal = 14.dp, vertical = 10.dp),
                                ) {
                                    Text(
                                        text = turn.text,
                                        color = Color.White,
                                        fontSize = 14.sp,
                                        lineHeight = 20.sp,
                                    )
                                }
                            }
                        }
                        else -> {
                            val isGretchen = turn.avatarKey == "gretchen"
                            val bubbleBg = if (isGretchen) Color(0xFF1D1829) else surfaceElevated
                            val nameAccent = if (isGretchen) Color(0xFFD2A8FF) else Color(0xFF79C0FF)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.Start,
                            ) {
                                Column(
                                    modifier = Modifier
                                        .widthIn(max = 310.dp)
                                        .clip(RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp))
                                        .background(bubbleBg)
                                        .padding(horizontal = 14.dp, vertical = 10.dp),
                                    verticalArrangement = Arrangement.spacedBy(2.dp),
                                ) {
                                    Text(
                                        text = turn.speaker,
                                        color = nameAccent,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Text(
                                        text = turn.text,
                                        color = Color(0xFFE6EDF3),
                                        fontSize = 14.sp,
                                        lineHeight = 20.sp,
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // 5. Unified Pill-Shaped Composer Bar (Compose 1.13 state-based BasicTextField + Mic + Send)
            val composerState = rememberTextFieldState(initialText = state.inputText)
            LaunchedEffect(state.inputText) {
                if (composerState.text.toString() != state.inputText) {
                    composerState.setTextAndPlaceCursorAtEnd(state.inputText)
                }
            }
            LaunchedEffect(composerState) {
                snapshotFlow { composerState.text.toString() }.collect { txt ->
                    if (txt != state.inputText) onInputChange(txt)
                }
            }
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
                    .clip(RoundedCornerShape(28.dp))
                    .background(surfaceElevated)
                    .padding(start = 18.dp, end = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Box(
                    modifier = Modifier.weight(1f),
                    contentAlignment = Alignment.CenterStart,
                ) {
                    if (composerState.text.isEmpty()) {
                        Text(
                            text = if (state.quiz.active && state.quiz.awaitingAnswer) "Speak or type your answer…"
                            else "Ask Puck or Gretchen…",
                            color = if (state.quiz.active && state.quiz.awaitingAnswer) Color(0xFF79C0FF) else textSecondary,
                            fontSize = 14.sp,
                            maxLines = 1,
                        )
                    }
                    BasicTextField(
                        state = composerState,
                        lineLimits = TextFieldLineLimits.SingleLine,
                        textStyle = TextStyle(
                            color = textPrimary,
                            fontSize = 14.sp,
                            fontFamily = FontFamily.SansSerif,
                        ),
                        cursorBrush = SolidColor(primaryBlue),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        onKeyboardAction = {
                            val txt = composerState.text.toString()
                            if (state.ready && !state.busy && txt.isNotBlank()) {
                                onSendQuestion(txt)
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                // Mic button inside composer pill
                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .clip(CircleShape)
                        .background(surfaceHigher)
                        .clickable(enabled = state.ready && !state.busy) { onMicClick() },
                    contentAlignment = Alignment.Center,
                ) {
                    MicIcon(
                        color = if (state.ready && !state.busy) textPrimary else Color(0xFF484F58),
                        modifier = Modifier.size(18.dp),
                    )
                }

                // Send (or Stop) circular action button inside composer pill
                val canSend = state.ready && !state.busy && state.inputText.isNotBlank()
                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                state.busy -> dangerRed
                                canSend -> primaryBlue
                                else -> surfaceHigher
                            }
                        )
                        .clickable(enabled = state.busy || canSend) {
                            if (state.busy) onStop() else onSendQuestion(state.inputText)
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    if (state.busy) {
                        Box(
                            modifier = Modifier
                                .size(12.dp)
                                .clip(RoundedCornerShape(3.dp))
                                .background(Color.White),
                        )
                    } else {
                        SendArrowIcon(
                            color = if (canSend) Color.White else Color(0xFF484F58),
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }
            }
        }
    }

    // =========================================================================
    // Progressive Disclosure Sheet 1: Stats, Telemetry & Live Engine Log
    // =========================================================================
    if (showStatsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showStatsSheet = false },
            containerColor = surfaceElevated,
            contentColor = textPrimary,
            shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 24.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column {
                        Text("Engine Telemetry", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = textPrimary)
                        Text(state.nanoBadge, fontSize = 12.sp, color = textSecondary)
                    }
                    Text(
                        text = state.acceleratorLabel,
                        color = emerald,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier
                            .clip(RoundedCornerShape(999.dp))
                            .background(surfaceHigher)
                            .padding(horizontal = 10.dp, vertical = 5.dp),
                    )
                }

                // Memory card (monospace strictly for numbers/telemetry!)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(16.dp))
                        .background(bgCanvas)
                        .clickable { onRefreshMem() }
                        .padding(horizontal = 14.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column {
                        Text("Memory Footprint", color = textSecondary, fontSize = 11.sp)
                        Text(
                            text = state.memMain,
                            color = emerald,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                        )
                    }
                    Text(
                        text = state.memSub,
                        color = textSecondary,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                    )
                }

                // 2x2 Timing Grid
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    StatTile("Nano LLM", state.metricLlm, bgCanvas, Modifier.weight(1f))
                    StatTile("Offline TTS", state.metricTts, bgCanvas, Modifier.weight(1f))
                }
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    StatTile("Motion (W2V+FM)", state.metricMotion, bgCanvas, Modifier.weight(1f))
                    StatTile("Renderer ${state.renderSide}²", state.metricRend, bgCanvas, Modifier.weight(1f))
                }

                // Live Engine Log
                Text("Recent Log", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                Text(
                    text = state.logText.lines().filter { it.isNotBlank() }.takeLast(10).joinToString("\n"),
                    color = Color(0xFFC9D1D9),
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    lineHeight = 14.sp,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 150.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(bgCanvas)
                        .verticalScroll(rememberScrollState())
                        .padding(12.dp),
                )
            }
        }
    }

    // =========================================================================
    // Progressive Disclosure Sheet 2: Studio Settings (Topic, Voices, Mode, GPU, UI Switch)
    // =========================================================================
    if (showSettingsSheet) {
        val draftTopicState = rememberTextFieldState(initialText = state.demoTopic)
        val draftTopic = draftTopicState.text.toString()
        val presets = listOf(
            state.xShowcaseTopic,
            state.defaultDemoTopic,
            "Coffee vs tea — which is better?",
            "Running AI 100% offline on Pixel 10",
        )
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            containerColor = surfaceElevated,
            contentColor = textPrimary,
            shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 20.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Text("Studio Settings", fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = textPrimary)

                // Interface Version Switcher (Flagship Studio v2 <-> Classic Console v1)
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Interface Layout", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(40.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(bgCanvas)
                            .padding(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                                .clip(RoundedCornerShape(10.dp))
                                .background(primaryBlue),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("Flagship Studio", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        }
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                                .clip(RoundedCornerShape(10.dp))
                                .clickable {
                                    showSettingsSheet = false
                                    onSwitchToClassic()
                                },
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("Classic Console", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                        }
                    }
                }

                // Voices (tap to cycle & preview per avatar)
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Offline Neural Voices (tap to cycle & audition)", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        VoiceTile("Puck", state.puckVoiceShort, bgCanvas, emerald, state.ready && !state.busy, Modifier.weight(1f)) {
                            onCycleVoice("puck")
                        }
                        VoiceTile("Gretchen", state.gretchenVoiceShort, bgCanvas, Color(0xFFD2A8FF), state.ready && !state.busy, Modifier.weight(1f)) {
                            onCycleVoice("gretchen")
                        }
                    }
                }

                // Playback Mode + Compute Backend
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Column(modifier = Modifier.weight(1.35f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Playback Sync", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(38.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(bgCanvas)
                                .padding(3.dp),
                            horizontalArrangement = Arrangement.spacedBy(3.dp),
                        ) {
                            listOf(2 to "Live", 0 to "12.5", 1 to "25").forEach { (mode, lbl) ->
                                val active = state.playbackMode == mode
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxHeight()
                                        .clip(RoundedCornerShape(9.dp))
                                        .background(if (active) surfaceActive else Color.Transparent)
                                        .clickable(enabled = !state.busy) { onPlaybackModeChange(mode) },
                                    contentAlignment = Alignment.Center,
                                ) {
                                    Text(lbl, color = if (active) textPrimary else textSecondary, fontSize = 11.sp, fontWeight = if (active) FontWeight.SemiBold else FontWeight.Normal)
                                }
                            }
                        }
                    }

                    Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Accelerator", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(38.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(bgCanvas)
                                .padding(3.dp),
                            horizontalArrangement = Arrangement.spacedBy(3.dp),
                        ) {
                            listOf(true to "GPU", false to "CPU").forEach { (gpu, lbl) ->
                                val active = state.preferGpu == gpu
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxHeight()
                                        .clip(RoundedCornerShape(9.dp))
                                        .background(if (active) surfaceActive else Color.Transparent)
                                        .clickable(enabled = !state.busy) {
                                            if (state.preferGpu != gpu) {
                                                showSettingsSheet = false
                                                onPreferGpuChange(gpu)
                                            }
                                        },
                                    contentAlignment = Alignment.Center,
                                ) {
                                    Text(lbl, color = if (active) textPrimary else textSecondary, fontSize = 11.sp, fontWeight = if (active) FontWeight.SemiBold else FontWeight.Normal)
                                }
                            }
                        }
                    }
                }

                // On-Device Gmail Notification Reader (Option 1: zero OAuth, 100% local!)
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Gmail Notification Reader (100% On-Device)", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(14.dp))
                            .background(bgCanvas)
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = if (state.gmailAccessGranted) "Notification Access Connected" else "Notification Access Off",
                                color = if (state.gmailAccessGranted) emerald else amber,
                                fontSize = 13.sp,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                text = if (state.gmailAccessGranted) {
                                    "${state.gmailCachedCount} Gmail notification${if (state.gmailCachedCount == 1) "" else "s"} captured · say \"Read my last email\""
                                } else {
                                    "Enable access so Puck/Gretchen can read Gmail shade notifications"
                                },
                                color = textSecondary,
                                fontSize = 11.sp,
                            )
                        }
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(
                            onClick = {
                                showSettingsSheet = false
                                onOpenGmailAccessSettings()
                            },
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (state.gmailAccessGranted) surfaceHigher else primaryBlue,
                                contentColor = if (state.gmailAccessGranted) textPrimary else Color.White,
                            ),
                            modifier = Modifier.weight(1f).height(40.dp),
                        ) {
                            Text(
                                text = if (state.gmailAccessGranted) "Access Settings" else "Enable Access",
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold,
                            )
                        }

                        Button(
                            onClick = {
                                showSettingsSheet = false
                                onReadLastEmail()
                            },
                            enabled = state.ready && !state.busy,
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (state.gmailAccessGranted) primaryBlue else surfaceHigher,
                                contentColor = if (state.gmailAccessGranted) Color.White else textSecondary,
                            ),
                            modifier = Modifier.weight(1f).height(40.dp),
                        ) { Text("Read Last Email", fontSize = 12.sp, fontWeight = FontWeight.SemiBold) }
                    }
                }

                // Study Quiz Deck (.txt / .md file loader)
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Voice Study Guide (.txt / .md)", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(14.dp))
                            .background(bgCanvas)
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(state.quiz.deckTitle, color = textPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text("${state.quiz.totalCards} study cards · zero-bloat per-card Nano context", color = textSecondary, fontSize = 11.sp)
                        }
                    }
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(
                            onClick = {
                                showSettingsSheet = false
                                onLoadStudyGuideFile()
                            },
                            enabled = !state.busy,
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = surfaceHigher, contentColor = textPrimary),
                            modifier = Modifier.weight(1f).height(40.dp),
                        ) { Text("Load .txt Guide", fontSize = 12.sp, fontWeight = FontWeight.Medium) }

                        Button(
                            onClick = {
                                onResetStudyGuide()
                                showSettingsSheet = false
                                if (state.ready && !state.busy) onStartStudyQuiz()
                            },
                            enabled = state.ready && !state.busy,
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = primaryBlue, contentColor = Color.White),
                            modifier = Modifier.weight(1f).height(40.dp),
                        ) { Text("Start Solar Quiz", fontSize = 12.sp, fontWeight = FontWeight.SemiBold) }
                    }
                }

                // 3-Turn Demo Topic & Quick Presets
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("3-Turn Demo Topic", color = textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                        Text(
                            text = "Reset Default",
                            color = amber,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.clickable {
                                draftTopicState.setTextAndPlaceCursorAtEnd(state.defaultDemoTopic)
                                onDemoTopicSave(state.defaultDemoTopic, false)
                            },
                        )
                    }

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(14.dp))
                            .background(bgCanvas)
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                    ) {
                        BasicTextField(
                            state = draftTopicState,
                            textStyle = TextStyle(color = textPrimary, fontSize = 13.sp, lineHeight = 18.sp),
                            cursorBrush = SolidColor(primaryBlue),
                            lineLimits = TextFieldLineLimits.MultiLine(maxHeightInLines = 3),
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }

                    presets.forEach { preset ->
                        val selected = draftTopic == preset
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(if (selected) surfaceActive else bgCanvas)
                                .clickable { draftTopicState.setTextAndPlaceCursorAtEnd(preset) }
                                .padding(horizontal = 12.dp, vertical = 10.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = preset,
                                color = if (selected) textPrimary else Color(0xFFC9D1D9),
                                fontSize = 12.sp,
                                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                                modifier = Modifier.weight(1f),
                            )
                            if (selected) {
                                Box(
                                    modifier = Modifier
                                        .size(7.dp)
                                        .clip(CircleShape)
                                        .background(emerald),
                                )
                            }
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Button(
                            onClick = {
                                onDemoTopicSave(draftTopic.trim().ifBlank { state.defaultDemoTopic }, false)
                                showSettingsSheet = false
                            },
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = surfaceHigher, contentColor = textPrimary),
                            modifier = Modifier.weight(1f).height(42.dp),
                        ) { Text("Save Topic", fontSize = 13.sp, fontWeight = FontWeight.Medium) }

                        Button(
                            onClick = {
                                onDemoTopicSave(draftTopic.trim().ifBlank { state.defaultDemoTopic }, true)
                                showSettingsSheet = false
                            },
                            enabled = state.ready && !state.busy,
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = primaryBlue, contentColor = Color.White),
                            modifier = Modifier.weight(1f).height(42.dp),
                        ) { Text("Save & Run", fontSize = 13.sp, fontWeight = FontWeight.SemiBold) }
                    }
                }
            }
        }
    }
}

@Composable
private fun StatTile(label: String, value: String, bg: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(bg)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(label, color = Color(0xFF8B949E), fontSize = 11.sp)
        Text(
            text = value,
            color = Color(0xFF79C0FF),
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun VoiceTile(
    name: String,
    voiceShort: String,
    bg: Color,
    accent: Color,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(bg)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column {
            Text(name, color = Color(0xFF8B949E), fontSize = 11.sp)
            Text(voiceShort, color = accent, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = FontFamily.Monospace)
        }
        Text("↻", color = Color(0xFF8B949E), fontSize = 15.sp)
    }
}

@Composable
private fun GearIcon(color: Color, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val rOuter = size.minDimension * 0.42f
        val rInner = size.minDimension * 0.16f
        val stroke = Stroke(width = size.minDimension * 0.11f, cap = StrokeCap.Round)
        drawCircle(color = color, radius = rOuter * 0.72f, center = Offset(cx, cy), style = stroke)
        drawCircle(color = color, radius = rInner, center = Offset(cx, cy))
        for (i in 0 until 8) {
            val angle = i * (Math.PI / 4.0)
            val cosA = kotlin.math.cos(angle).toFloat()
            val sinA = kotlin.math.sin(angle).toFloat()
            drawLine(
                color = color,
                start = Offset(cx + cosA * (rOuter * 0.72f), cy + sinA * (rOuter * 0.72f)),
                end = Offset(cx + cosA * rOuter, cy + sinA * rOuter),
                strokeWidth = size.minDimension * 0.12f,
                cap = StrokeCap.Round,
            )
        }
    }
}

@Composable
private fun MicIcon(color: Color, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val stroke = Stroke(width = w * 0.11f, cap = StrokeCap.Round)
        // Capsule
        drawRoundRect(
            color = color,
            topLeft = Offset(w * 0.34f, h * 0.12f),
            size = Size(w * 0.32f, h * 0.46f),
            cornerRadius = CornerRadius(w * 0.16f, w * 0.16f),
        )
        // U-arc
        drawArc(
            color = color,
            startAngle = 0f,
            sweepAngle = 180f,
            useCenter = false,
            topLeft = Offset(w * 0.20f, h * 0.28f),
            size = Size(w * 0.60f, h * 0.44f),
            style = stroke,
        )
        // Stem + base
        drawLine(color, Offset(w * 0.50f, h * 0.72f), Offset(w * 0.50f, h * 0.88f), strokeWidth = w * 0.11f, cap = StrokeCap.Round)
        drawLine(color, Offset(w * 0.34f, h * 0.88f), Offset(w * 0.66f, h * 0.88f), strokeWidth = w * 0.11f, cap = StrokeCap.Round)
    }
}

@Composable
private fun SendArrowIcon(color: Color, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val stroke = Stroke(width = w * 0.13f, cap = StrokeCap.Round, join = StrokeJoin.Round)
        drawLine(color, Offset(w * 0.50f, h * 0.78f), Offset(w * 0.50f, h * 0.24f), strokeWidth = w * 0.13f, cap = StrokeCap.Round)
        val head = Path().apply {
            moveTo(w * 0.25f, h * 0.48f)
            lineTo(w * 0.50f, h * 0.22f)
            lineTo(w * 0.75f, h * 0.48f)
        }
        drawPath(head, color = color, style = stroke)
    }
}

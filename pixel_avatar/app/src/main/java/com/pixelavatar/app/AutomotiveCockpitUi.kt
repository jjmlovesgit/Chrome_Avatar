package com.pixelavatar.app

import androidx.compose.animation.CapturedAnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.blur.BlurRadiusSpec
import androidx.compose.ui.graphics.blur.BlurStop
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Widescreen Android Automotive OS (AAOS) Cockpit UI built with Jetpack Compose November '26
 * (Compose 1.13 Progressive Blur + CapturedAnimatedVisibility + Material 3 1.5 Wavy Progress).
 *
 * Features >= 64-76dp automotive touch targets, high-contrast glanceable cabin typography,
 * and a Parked / Driving UX-restriction mode toggle demonstrating distraction-optimized design.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AutomotiveCockpitUi(
    state: StudioUiState,
    isDriving: Boolean,
    onToggleDriving: () -> Unit,
    onSelectAvatar: (String) -> Unit,
    onPushToTalk: () -> Unit,
    onRunThreeTurnDemo: () -> Unit,
    onStartStudyQuiz: () -> Unit,
    onEndStudyQuiz: () -> Unit,
    onSkipQuizCard: () -> Unit,
    onReadLastEmail: () -> Unit,
    onQuickAsk: (String) -> Unit,
    onStop: () -> Unit,
    onCycleVoice: (String) -> Unit,
    onPreferGpuChange: (Boolean) -> Unit,
    onRefreshMem: () -> Unit,
    onSwitchUiMode: (Int) -> Unit, // 0 = Studio, 1 = Classic, 2 = Auto Cockpit
) {
    val bgCanvas = Color(0xFF0A0D12)
    val surfaceCard = Color(0xFF151A22)
    val surfaceRaised = Color(0xFF1E2530)
    val surfaceActive = Color(0xFF2B3545)
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
        Row(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(18.dp),
            horizontalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            // =================================================================
            // LEFT PANE (42%): Hero Avatar Cockpit Viewport w/ Progressive Blur
            // =================================================================
            Box(
                modifier = Modifier
                    .weight(0.42f)
                    .fillMaxHeight()
                    .clip(RoundedCornerShape(28.dp))
                    .background(surfaceCard),
                contentAlignment = Alignment.Center,
            ) {
                val bmp = state.frameBmp
                if (bmp != null) {
                    key(state.frameVersion) {
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "In-Car Live 512x512 HD Avatar",
                            modifier = Modifier
                                .fillMaxSize()
                                .blur {
                                    // Compose 1.13 Progressive Blur: crisp face, frosted bottom caption bar
                                    blurRadiusSpec = BlurRadiusSpec.verticalGradient(
                                        listOf(
                                            BlurStop(0f, 0.dp),
                                            BlurStop(0.75f, 0.dp),
                                            BlurStop(1f, 16.dp),
                                        )
                                    )
                                },
                            contentScale = ContentScale.Crop,
                        )
                    }
                }

                // Compose 1.13 CapturedAnimatedVisibility + M3 1.5 Expressive Wavy loading bar
                val isPreparing = !state.ready || (state.busy && !state.statusBadge.startsWith("Speaking"))
                CapturedAnimatedVisibility(
                    visible = isPreparing,
                    enter = fadeIn(),
                    exit = fadeOut(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.TopCenter)
                        .padding(horizontal = 20.dp, vertical = 14.dp),
                ) {
                    LinearWavyProgressIndicator(
                        color = primaryBlue,
                        trackColor = Color(0x550A0D12),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }

                // Top-left gear/drive state pill (tap to simulate Parked <-> Driving UX restrictions)
                Row(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(16.dp)
                        .clip(RoundedCornerShape(999.dp))
                        .background(Color(0xCC0A0D12))
                        .clickable { onToggleDriving() }
                        .padding(horizontal = 14.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Box(
                        modifier = Modifier
                            .size(9.dp)
                            .clip(CircleShape)
                            .background(if (isDriving) amber else emerald),
                    )
                    Text(
                        text = if (isDriving) "D · DRIVING" else "P · PARKED",
                        color = textPrimary,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                    )
                }

                // Bottom frosted gradient overlay with large glanceable speaker identity
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.BottomCenter)
                        .background(
                            Brush.verticalGradient(listOf(Color.Transparent, Color(0xD90A0D12)))
                        )
                        .padding(horizontal = 22.dp, vertical = 18.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Column {
                        Text(
                            text = if (state.activeAvatar == "gretchen") "Gretchen" else "Puck",
                            color = Color.White,
                            fontSize = 26.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = (-0.5).sp,
                        )
                        Text(
                            text = if (state.busy) state.statusBadge else "100% Offline In-Car AI · ${state.renderSide}² HD",
                            color = Color(0xCCF0F6FC),
                            fontSize = 13.sp,
                        )
                    }
                }
            }

            // =================================================================
            // RIGHT PANE (58%): Glanceable Automotive Controls & Cabin Transcript
            // =================================================================
            Column(
                modifier = Modifier
                    .weight(0.58f)
                    .fillMaxHeight(),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                // Header Row: Brand + Telemetry Pill + Settings Button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column {
                        Text(
                            text = "Pixel Auto Assistant",
                            color = textPrimary,
                            fontSize = 24.sp,
                            fontWeight = FontWeight.SemiBold,
                            letterSpacing = (-0.4).sp,
                        )
                        val llmTag = if (state.nanoReady) state.llmEngineLabel else "Offline Fallback"
                        Text(
                            text = if (isDriving) "Hands-Free Driving · ${state.acceleratorLabel} · $llmTag"
                            else "LiteRT ${state.acceleratorLabel} · $llmTag · Cabin Ducking",
                            color = if (isDriving) amber else textSecondary,
                            fontSize = 13.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }

                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(999.dp))
                                .background(surfaceCard)
                                .clickable {
                                    onRefreshMem()
                                    showStatsSheet = true
                                }
                                .padding(horizontal = 16.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Box(modifier = Modifier.size(9.dp).clip(CircleShape).background(statusDotColor))
                            Text(
                                text = state.statusBadge,
                                color = textPrimary,
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Medium,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }

                        Box(
                            modifier = Modifier
                                .size(44.dp)
                                .clip(CircleShape)
                                .background(surfaceCard)
                                .clickable { showSettingsSheet = true },
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("⚙", color = textPrimary, fontSize = 20.sp)
                        }
                    }
                }

                // Big Automotive Control Row: Puck | Gretchen + Primary Push-to-Talk / Stop (>= 68dp tall!)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Row(
                        modifier = Modifier
                            .weight(1f)
                            .height(68.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(surfaceCard)
                            .padding(6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        val enabled = state.ready && !state.busy
                        listOf("puck" to "Puck", "gretchen" to "Gretchen").forEach { (key, label) ->
                            val selected = state.activeAvatar == key
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxHeight()
                                    .clip(RoundedCornerShape(15.dp))
                                    .background(if (selected) surfaceActive else Color.Transparent)
                                    .clickable(enabled = enabled) { onSelectAvatar(key) },
                                contentAlignment = Alignment.Center,
                            ) {
                                Text(
                                    text = label,
                                    color = if (selected) textPrimary else textSecondary,
                                    fontSize = 17.sp,
                                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                                    maxLines = 1,
                                )
                            }
                        }
                    }

                    if (state.busy) {
                        Button(
                            onClick = onStop,
                            shape = RoundedCornerShape(20.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = dangerRed, contentColor = Color.White),
                            modifier = Modifier.weight(1f).height(68.dp),
                        ) {
                            Text("■   Stop Speaking", fontSize = 17.sp, fontWeight = FontWeight.Bold)
                        }
                    } else if (state.isListening) {
                        Button(
                            onClick = onPushToTalk,
                            shape = RoundedCornerShape(20.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = amber, contentColor = Color(0xFF0A0D12)),
                            modifier = Modifier.weight(1f).height(68.dp),
                        ) {
                            Text(
                                text = "🎙  " + state.listeningPartial.ifBlank { "Listening… (Tap to Stop)" },
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                    } else {
                        Button(
                            onClick = onPushToTalk,
                            enabled = state.ready,
                            shape = RoundedCornerShape(20.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = primaryBlue,
                                contentColor = Color.White,
                                disabledContainerColor = surfaceCard,
                                disabledContentColor = textSecondary,
                            ),
                            modifier = Modifier.weight(1f).height(68.dp),
                        ) {
                            Text("🎤  Tap to Speak", fontSize = 17.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }

                // Quick Cockpit Voice Actions (large 64dp automotive tiles)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    if (isDriving) {
                        // Distraction-optimized driving presets (zero keyboard, single big taps)
                        CockpitTile("✉  Read Email", surfaceCard, textPrimary, state.ready && !state.busy, Modifier.weight(1f)) {
                            onReadLastEmail()
                        }
                        CockpitTile("★  Voice Quiz", surfaceCard, if (state.quiz.active) emerald else textPrimary, state.ready && !state.busy, Modifier.weight(1f)) {
                            if (state.quiz.active) onEndStudyQuiz() else onStartStudyQuiz()
                        }
                        CockpitTile("🚗  Trip Idea", surfaceCard, textPrimary, state.ready && !state.busy, Modifier.weight(1f)) {
                            onQuickAsk("Suggest a fun scenic stop on our drive in one short sentence.")
                        }
                    } else {
                        CockpitTile("▶  3-Turn Demo", surfaceCard, textPrimary, state.ready && !state.busy, Modifier.weight(1f)) {
                            onRunThreeTurnDemo()
                        }
                        CockpitTile(
                            label = if (state.quiz.active) "End Quiz" else "★  Solar Quiz",
                            bg = if (state.quiz.active) surfaceActive else surfaceCard,
                            fg = if (state.quiz.active) emerald else textPrimary,
                            enabled = state.ready && !state.busy,
                            modifier = Modifier.weight(1f),
                        ) {
                            if (state.quiz.active) onEndStudyQuiz() else onStartStudyQuiz()
                        }
                        CockpitTile("✉  Read Email", surfaceCard, textPrimary, state.ready && !state.busy, Modifier.weight(1f)) {
                            onReadLastEmail()
                        }
                    }
                }

                // Active Quiz Progress Bar (Compose 1.13 CapturedAnimatedVisibility + M3 1.5 Wavy Progress)
                CapturedAnimatedVisibility(
                    visible = state.quiz.active,
                    enter = fadeIn(),
                    exit = fadeOut(),
                ) {
                    val progress = if (state.quiz.totalCards > 0) {
                        (state.quiz.masteredCount.toFloat() / state.quiz.totalCards.toFloat()).coerceIn(0f, 1f)
                    } else 0f
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(18.dp))
                            .background(surfaceCard)
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = "Card ${state.quiz.cardIndex + 1}/${state.quiz.totalCards} · ${state.quiz.currentTopic}  (${state.quiz.masteredCount} mastered)",
                                color = textPrimary,
                                fontSize = 15.sp,
                                fontWeight = FontWeight.SemiBold,
                            )
                            if (!state.quiz.completed && !state.busy) {
                                Text(
                                    text = "Skip →",
                                    color = primaryBlue,
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.clickable { onSkipQuizCard() },
                                )
                            }
                        }
                        LinearWavyProgressIndicator(
                            progress = { progress.coerceAtLeast(0.05f) },
                            color = emerald,
                            trackColor = surfaceRaised,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                }

                // Glanceable Cabin Transcript Stream (large 17sp text, high contrast, zero clutter)
                LazyColumn(
                    state = state.chatListState,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .clip(RoundedCornerShape(22.dp))
                        .background(surfaceCard)
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    if (state.chatHistory.isEmpty()) {
                        item {
                            Text(
                                text = if (isDriving) {
                                    "Driving Mode active. Tap 'Tap to Speak' or ask Puck to read your last email or run a hands-free Solar System voice quiz."
                                } else {
                                    "Ready in Parked Mode. Tap '▶ 3-Turn Demo' for the LiteRT X Showcase, '★ Solar Quiz' for hands-free study, or '🎤 Tap to Speak'!"
                                },
                                color = textSecondary,
                                fontSize = 16.sp,
                                lineHeight = 24.sp,
                            )
                        }
                    }
                    items(state.chatHistory) { turn ->
                        val isSystem = turn.speaker == "System"
                        val isUser = turn.speaker == "You"
                        val isGretchen = turn.avatarKey == "gretchen"
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp))
                                .background(
                                    when {
                                        isSystem -> Color.Transparent
                                        isUser -> primaryBlue
                                        isGretchen -> Color(0xFF211A30)
                                        else -> surfaceRaised
                                    }
                                )
                                .padding(horizontal = 16.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(3.dp),
                        ) {
                            if (!isSystem) {
                                Text(
                                    text = turn.speaker,
                                    color = if (isUser) Color(0xCCFFFFFF) else if (isGretchen) Color(0xFFD2A8FF) else Color(0xFF79C0FF),
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold,
                                )
                            }
                            Text(
                                text = turn.text,
                                color = if (isSystem) textSecondary else textPrimary,
                                fontSize = if (isSystem) 13.sp else 17.sp,
                                lineHeight = if (isSystem) 18.sp else 24.sp,
                            )
                        }
                    }
                }
            }
        }
    }

    // Telemetry Sheet
    if (showStatsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showStatsSheet = false },
            containerColor = surfaceCard,
            contentColor = textPrimary,
            shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Automotive Engine Telemetry (${state.acceleratorLabel})", fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
                Text("${state.memMain} · ${state.memSub}", color = emerald, fontSize = 14.sp, fontFamily = FontFamily.Monospace)
                Text(
                    "LLM: ${state.metricLlm}  |  TTS: ${state.metricTts}  |  Motion: ${state.metricMotion}  |  Rend ${state.renderSide}²: ${state.metricRend}",
                    color = Color(0xFF79C0FF),
                    fontSize = 13.sp,
                    fontFamily = FontFamily.Monospace,
                )
                Text(
                    text = state.logText.lines().filter { it.isNotBlank() }.takeLast(10).joinToString("\n"),
                    color = Color(0xFFC9D1D9),
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 160.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(bgCanvas)
                        .verticalScroll(rememberScrollState())
                        .padding(12.dp),
                )
            }
        }
    }

    // Cockpit Settings Sheet (includes UI layout switcher back to Phone Studio / Classic Console)
    if (showSettingsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            containerColor = surfaceCard,
            contentColor = textPrimary,
            shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
                    .padding(bottom = 28.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Text("Cockpit Settings", fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    listOf(2 to "Auto Cockpit", 0 to "Phone Studio", 1 to "Classic Console").forEach { (mode, lbl) ->
                        Button(
                            onClick = {
                                showSettingsSheet = false
                                onSwitchUiMode(mode)
                            },
                            shape = RoundedCornerShape(14.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (mode == 2) primaryBlue else surfaceRaised,
                                contentColor = textPrimary,
                            ),
                            modifier = Modifier.weight(1f).height(48.dp),
                        ) { Text(lbl, fontSize = 13.sp, fontWeight = FontWeight.SemiBold) }
                    }
                }
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(
                        onClick = { onCycleVoice("puck") },
                        enabled = state.ready && !state.busy,
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = surfaceRaised, contentColor = emerald),
                        modifier = Modifier.weight(1f).height(48.dp),
                    ) { Text("Puck Voice: ${state.puckVoiceShort}", fontSize = 13.sp) }

                    Button(
                        onClick = { onCycleVoice("gretchen") },
                        enabled = state.ready && !state.busy,
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = surfaceRaised, contentColor = Color(0xFFD2A8FF)),
                        modifier = Modifier.weight(1f).height(48.dp),
                    ) { Text("Gretchen Voice: ${state.gretchenVoiceShort}", fontSize = 13.sp) }

                    Button(
                        onClick = {
                            showSettingsSheet = false
                            onPreferGpuChange(!state.preferGpu)
                        },
                        enabled = !state.busy,
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = surfaceRaised, contentColor = primaryBlue),
                        modifier = Modifier.weight(0.8f).height(48.dp),
                    ) { Text(if (state.preferGpu) "Backend: GPU" else "Backend: CPU", fontSize = 13.sp) }
                }
            }
        }
    }
}

@Composable
private fun CockpitTile(
    label: String,
    bg: Color,
    fg: Color,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Box(
        modifier = modifier
            .height(60.dp)
            .clip(RoundedCornerShape(18.dp))
            .background(bg)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 12.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = label,
            color = if (enabled) fg else Color(0xFF484F58),
            fontSize = 15.sp,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

package dev.scorefromaudio.app.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import dev.scorefromaudio.app.Format
import dev.scorefromaudio.app.LevelHint
import dev.scorefromaudio.app.Recorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

private val METERS = listOf("Not sure", "3/4", "4/4", "6/8")

/**
 * Record: a tip, the timer, a live level meter with a hint, optional beats per bar; discard, record/stop and
 * "Make score". [onMake] gets the stopped recording and the meter picked (null for "Not sure").
 */
@Composable
fun RecordScreen(onBack: () -> Unit, onMake: (File, String?) -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val state by Recorder.state.collectAsState()
    var meter by rememberSaveable { mutableStateOf(METERS[0]) }
    var confirmDiscard by rememberSaveable { mutableStateOf(false) }
    var denied by rememberSaveable { mutableStateOf(false) }
    val recording = state?.recording == true
    val stopped = state != null && !recording && state?.file != null
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        denied = !granted
        if (granted) Recorder.start(context)
    }
    fun record() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) Recorder.start(context)
        else permission.launch(Manifest.permission.RECORD_AUDIO)
    }
    fun stop(then: () -> Unit = {}) = scope.launch {
        withContext(Dispatchers.IO) { Recorder.stop(context) }
        then()
    }
    fun make() = stop {
        val file = Recorder.release() ?: return@stop
        onMake(file, meter.takeIf { it != METERS[0] })
    }
    fun leave() {
        if (recording || stopped) confirmDiscard = true else { Recorder.discard(context); onBack() }
    }
    androidx.activity.compose.BackHandler { leave() }
    // Start listening as soon as the screen opens: one tap fewer, and the level meter shows at once.
    LaunchedEffect(Unit) { if (state == null) record() }

    if (confirmDiscard) AlertDialog(
        onDismissRequest = { confirmDiscard = false },
        title = { Text("Discard this recording?") },
        text = { Text("It will be deleted and no score made from it.") },
        confirmButton = { TextButton(onClick = {
            confirmDiscard = false
            scope.launch { withContext(Dispatchers.IO) { Recorder.discard(context) }; onBack() }
        }) { Text("Discard", color = Ink.Red) } },
        dismissButton = { TextButton(onClick = { confirmDiscard = false }) { Text("Keep") } },
    )

    Column(Modifier.fillMaxSize().background(Ink.Paper)) {
        Header("New recording", ::leave)
        Text("Put the phone near the piano in a quiet room. Only piano works for now.",
            fontSize = 14.sp, lineHeight = 20.sp, color = Ink.BlueDeep,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp).fillMaxWidth()
                .clip(RoundedCornerShape(12.dp)).background(Ink.BlueTint).padding(horizontal = 14.dp, vertical = 12.dp))
        Column(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(22.dp, Alignment.CenterVertically),
            horizontalAlignment = Alignment.CenterHorizontally) {
            Text(Format.clock(state?.elapsedMs ?: 0), fontFamily = FontFamily.Serif, fontSize = 56.sp, fontWeight = FontWeight.Medium, color = Ink.Ink)
            Meter(state?.bars.orEmpty(), recording)
            val hint = state?.hint
            val error = state?.error
            when {
                denied -> Text("The app needs the microphone to record. Allow it in the phone's settings.", color = Ink.Red, fontSize = 14.sp)
                error != null -> Text(error, color = Ink.Red, fontSize = 14.sp)
                stopped -> Text("Stopped. Make the score, or discard it.", style = Type.Small)
                recording && hint != null -> HintLine(hint)
                else -> Text(" ", fontSize = 14.sp)
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Beats per bar (optional)", style = Type.Small)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (m in METERS) Chip(m, m == meter, { meter = m })
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 32.dp),
            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            Box(Modifier.size(56.dp).clip(CircleShape).border(1.dp, Ink.Outline, CircleShape)
                .clickable(enabled = recording || stopped, role = Role.Button) { confirmDiscard = true }
                .semantics { contentDescription = "Discard recording" }, contentAlignment = Alignment.Center) {
                Canvas(Modifier.size(22.dp)) { Icons.trash(this, if (recording || stopped) Ink.Ink else Ink.Outline) }
            }
            val label = if (recording) "Stop recording" else "Start recording"
            Box(Modifier.size(84.dp).clip(CircleShape).background(if (stopped) Ink.Track else Ink.Red)
                .border(4.dp, if (stopped) Ink.Outline else Ink.RedRing, CircleShape)
                .clickable(enabled = !stopped, role = Role.Button) { if (recording) stop() else record() }
                .semantics { contentDescription = label }, contentAlignment = Alignment.Center) {
                Canvas(Modifier.size(28.dp)) { if (recording || stopped) Icons.stop(this, Color.White) else Icons.dot(this, Color.White) }
            }
            Pill("Make score", ::make, height = 56.dp, enabled = (recording || stopped) && (state?.elapsedMs ?: 0) >= 2_000)
        }
    }
}

@Composable
private fun HintLine(hint: LevelHint) {
    val color = when (hint) { LevelHint.GOOD -> Ink.Green; LevelHint.LISTENING -> Ink.Muted; else -> Ink.Warn }
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (hint == LevelHint.GOOD) Canvas(Modifier.size(16.dp)) { Icons.check(this, color) }
        Text(hint.text, color = color, fontSize = 14.sp, fontWeight = FontWeight.Medium)
    }
}

/** The newest loudness values as rounded bars from the left, dots where there is nothing yet. */
@Composable
private fun Meter(bars: List<Float>, live: Boolean) {
    Canvas(Modifier.fillMaxWidth().height(90.dp).semantics { contentDescription = "Live sound level" }) {
        val slots = 36
        val step = size.width / slots
        val w = step * 0.6f
        for (i in 0 until slots) {
            val v = bars.getOrNull(i)
            val h = if (v == null) w else maxOf(w, v * size.height)
            drawRoundRect(if (v == null) Ink.Outline else if (live) Ink.Blue else Ink.Muted,
                Offset(i * step, (size.height - h) / 2), Size(w, h), CornerRadius(w / 2))
        }
    }
}

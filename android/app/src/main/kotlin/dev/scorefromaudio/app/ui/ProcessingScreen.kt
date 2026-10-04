package dev.scorefromaudio.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.scorefromaudio.app.JobMeta
import dev.scorefromaudio.app.Progress
import dev.scorefromaudio.app.Reasons
import dev.scorefromaudio.app.Step
import dev.scorefromaudio.app.Steps
import kotlin.math.roundToInt

/**
 * Making your score: "Step N of 4", a progress bar and the four named steps. The work runs in a foreground service,
 * so leaving this screen does not stop it. A failed job keeps its recording and offers Try again with a plain reason.
 */
@Composable
fun ProcessingScreen(job: JobMeta?, progress: Progress?, onBack: () -> Unit, onStop: () -> Unit, onRetry: () -> Unit,
                     onDelete: () -> Unit) {
    val failed = job?.status == "failed"
    val queued = job?.status == "queued"
    val live = progress?.takeIf { job != null && it.jobId == job.id }
    val step = when {
        job?.status == "done" -> Step.LAYOUT
        live != null -> Steps.of(live.stage)
        job != null && job.stage.isNotEmpty() -> Steps.of(job.stage)
        else -> Step.LISTENING
    }
    val overall = when {
        job?.status == "done" -> 0.98f
        live != null -> Steps.overall(live.stage, live.fraction)
        else -> 0f
    }
    Column(Modifier.fillMaxSize().background(Ink.Paper)) {
        Header(if (failed) "Your score couldn't be made" else "Making your score", onBack, size = if (failed) 20 else 24)
        if (failed) {
            Failed(job!!, onRetry, onDelete, onBack)
            return@Column
        }
        Column(Modifier.padding(start = 24.dp, end = 24.dp, top = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(if (queued) "Waiting its turn" else "Step ${step.ordinal + 1} of 4", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Ink.Ink)
                Text(if (queued) "" else remaining(live, overall), fontSize = 15.sp, color = Ink.Muted)
            }
            Bar(overall)
        }
        Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
            for (s in Step.entries) StepRow(s, when {
                queued -> 0
                s.ordinal < step.ordinal -> 2
                s == step -> 1
                else -> 0
            })
        }
        if (queued) Text("Another score is being made. This one starts when it is done.", style = Type.Small,
            modifier = Modifier.padding(horizontal = 24.dp))
        Box(Modifier.weight(1f))
        Text("You can leave this screen or lock your phone. We'll send a notification when it's ready.",
            fontSize = 14.sp, lineHeight = 20.sp, color = Ink.Muted, textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, bottom = 16.dp))
        Column(Modifier.padding(start = 20.dp, end = 20.dp, bottom = 24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            OutlinePill("Back to scores", onBack, Modifier.fillMaxWidth(), height = 52.dp)
            if (live != null) TextButton(onClick = onStop, modifier = Modifier.height(48.dp)) { Text("Stop", color = Ink.Muted) }
        }
    }
}

private class Start(val millis: Long, val fraction: Float)

/** Rough time left from how fast the job has moved while this screen watched it; a percentage until that is known. */
@Composable
private fun remaining(live: Progress?, overall: Float): String {
    val start = remember(live?.jobId) { arrayOfNulls<Start>(1) }
    val percent = "${(overall * 100).roundToInt()}%"
    if (live == null || live.jobId.isEmpty()) return percent
    val now = System.currentTimeMillis()
    val from = start[0] ?: Start(now, overall).also { start[0] = it }
    val done = overall - from.fraction
    val took = now - from.millis
    if (done < 0.05f || took < 5_000) return percent
    val left = took * (1 - overall) / done / 1000
    return when {
        left < 50 -> "Less than a minute left"
        left < 90 -> "About a minute left"
        else -> "About ${(left / 60.0).roundToInt()} minutes left"
    }
}

@Composable
private fun Bar(fraction: Float) {
    Box(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp)).background(Ink.Track)) {
        Box(Modifier.fillMaxWidth(fraction.coerceIn(0f, 1f)).height(10.dp).clip(RoundedCornerShape(5.dp)).background(Ink.Blue))
    }
}

/** [state] 0 waiting, 1 working, 2 done. */
@Composable
private fun StepRow(step: Step, state: Int) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
        when (state) {
            2 -> Box(Modifier.size(32.dp).clip(CircleShape).background(Ink.Green), contentAlignment = Alignment.Center) {
                Canvas(Modifier.size(18.dp)) { Icons.check(this, Color.White, 3f) }
            }
            1 -> Box(Modifier.size(32.dp).border(3.dp, Ink.Blue, CircleShape), contentAlignment = Alignment.Center) {
                Box(Modifier.size(10.dp).clip(CircleShape).background(Ink.Blue))
            }
            else -> Box(Modifier.size(32.dp).border(2.dp, Ink.Outline, CircleShape))
        }
        Column {
            Text(step.label, fontSize = 16.sp, fontWeight = if (state == 1) FontWeight.SemiBold else FontWeight.Medium,
                color = when (state) { 1 -> Ink.Blue; 2 -> Ink.Ink; else -> Ink.Muted })
            when (state) {
                2 -> Text("Done", style = Type.Caption)
                1 -> Text("Working", style = Type.Caption)
            }
        }
    }
}

@Composable
private fun Failed(job: JobMeta, onRetry: () -> Unit, onDelete: () -> Unit, onBack: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(job.title, fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Ink.Ink)
        Text(Reasons.plain(job), fontSize = 16.sp, lineHeight = 22.sp, color = Ink.Ink)
        Text(if (job.audio != null) "Your recording is kept, so you can try again." else "Nothing was downloaded, so trying again fetches the link again.",
            style = Type.Small)
        Box(Modifier.weight(1f))
        Pill("Try again", onRetry, Modifier.fillMaxWidth())
        OutlinePill(if (job.audio != null) "Delete recording" else "Delete", onDelete, Modifier.fillMaxWidth())
        TextButton(onClick = onBack, modifier = Modifier.align(Alignment.CenterHorizontally).height(48.dp)) { Text("Back to scores", color = Ink.Muted) }
    }
}

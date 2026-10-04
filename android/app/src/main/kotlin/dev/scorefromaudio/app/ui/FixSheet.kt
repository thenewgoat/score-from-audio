package dev.scorefromaudio.app.ui

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.scorefromaudio.app.Format
import dev.scorefromaudio.pipeline.score.Basics

private val METERS = listOf("2/4", "3/4", "4/4", "6/8")

/**
 * "Fix the basics": beats per bar, key and tempo. Redraw hands back the whole set, which redraws the whole score;
 * [redrawing] while that runs.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FixSheet(meter: String?, fifths: Int?, bpm: Int?, redrawing: Boolean, onDismiss: () -> Unit, onRedraw: (Basics) -> Unit) {
    var chosen by rememberSaveable { mutableStateOf(meter?.takeIf { it in METERS } ?: "4/4") }
    var key by rememberSaveable { mutableIntStateOf(fifths ?: 0) }
    var tempo by rememberSaveable { mutableIntStateOf(bpm ?: 120) }
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        containerColor = Ink.Card) {
        Column(Modifier.padding(start = 20.dp, end = 20.dp, bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(22.dp)) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Fix the basics", style = Type.Heading)
                Text("Changing these redraws the whole score.", style = Type.Small)
            }
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Beats per bar", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Ink.Ink)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (m in METERS) Chip(m, m == chosen, { chosen = m }, Modifier.weight(1f))
                }
            }
            Stepper("Key", Format.key(key), "Previous key", "Next key", true,
                { key = if (key <= -7) 7 else key - 1 }, { key = if (key >= 7) -7 else key + 1 })
            Stepper("Tempo", "$tempo beats per minute", "Slower", "Faster", false,
                { tempo = (tempo - 2).coerceAtLeast(20) }, { tempo = (tempo + 2).coerceAtMost(300) })
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinePill("Cancel", onDismiss, Modifier.weight(1f), height = 52.dp)
                Pill(if (redrawing) "Redrawing…" else "Redraw score", {
                    val (beats, type) = Basics.meter(chosen)!!
                    onRedraw(Basics(beats, type, key, tempo))
                }, Modifier.weight(2f), enabled = !redrawing)
            }
        }
    }
}

@Composable
private fun Stepper(title: String, value: String, less: String, more: String, arrows: Boolean, onLess: () -> Unit, onMore: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = Ink.Ink)
        Row(Modifier.fillMaxWidth().height(52.dp).border(1.dp, Ink.Outline, RoundedCornerShape(14.dp)).padding(horizontal = 2.dp),
            verticalAlignment = Alignment.CenterVertically) {
            IconTap(less, onLess) { if (arrows) Icons.chevronLeft(this, Ink.Ink) else drawLine(Ink.Ink,
                androidx.compose.ui.geometry.Offset(size.width * 0.25f, size.height / 2), androidx.compose.ui.geometry.Offset(size.width * 0.75f, size.height / 2), size.width / 12) }
            Text(value, fontSize = 16.sp, fontWeight = FontWeight.Medium, color = Ink.Ink, modifier = Modifier.weight(1f),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center)
            IconTap(more, onMore) {
                if (arrows) Icons.chevronRight(this, Ink.Ink) else {
                    drawLine(Ink.Ink, androidx.compose.ui.geometry.Offset(size.width * 0.25f, size.height / 2), androidx.compose.ui.geometry.Offset(size.width * 0.75f, size.height / 2), size.width / 12)
                    drawLine(Ink.Ink, androidx.compose.ui.geometry.Offset(size.width / 2, size.height * 0.25f), androidx.compose.ui.geometry.Offset(size.width / 2, size.height * 0.75f), size.width / 12)
                }
            }
        }
    }
}

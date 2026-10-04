package dev.scorefromaudio.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.scorefromaudio.app.Format
import dev.scorefromaudio.app.JobMeta
import dev.scorefromaudio.app.Progress
import dev.scorefromaudio.app.Steps

/**
 * Home: past scores, newest first, with the one being made showing its progress; "Record piano" and
 * "Import audio file" at the bottom. Links and About (the Instagram login, terms, privacy, credits)
 * sit in the ⋮ menu.
 */
@Composable
fun LibraryScreen(jobs: List<JobMeta>, progress: Progress?, onOpen: (JobMeta) -> Unit, onDelete: (JobMeta) -> Unit,
                  onRecord: () -> Unit, onImport: () -> Unit, onPasteLink: () -> Unit, onSettings: () -> Unit) {
    var deleting by remember { mutableStateOf<JobMeta?>(null) }
    deleting?.let { job ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("Delete this score?") },
            text = { Text("“${job.title}”, its sheet music and its recording are deleted from the phone. This can't be undone.") },
            confirmButton = { TextButton(onClick = { deleting = null; onDelete(job) }) { Text("Delete", color = Ink.Red) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("Cancel") } },
        )
    }
    Column(Modifier.fillMaxSize().background(Ink.Paper)) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 24.dp, bottom = 12.dp), verticalAlignment = Alignment.Top) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Scores", style = Type.Title)
                Text("Record piano, get sheet music.", style = Type.Small.copy(fontSize = 15.sp))
            }
            var menu by remember { mutableStateOf(false) }
            Box {
                IconTap("More", { menu = true }) { Icons.more(this, Ink.Ink) }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(text = { Text("Paste a YouTube or Instagram link") }, onClick = { menu = false; onPasteLink() })
                    DropdownMenuItem(text = { Text("Settings, terms and privacy") }, onClick = { menu = false; onSettings() })
                }
            }
        }
        if (jobs.isEmpty()) {
            Column(Modifier.weight(1f).fillMaxWidth().padding(24.dp), verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally) {
                Text("No scores yet", style = Type.Heading.copy(fontSize = 20.sp))
                Text("Record your piano, or import a recording, and the sheet music appears here.",
                    style = Type.Small, modifier = Modifier.padding(top = 8.dp))
            }
        } else {
            LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 20.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(jobs, key = { it.id }) { job -> Row(job, progress?.takeIf { it.jobId == job.id }, { onOpen(job) }, { deleting = job }) }
            }
        }
        HorizontalDivider(color = Ink.Line)
        Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 24.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Pill("Record piano", onRecord, Modifier.fillMaxWidth(), color = Ink.Red, height = 60.dp,
                icon = { Canvas(Modifier.size(22.dp)) { Icons.mic(this, Color.White) } })
            OutlinePill("Import audio file", onImport, Modifier.fillMaxWidth(),
                icon = { Canvas(Modifier.size(20.dp)) { Icons.file(this, Ink.Ink) } })
        }
    }
}

@Composable
private fun Row(job: JobMeta, progress: Progress?, onClick: () -> Unit, onDelete: () -> Unit) {
    val working = job.status == "running" || job.status == "queued"
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Ink.Card)
            .border(1.dp, Ink.Line, RoundedCornerShape(14.dp)).clickable(role = Role.Button, onClick = onClick).padding(start = 14.dp, top = 14.dp, bottom = 14.dp, end = 4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Box(Modifier.size(44.dp).clip(RoundedCornerShape(10.dp)).background(Color(0xFFEEE8DB)).border(1.dp, Ink.Line, RoundedCornerShape(10.dp)),
            contentAlignment = Alignment.Center) {
            if (working) CircularProgressIndicator(Modifier.size(20.dp), color = Ink.Blue, strokeWidth = 2.5.dp)
            else Canvas(Modifier.size(44.dp)) { Icons.staff(this, job.id.hashCode() % 2 == 0) }
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(job.title, fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = Ink.Ink, maxLines = 1, overflow = TextOverflow.Ellipsis)
            when {
                progress != null -> {
                    val stage = progress.stage
                    Text("${Steps.of(stage).label}… ${(Steps.overall(stage, progress.fraction) * 100).toInt()}%",
                        fontSize = 13.sp, color = Ink.Blue, fontWeight = FontWeight.Medium)
                }
                job.status == "running" -> Text("Making the score…", fontSize = 13.sp, color = Ink.Blue, fontWeight = FontWeight.Medium)
                job.status == "queued" -> Text("Waiting its turn", fontSize = 13.sp, color = Ink.Blue, fontWeight = FontWeight.Medium)
                job.status == "failed" -> Text("Couldn't make the score · tap to try again", fontSize = 13.sp, color = Ink.Red)
                else -> Text(Format.summary(job), style = Type.Caption)
            }
        }
        // A job being made cannot be deleted from under the service; one waiting its turn can.
        if (job.status != "running" && progress == null) {
            var menu by remember { mutableStateOf(false) }
            Box {
                IconTap("Options for ${job.title}", { menu = true }) { Icons.more(this, Ink.Muted) }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(text = { Text("Delete", color = Ink.Red) }, onClick = { menu = false; onDelete() })
                }
            }
        }
    }
}


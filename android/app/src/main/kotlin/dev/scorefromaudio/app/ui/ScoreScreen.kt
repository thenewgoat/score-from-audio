package dev.scorefromaudio.app.ui

import android.annotation.SuppressLint
import android.os.Handler
import android.os.Looper
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebViewAssetLoader
import dev.scorefromaudio.app.Exports
import dev.scorefromaudio.app.Format
import dev.scorefromaudio.app.JobMeta
import dev.scorefromaudio.app.JobStore
import dev.scorefromaudio.app.ScoreFiles
import dev.scorefromaudio.app.Step
import dev.scorefromaudio.pipeline.score.Basics
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonPrimitive
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

private val SPEEDS = listOf(0.5f, 0.75f, 0.9f, 1f, 1.25f)

/** A bar's times: written-tempo score ms and recording ms, as the viewer reports them on a tap. */
private data class BarTimes(val index: Int, val scoreStart: Double, val scoreEnd: Double, val audioStart: Double, val audioEnd: Double)

/**
 * The engraved score (Verovio, fitted to the phone's width, scrolling down) with playback: the original recording
 * synced to the score, or the score through the phone's synthesiser; a moving cursor; tap a bar to play from it;
 * loop bars; speed; share as PDF, MusicXML or MIDI; and "Fix the basics".
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun ScoreScreen(store: JobStore, initial: JobMeta, onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val jobDir = store.dir(initial.id)
    var job by remember { mutableStateOf(initial) }
    val main = remember { Handler(Looper.getMainLooper()) }
    val synth = remember { Player() }
    val recording = remember { Player() }
    val audioFile = job.audio?.let { File(jobDir, it) }?.takeIf { it.isFile }
    val sync = remember(job.bpm, job.meter) { runCatching { File(jobDir, "sync.json").readText() }.getOrDefault("[]") }
    val canSync = audioFile != null && sync.length > 2
    var useRecording by remember { mutableStateOf(canSync) }
    var playing by remember { mutableStateOf(false) }
    var position by remember { mutableIntStateOf(0) }
    var duration by remember { mutableIntStateOf(0) }
    var speed by remember { mutableFloatStateOf(1f) }
    var appliedSpeed by remember { mutableFloatStateOf(1f) }
    var rendered by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var bar by remember { mutableStateOf("") }
    var webView by remember { mutableStateOf<WebView?>(null) }
    var pendingResume by remember { mutableStateOf<Int?>(null) }
    var fixing by remember { mutableStateOf(false) }
    var redrawing by remember { mutableStateOf(false) }
    var sharing by remember { mutableStateOf(false) }
    var speedMenu by remember { mutableStateOf(false) }
    var preparing by remember { mutableStateOf<String?>(null) }
    // Loop: null when off; picking = 1 waits for the first bar, 2 for the last.
    var picking by remember { mutableIntStateOf(0) }
    var loopFrom by remember { mutableStateOf<BarTimes?>(null) }
    var loopTo by remember { mutableStateOf<BarTimes?>(null) }
    val pdfPages = remember { ArrayList<String>() }
    val disposed = remember { AtomicBoolean(false) }
    val midi = File(jobDir, "score.mid")
    fun current(): Player = if (useRecording) recording else synth

    fun js(script: String) = webView?.evaluateJavascript(script, null)
    fun loopStart(b: BarTimes) = if (useRecording) b.audioStart.toInt() else (b.scoreStart / appliedSpeed).toInt()
    fun loopEnd(b: BarTimes) = if (useRecording) b.audioEnd.toInt() else (b.scoreEnd / appliedSpeed).toInt()
    fun pause() { synth.pause(); recording.pause(); playing = false }
    fun showPosition() { if (useRecording) js("Viewer.atAudio($position)") else js("Viewer.at($position)") }

    DisposableEffect(Unit) {
        if (audioFile != null) runCatching { recording.load(audioFile, 0) }.onFailure { useRecording = false }
        onDispose {
            disposed.set(true)
            synth.release()
            recording.release()
        }
    }

    LaunchedEffect(playing, useRecording) {
        val player = current()
        while (playing) {
            if (pendingResume == null) {
                position = player.position
                val from = loopFrom
                val to = loopTo
                if (from != null && to != null && position >= loopEnd(to)) {
                    position = loopStart(from)
                    player.seekTo(position)
                }
                showPosition()
            }
            if (!player.playing && pendingResume == null) playing = false
            delay(40)
        }
    }

    val bridge = remember {
        object {
            @JavascriptInterface
            fun onViewerReady() = main.post { if (!disposed.get()) loadScore(webView, jobDir, sync, context) }

            @JavascriptInterface
            fun onMidi(base64: String, durationMs: Double) = main.post {
                if (disposed.get()) return@post
                midi.writeBytes(android.util.Base64.decode(base64, android.util.Base64.DEFAULT))
                val resume = pendingResume
                val start = resume ?: if (useRecording) 0 else position
                synth.load(midi, start)
                pendingResume = null
                if (!useRecording) {
                    position = start
                    duration = durationMs.toInt()
                    if (playing) synth.play()
                }
            }

            @JavascriptInterface
            fun onRendered(bars: Int) = main.post {
                if (disposed.get()) return@post
                rendered = true
                if (useRecording) duration = recording.duration
                showPosition()
            }

            @JavascriptInterface
            fun onBarShown(index: Int, label: String) = main.post { bar = label.ifEmpty { "${index + 1}" } }

            @JavascriptInterface
            fun onBar(index: Int, scoreStart: Double, scoreEnd: Double, audioStart: Double, audioEnd: Double) = main.post {
                if (disposed.get()) return@post
                val times = BarTimes(index, scoreStart, scoreEnd, audioStart, audioEnd)
                when (picking) {
                    1 -> { loopFrom = times; picking = 2; js("Viewer.setLoop($index, $index)") }
                    2 -> {
                        val from = loopFrom!!
                        if (index < from.index) { loopTo = from; loopFrom = times } else loopTo = times
                        picking = 0
                        js("Viewer.setLoop(${loopFrom!!.index}, ${loopTo!!.index})")
                        position = loopStart(loopFrom!!)
                        current().seekTo(position)
                        current().play(); playing = true
                    }
                    else -> {
                        position = loopStart(times)
                        current().seekTo(position)
                        showPosition()
                        current().play(); playing = true
                    }
                }
            }

            @JavascriptInterface
            fun onExport(kind: String, base64: String) = main.post {
                if (disposed.get()) return@post
                scope.launch {
                    val file = withContext(Dispatchers.IO) { Exports.midi(jobDir, job.title, base64) }
                    preparing = null
                    Exports.share(context, file, Exports.Kind.MIDI)
                }
            }

            @JavascriptInterface
            fun onPdfPage(page: Int, count: Int, base64: String) = main.post {
                if (disposed.get()) return@post
                if (page == 1) pdfPages.clear()
                pdfPages.add(base64)
                preparing = "Preparing the PDF… page $page of $count"
                if (page == count) scope.launch {
                    val pages = ArrayList(pdfPages)
                    pdfPages.clear()
                    val file = withContext(Dispatchers.IO) { Exports.pdf(jobDir, job.title, pages) }
                    preparing = null
                    Exports.share(context, file, Exports.Kind.PDF)
                }
            }

            @JavascriptInterface
            fun onError(message: String) = main.post { if (!disposed.get()) { error = message; preparing = null } }
        }
    }

    fun switchTo(toRecording: Boolean) {
        if (toRecording == useRecording) return
        val wasPlaying = playing
        pause()
        val here = position
        // Ask the viewer where this place is on the other clock, then carry on from there.
        webView?.evaluateJavascript(if (toRecording) "Viewer.midiToAudio($here)" else "Viewer.audioToMidi($here)") { result ->
            val there = result?.toDoubleOrNull()?.toInt() ?: 0
            useRecording = toRecording
            val next = if (toRecording) recording else synth
            duration = next.duration
            next.seekTo(there)
            position = there
            if (toRecording) recording.setSpeed(speed)
            showPosition()
            if (wasPlaying) { next.play(); playing = true }
        }
    }

    fun setSpeed(s: Float) {
        speed = s
        recording.setSpeed(s)
        // The MIDI is rendered again at the new speed; onMidi reloads the synth at `pendingResume`, keeping the place.
        if (s != appliedSpeed) {
            val base = pendingResume ?: synth.position
            pendingResume = (base * appliedSpeed / s).toInt()
            if (!useRecording) position = pendingResume!!
            appliedSpeed = s
            js("Viewer.setSpeed($s)")
        }
    }

    Column(Modifier.fillMaxSize().background(Ink.Paper)) {
        Header(job.title, onBack, size = 20) {
            OutlinePill("Fix basics", { pause(); fixing = true }, height = 40.dp,
                icon = { Canvas(Modifier.size(18.dp)) { Icons.sliders(this, Ink.Ink) } })
            Box {
                IconTap("Share or export", { sharing = true }) { Icons.share(this, Ink.Ink) }
                DropdownMenu(expanded = sharing, onDismissRequest = { sharing = false }) {
                    for (kind in Exports.Kind.entries) DropdownMenuItem(text = { Text(kind.label) }, onClick = {
                        sharing = false
                        when (kind) {
                            Exports.Kind.MUSICXML -> scope.launch {
                                val file = withContext(Dispatchers.IO) { Exports.musicXml(jobDir, job.title) }
                                Exports.share(context, file, kind)
                            }
                            Exports.Kind.MIDI -> { preparing = "Preparing the MIDI…"; js("Viewer.exportMidi()") }
                            Exports.Kind.PDF -> { preparing = "Preparing the PDF…"; js("Viewer.exportPdf()") }
                        }
                    })
                }
            }
        }
        Column(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp).clip(RoundedCornerShape(14.dp))
            .background(Ink.Card).border(1.dp, Ink.Line, RoundedCornerShape(14.dp))) {
            Text(Format.basics(job.meter, job.fifths, job.bpm), style = Type.Caption.copy(fontSize = 12.sp),
                modifier = Modifier.padding(start = 14.dp, end = 14.dp, top = 12.dp, bottom = 4.dp))
            val note = error ?: preparing ?: when (picking) { 1 -> "Tap the first bar to loop"; 2 -> "Now tap the last bar"; else -> null }
            note?.let { Text(it, fontSize = 13.sp, color = if (error != null) Ink.Red else Ink.Blue, fontWeight = FontWeight.Medium,
                modifier = Modifier.padding(horizontal = 14.dp)) }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { ctx ->
                        val loader = WebViewAssetLoader.Builder()
                            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(ctx)).build()
                        WebView(ctx).apply {
                            settings.javaScriptEnabled = true
                            setBackgroundColor(android.graphics.Color.WHITE)
                            webViewClient = object : WebViewClient() {
                                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                                    loader.shouldInterceptRequest(request.url)
                            }
                            addJavascriptInterface(bridge, "Android")
                            loadUrl("https://appassets.androidplatform.net/assets/viewer/index.html")
                            webView = this
                        }
                    },
                    onRelease = { it.destroy() },
                )
                if (!rendered || redrawing) Column(Modifier.fillMaxSize().background(Ink.Card), verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator(color = Ink.Blue)
                    Text(if (redrawing) "Redrawing the score…" else "${Step.LAYOUT.label}…", style = Type.Small,
                        modifier = Modifier.padding(top = 12.dp))
                }
            }
        }
        Column(Modifier.padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 24.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (canSync) Segmented(useRecording, ::switchTo)
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Box(Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)).background(Ink.Track)) {
                    Box(Modifier.fillMaxWidth((position.toFloat() / maxOf(duration, 1)).coerceIn(0f, 1f)).height(6.dp)
                        .clip(RoundedCornerShape(3.dp)).background(Ink.Blue))
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    val shown = if (useRecording || appliedSpeed == 1f) position else (position * appliedSpeed).toInt()
                    val total = if (useRecording || appliedSpeed == 1f) duration else (duration * appliedSpeed).toInt()
                    Text(Format.clock(shown.toLong()) + if (bar.isNotEmpty()) " · bar $bar" else "", style = Type.Caption.copy(fontSize = 12.sp))
                    Text(Format.clock(total.toLong()), style = Type.Caption.copy(fontSize = 12.sp))
                }
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                val looping = loopFrom != null && loopTo != null
                OutlinePill(when { looping -> "Loop on"; picking > 0 -> "Cancel loop"; else -> "Loop bars" }, {
                    if (looping || picking > 0) { loopFrom = null; loopTo = null; picking = 0; js("Viewer.setLoop(-1, -1)") }
                    else { pause(); picking = 1 }
                }, height = 44.dp, icon = { Canvas(Modifier.size(18.dp)) { Icons.loop(this, if (looping) Ink.Blue else Ink.Ink) } })
                Box(Modifier.size(68.dp).clip(CircleShape).background(if (rendered) Ink.Blue else Ink.Track)
                    .clickable(enabled = rendered && current().loaded, role = Role.Button) {
                        if (playing) pause() else {
                            val player = current()
                            loopFrom?.let { from -> loopTo?.let { to -> if (position < loopStart(from) || position >= loopEnd(to)) player.seekTo(loopStart(from)) } }
                            player.play(); playing = true
                        }
                    }.semantics { contentDescription = if (playing) "Pause" else "Play" }, contentAlignment = Alignment.Center) {
                    Canvas(Modifier.size(28.dp)) { if (playing) Icons.pause(this, Color.White) else Icons.play(this, Color.White) }
                }
                Box {
                    OutlinePill("Speed ${(speed * 100).toInt()}%", { speedMenu = true }, height = 44.dp)
                    DropdownMenu(expanded = speedMenu, onDismissRequest = { speedMenu = false }) {
                        for (s in SPEEDS) DropdownMenuItem(text = { Text("${(s * 100).toInt()}%") }, onClick = { speedMenu = false; setSpeed(s) })
                    }
                }
            }
        }
    }

    if (fixing) FixSheet(job.meter, job.fifths, job.bpm, redrawing, onDismiss = { if (!redrawing) fixing = false }) { basics: Basics ->
        redrawing = true
        scope.launch {
            val result = runCatching { withContext(Dispatchers.Default) { ScoreFiles.rewrite(store, job.id, basics) } }
            result.onSuccess { meta ->
                job = meta
                loopFrom = null; loopTo = null; picking = 0
                rendered = false
                pendingResume = if (useRecording) null else 0
                position = if (useRecording) position else 0
                loadScore(webView, jobDir, File(jobDir, "sync.json").readText(), context)
            }.onFailure { Toast.makeText(context, "The score could not be redrawn", Toast.LENGTH_LONG).show() }
            redrawing = false
            fixing = false
        }
    }
}

private fun loadScore(webView: WebView?, jobDir: File, sync: String, context: android.content.Context) {
    val view = webView ?: return
    val xml = File(jobDir, "score.musicxml").readText()
    val width = (view.width / context.resources.displayMetrics.density).toInt().takeIf { it > 0 } ?: 360
    view.evaluateJavascript("Viewer.load(${JsonPrimitive(xml)}, $width, $sync)", null)
}

/** "My recording" / "Piano sound". */
@Composable
private fun Segmented(recording: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(Ink.Sunken).padding(4.dp)) {
        for ((label, value) in listOf("My recording" to true, "Piano sound" to false)) {
            val on = value == recording
            Box(Modifier.weight(1f).height(40.dp).clip(RoundedCornerShape(20.dp)).background(if (on) Ink.Card else Color.Transparent)
                .clickable(role = Role.Tab) { onChange(value) }, contentAlignment = Alignment.Center) {
                Text(label, fontSize = 14.sp, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal, color = if (on) Ink.Ink else Color(0xFF3F3B35))
            }
        }
    }
}

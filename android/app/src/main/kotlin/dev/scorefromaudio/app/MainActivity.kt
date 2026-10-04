package dev.scorefromaudio.app

import android.Manifest
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.currentStateAsState
import androidx.lifecycle.lifecycleScope
import dev.scorefromaudio.app.ui.Ink
import dev.scorefromaudio.app.ui.LibraryScreen
import dev.scorefromaudio.app.ui.ProcessingScreen
import dev.scorefromaudio.app.ui.RecordScreen
import dev.scorefromaudio.app.ui.ScoreScreen
import dev.scorefromaudio.app.ui.ScoresTheme
import dev.scorefromaudio.app.ui.SettingsScreen
import dev.scorefromaudio.app.ui.TermsGate
import dev.scorefromaudio.app.ui.Consent
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

/**
 * One activity, five places: the library, recording, a score being made, a score, and settings. Where to go is a
 * string ("library", "record", "processing:<id>", "score:<id>", "settings") so it survives rotation; "processing:"
 * with no id follows whatever job is running (a link, whose job is made by the service).
 */
class MainActivity : ComponentActivity() {
    private val store by lazy { JobStore.forApp(this) }
    private val navigate = MutableStateFlow<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.light(Ink.Paper.toArgb(), Ink.Paper.toArgb()), SystemBarStyle.light(Ink.Paper.toArgb(), Ink.Paper.toArgb()))
        super.onCreate(savedInstanceState)
        if (Jobs.progress.value == null) {
            store.markInterrupted()
            store.sweepScratch()
            if (store.nextQueued() != null) runCatching { TranscriptionService.resume(this) }
        }
        // A recreated activity (rotation, process restore) still holds the launch intent: it was handled already.
        if (savedInstanceState == null) handle(intent)
        setContent {
            ScoresTheme {
                Box(Modifier.fillMaxSize().background(Ink.Paper).safeDrawingPadding()) { App() }
            }
        }
    }

    @Composable
    private fun App() {
        val progress by Jobs.progress.collectAsState()
        val changed by Jobs.changed.collectAsState()
        var screen by rememberSaveable { mutableStateOf("library") }
        var pasting by rememberSaveable { mutableStateOf(false) }
        var agreed by remember { mutableStateOf(Consent.accepted(this@MainActivity)) }
        val scope = rememberCoroutineScope()
        val requested by navigate.collectAsState()
        LaunchedEffect(requested) { requested?.let { screen = it; navigate.value = null } }
        val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) scope.launch { Imports.start(this@MainActivity, uri)?.let { screen = "processing:$it" } }
        }
        val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
        LaunchedEffect(agreed) {
            if (agreed && Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this@MainActivity,
                    Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                permission.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        // Nothing else is reachable until the current terms are agreed to.
        if (!agreed) {
            TermsGate { Consent.accept(this@MainActivity); agreed = true; handle(intent) }
            return
        }
        // The dialog stays open, with its text, until a job starts: a refused link can be corrected.
        if (pasting) PasteLinkDialog(onDismiss = { pasting = false }) { text ->
            if (Jobs.progress.value != null) busy()
            else if (Imports.startLink(this@MainActivity, text)) { pasting = false; screen = "processing:" }
        }
        val home = { screen = "library" }
        val (place, id) = screen.split(":", limit = 2).let { it[0] to it.getOrNull(1).orEmpty() }
        if (place != "library") BackHandler { home() }
        when (place) {
            "settings" -> SettingsScreen(home)
            "record" -> RecordScreen(onBack = home) { file, hint ->
                scope.launch {
                    val job = Imports.recording(this@MainActivity, file, Format.recordingTitle(System.currentTimeMillis()), hint)
                    screen = if (job != null) "processing:$job" else "library"
                }
            }
            "processing" -> {
                // Follow the running job until it has an id, then stay with that job.
                val followed = id.ifEmpty { progress?.jobId.orEmpty() }
                LaunchedEffect(followed) { if (id.isEmpty() && followed.isNotEmpty()) screen = "processing:$followed" }
                val job = remember(followed, changed, progress?.stage) { followed.takeIf { it.isNotEmpty() }?.let { store.load(it) } }
                Watching(job?.id)
                // A link stopped before its audio arrived leaves no job behind: nothing more to show here.
                LaunchedEffect(job == null, progress == null) { if (id.isNotEmpty() && job == null && progress == null) home() }
                LaunchedEffect(job?.status) {
                    if (job?.status == "done") { delay(700); screen = "score:${job.id}" }
                }
                ProcessingScreen(job, progress, onBack = home,
                    onStop = { TranscriptionService.cancel(this@MainActivity) },
                    onRetry = { job?.let { TranscriptionService.retry(this@MainActivity, it.id) } },
                    onDelete = { job?.let { store.delete(it.id); Jobs.changed.value++ }; home() })
            }
            "score" -> {
                val job = remember(id) { store.load(id) }
                LaunchedEffect(id) { TranscriptionService.dismiss(this@MainActivity, id) }
                if (job?.status == "done") ScoreScreen(store, job, home) else LaunchedEffect(id) { screen = "processing:$id" }
            }
            else -> {
                val jobs = remember(changed, progress?.jobId, progress?.stage) { store.list() }
                LibraryScreen(
                    jobs = jobs, progress = progress,
                    onOpen = { screen = if (it.status == "done") "score:${it.id}" else "processing:${it.id}" },
                    onDelete = {
                        store.delete(it.id)
                        TranscriptionService.dismiss(this@MainActivity, it.id)
                        Jobs.changed.value++
                    },
                    onRecord = { screen = "record" },
                    onImport = { picker.launch(arrayOf("audio/*", "video/*")) },
                    onPasteLink = { pasting = true },
                    onSettings = { screen = "settings" },
                )
            }
        }
    }

    /** Marks [id] as on screen while the app is in front, so its "ready" notification is not sent. */
    @Composable
    private fun Watching(id: String?) {
        val resumed = LocalLifecycleOwner.current.lifecycle.currentStateAsState().value.isAtLeast(Lifecycle.State.RESUMED)
        DisposableEffect(id, resumed) {
            Jobs.watching.value = if (resumed) id else null
            onDispose { Jobs.watching.value = null }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handle(intent)
    }

    /** A tapped notification opens its job; a share starts one. Each is handled once, then the intent is replaced. */
    private fun handle(intent: Intent?) {
        intent?.getStringExtra(EXTRA_OPEN_JOB)?.let { id ->
            setIntent(Intent(this, MainActivity::class.java))
            navigate.value = if (store.load(id)?.status == "done") "score:$id" else "processing:$id"
            return
        }
        if (intent?.action != Intent.ACTION_SEND) return
        // A share that arrives before the terms are agreed to waits in the activity's intent until they are.
        if (!Consent.accepted(this)) return
        setIntent(Intent(this, MainActivity::class.java))
        if (intent.type == "text/plain") {
            val text = intent.getStringExtra(Intent.EXTRA_TEXT)?.takeIf { it.isNotBlank() } ?: return
            if (Jobs.progress.value != null) busy() else if (Imports.startLink(this, text)) navigate.value = "processing:"
            return
        }
        val uri = IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java) ?: return
        lifecycleScope.launch { Imports.start(this@MainActivity, uri)?.let { navigate.value = "processing:$it" } }
    }

    private fun busy() = Toast.makeText(this, "Another score is being made. Share the link again when it is done.", Toast.LENGTH_LONG).show()

    companion object {
        const val EXTRA_OPEN_JOB = "open_job"
    }
}

/** A link typed or pasted in, pre-filled from the clipboard (read once, when the dialog opens) if it holds one. */
@Composable
private fun PasteLinkDialog(onDismiss: () -> Unit, onTranscribe: (String) -> Unit) {
    val context = LocalContext.current
    var text by remember {
        val clip = context.getSystemService(ClipboardManager::class.java)?.primaryClip
            ?.takeIf { it.itemCount > 0 }?.getItemAt(0)?.coerceToText(context)?.toString()
        mutableStateOf(clip?.takeIf { "http" in it } ?: "")
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Paste a link") },
        text = {
            OutlinedTextField(value = text, onValueChange = { text = it }, singleLine = true,
                label = { Text("YouTube or Instagram link") })
        },
        confirmButton = { TextButton(onClick = { onTranscribe(text) }) { Text("Make score") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

private fun androidx.compose.ui.graphics.Color.toArgb(): Int = android.graphics.Color.argb(
    (alpha * 255).toInt(), (red * 255).toInt(), (green * 255).toInt(), (blue * 255).toInt())

package dev.scorefromaudio.app

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat
import dev.scorefromaudio.pipeline.Links
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.concurrent.CancellationException

/**
 * Starting a score from a recording, a file or a link. Recordings and files become queued jobs holding their audio,
 * so the returned job id can be shown at once, and the service is asked to run what is queued.
 */
object Imports {
    /** Queues a finished recording; the job's id, or null (with a toast) if it could not be started. */
    suspend fun recording(context: Context, file: File, title: String, hint: String?): String? = queue(context) {
        withContext(Dispatchers.IO) { JobStore.forApp(context).enqueue(file, title, hint).id }
    }

    /**
     * Queues a file from a content URI. The file is copied into the app first: the permission to read a shared URI
     * belongs to the activity that received it and may not outlive it, and the job runs for minutes. Anything that
     * stops the import is shown as a toast and leaves no copy behind.
     */
    suspend fun start(context: Context, uri: Uri): String? {
        var copy: File? = null
        val problem = try {
            val name = withContext(Dispatchers.IO) {
                context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                    if (it.moveToFirst()) it.getString(0) else null
                }
            } ?: "Recording"
            val file = File(context.cacheDir, "input-${System.currentTimeMillis()}-${name.takeLast(40).replace('/', '_')}")
            copy = file
            withContext(Dispatchers.IO) {
                val input = context.contentResolver.openInputStream(uri) ?: throw IOException("no stream")
                input.use { i -> file.outputStream().use { i.copyTo(it, 1 shl 20) } }
            }
            return queue(context) { withContext(Dispatchers.IO) { JobStore.forApp(context).enqueue(file, name.substringBeforeLast('.')).id } }
        } catch (e: CancellationException) {
            copy?.delete()
            throw e
        } catch (e: SecurityException) {
            "This app is not allowed to read that file"
        } catch (e: IOException) {
            "This file could not be read"
        } catch (e: IllegalArgumentException) {
            // A URI no provider will open.
            "This file could not be read"
        }
        Log.w("Imports", "Import of $uri failed: $problem")
        copy?.delete()
        withContext(Dispatchers.Main) { Toast.makeText(context, problem, Toast.LENGTH_LONG).show() }
        return null
    }

    private suspend fun queue(context: Context, enqueue: suspend () -> String): String? {
        val id = try { enqueue() } catch (e: IOException) {
            withContext(Dispatchers.Main) { Toast.makeText(context, "There is no room on the phone to keep this recording", Toast.LENGTH_LONG).show() }
            return null
        }
        Jobs.changed.update { it + 1 }
        try {
            TranscriptionService.resume(context)
        } catch (e: IllegalStateException) {
            // ForegroundServiceStartNotAllowedException is one: the app was not in the foreground. The job stays
            // queued and runs the next time the app opens.
            Log.w("Imports", "Could not start the service; $id stays queued", e)
        }
        return id
    }

    /**
     * Start a transcription from shared or pasted text holding a YouTube or Instagram link. Call it on the main
     * thread: it shows its problems as toasts. True when the transcription was started.
     */
    fun startLink(context: Context, text: String): Boolean {
        val url = Links.extractUrl(text)
        val problem = when {
            url == null || Links.site(url) == null -> Links.UNSUPPORTED
            else -> try {
                ContextCompat.startForegroundService(context, Intent(context, TranscriptionService::class.java)
                    .setAction(TranscriptionService.ACTION_START).putExtra(TranscriptionService.EXTRA_URL, url))
                null
            } catch (e: IllegalStateException) { "The transcription could not be started" }
        }
        if (problem != null) Toast.makeText(context, problem, Toast.LENGTH_LONG).show()
        return problem == null
    }
}

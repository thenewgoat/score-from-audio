package dev.scorefromaudio.app

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import dev.scorefromaudio.pipeline.Links
import dev.scorefromaudio.pipeline.Site
import java.io.File
import java.util.concurrent.CancellationException
import org.json.JSONObject

class Fetched(val file: File, val title: String?)

/** A fetch that failed, with the message to show the person. */
class FetchFailed(message: String) : Exception(message)

/** Downloads a link's audio into [outDir]. Throws [FetchFailed] or java.util.concurrent.CancellationException. */
interface Fetcher {
    fun fetch(url: String, site: Site, outDir: File, progress: (Float) -> Unit, cancelled: () -> Boolean): Fetched
}

sealed interface Source {
    data class Picked(val file: File, val title: String) : Source
    data class Link(val url: String, val site: Site) : Source
}

/**
 * yt-dlp inside the app, through Chaquopy. YouTube is tried without a JavaScript
 * runtime first (faster, and works while YouTube still serves a client that needs
 * none), then once more with QuickJS if the failure looks like a challenge.
 */
class PythonFetcher(private val context: Context) : Fetcher {
    fun interface ProgressSink { fun onProgress(done: Long, total: Long) }
    fun interface CancelFlag { fun get(): Boolean }

    companion object {
        private val startLock = Any()
        // At most one progress update per interval, so a busy download doesn't flood the UI; the final 1.0 always passes.
        private const val PROGRESS_INTERVAL_MS = 200L
    }

    override fun fetch(url: String, site: Site, outDir: File, progress: (Float) -> Unit, cancelled: () -> Boolean): Fetched {
        if (!Python.isStarted()) synchronized(startLock) { if (!Python.isStarted()) Python.start(AndroidPlatform(context)) }
        val module = Python.getInstance().getModule("fetcher")
        val qjs = File(context.applicationInfo.nativeLibraryDir, "libqjs.so").takeIf { it.isFile }?.path
        val cookies = if (site == Site.INSTAGRAM && InstagramLogin.isLoggedIn(context)) InstagramLogin.cookieFile(context).path else null
        var lastTime = 0L
        val sink = ProgressSink { done, total ->
            if (total > 0) {
                val fraction = (done.toDouble() / total).toFloat().coerceIn(0f, 1f)
                val now = System.currentTimeMillis()
                if (fraction >= 1f || now - lastTime >= PROGRESS_INTERVAL_MS) {
                    lastTime = now
                    progress(fraction)
                }
            }
        }
        val flag = CancelFlag { cancelled() }

        fun attempt(useJs: Boolean): JSONObject =
            JSONObject(module.callAttr("fetch", url, outDir.path, cookies, qjs, useJs, sink, flag).toString())

        var result = attempt(useJs = false)
        if (!result.optBoolean("ok") && !result.optBoolean("cancelled") && !cancelled() && site == Site.YOUTUBE && qjs != null &&
            Links.isRetryableWithJs(result.optString("error"))) {
            result = attempt(useJs = true)
        }
        if (result.optBoolean("cancelled") || cancelled()) throw CancellationException("cancelled")
        if (!result.optBoolean("ok")) throw FetchFailed(Links.errorMessage(site, result.optString("error")))
        if (result.isNull("path")) throw FetchFailed(Links.errorMessage(site, "no file"))
        val file = File(result.optString("path"))
        if (!file.isFile) throw FetchFailed(Links.errorMessage(site, "no file"))
        return Fetched(file, result.optString("title").takeIf { it.isNotBlank() && it != "null" })
    }
}

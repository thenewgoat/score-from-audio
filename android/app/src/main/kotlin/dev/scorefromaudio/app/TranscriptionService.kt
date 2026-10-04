package dev.scorefromaudio.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Debug
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import dev.scorefromaudio.pipeline.Links
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.CancellationException
import kotlin.coroutines.coroutineContext

/**
 * Runs transcriptions one at a time in the foreground, so leaving the app or locking the phone does not stop them.
 * Recordings and files wait their turn as queued jobs in the store (the UI enqueues them, then calls [resume]), so
 * one that arrives while another runs is never lost; a link is fetched only when nothing is running. Each job that ends posts "Your score is ready" (or why it
 * is not), unless its progress was on screen.
 */
class TranscriptionService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    // Both touched only on the main thread: onStartCommand runs there, and so does a job's finish().
    private var running: Job? = null
    private var lastStartId = 0
    private val store by lazy { JobStore.forApp(this) }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        lastStartId = startId
        when (intent?.action) {
            ACTION_CANCEL -> if (running != null) running?.cancel() else stopSelf(startId) // nothing to cancel: do not linger
            ACTION_START, ACTION_RETRY, ACTION_NEXT -> {
                // Android punishes a start that does not call startForeground: do it first, whatever follows.
                startForeground(NOTIFICATION, progressNotification("Starting", 0f), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
                accept(intent)
                if (running == null) next()
            }
            else -> if (running == null) stopSelf(startId)
        }
        return START_NOT_STICKY
    }

    /** A retry is queued again; a link runs only when nothing else is running (the UI says so when busy). */
    private fun accept(intent: Intent) {
        intent.getStringExtra(EXTRA_JOB)?.let { id ->
            store.load(id)?.takeIf { it.status == "failed" }?.let { it.status = "queued"; it.error = null; store.save(it) }
            Jobs.changed.update { it + 1 }
        }
        val url = intent.getStringExtra(EXTRA_URL)
        val site = url?.let { Links.site(it) }
        if (url != null && site != null && running == null) launch { r, cancelled, progress -> r.run(Source.Link(url, site), cancelled, progress) }
    }

    /** Starts the oldest queued job, or stops the service when there is none. */
    private fun next() {
        val queued = store.nextQueued()
        if (queued == null) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf(lastStartId)
            return
        }
        launch { r, cancelled, progress -> r.run(queued.id, cancelled, progress) }
    }

    private fun runner() = JobRunner(store, MediaDecoder::decode, OrtEngines { ModelStore.ensure(this) },
        fetcher = PythonFetcher(this)) { memoryMb() }

    private fun launch(work: (JobRunner, () -> Boolean, (Progress) -> Unit) -> JobMeta?) {
        // Set synchronously, before the coroutine below reaches the runner's first progress callback: onStartCommand
        // runs on the main thread, so any activity recreated after this line sees a job in progress and does not
        // call markInterrupted() on it. The empty jobId is replaced by the runner's own first callback.
        Jobs.progress.value = Progress("", Stage.DECODING, 0f)
        running = scope.launch {
            val job = coroutineContext[Job]!!
            var result: JobMeta? = null
            try {
                result = work(runner(), { job.isCancelled }) { p ->
                    Jobs.progress.value = p
                    getSystemService(NotificationManager::class.java).notify(NOTIFICATION, progressNotification(Steps.of(p.stage).label, p.fraction))
                }
            } catch (e: CancellationException) {
                // The runner records a stop itself; nothing more to do.
            } catch (e: Throwable) {
                // Last resort: the runner records every failure it can see; anything escaping it must not kill the process.
                Log.e(TAG, "Transcription failed outside the job runner", e)
            } finally {
                withContext(NonCancellable + Dispatchers.Main) { finish(result) }
            }
        }
    }

    /**
     * On the main thread, so no start command can land halfway through. `running` is cleared first, so a start
     * command that arrives after this is run rather than refused as busy. The UI learns last that the job is over.
     */
    private fun finish(result: JobMeta?) {
        running = null
        if (result != null && result.error != JobRunner.STOPPED && Jobs.watching.value != result.id) announce(this, result)
        Jobs.progress.value = null
        Jobs.changed.update { it + 1 }
        next()
    }

    /** Resident memory, which includes the runtime's native buffers; the older heap counters where /proc cannot be read. */
    private fun memoryMb(): Long =
        runCatching { Memory.residentMb(File("/proc/self/status").readText()) }.getOrNull()
            ?: ((Debug.getNativeHeapAllocatedSize() + Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() }) / (1 shl 20))

    private fun progressNotification(text: String, fraction: Float): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "Making scores", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentTitle("Making your score")
            .setContentText(text)
            .setProgress(100, (fraction * 100).toInt(), false)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "dev.scorefromaudio.app.START"
        const val ACTION_RETRY = "dev.scorefromaudio.app.RETRY"
        const val ACTION_NEXT = "dev.scorefromaudio.app.NEXT"
        const val ACTION_CANCEL = "dev.scorefromaudio.app.CANCEL"
        const val EXTRA_URL = "url"
        const val EXTRA_JOB = "job"
        private const val TAG = "TranscriptionService"
        private const val CHANNEL = "transcription"
        private const val RESULTS = "results"
        private const val NOTIFICATION = 1

        /** Runs a failed or stopped job again from its kept recording (or its link). */
        fun retry(context: Context, id: String) = ContextCompat.startForegroundService(context,
            Intent(context, TranscriptionService::class.java).setAction(ACTION_RETRY).putExtra(EXTRA_JOB, id))

        /** Picks up jobs left queued when the app was last closed. */
        fun resume(context: Context) = ContextCompat.startForegroundService(context,
            Intent(context, TranscriptionService::class.java).setAction(ACTION_NEXT))

        fun cancel(context: Context) =
            context.startService(Intent(context, TranscriptionService::class.java).setAction(ACTION_CANCEL))

        /** "Your score is ready", opening the score; or that it could not be made, opening the job to try again. */
        fun announce(context: Context, meta: JobMeta) {
            val manager = context.getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(NotificationChannel(RESULTS, "Finished scores", NotificationManager.IMPORTANCE_DEFAULT))
            val open = PendingIntent.getActivity(context, meta.id.hashCode(),
                Intent(context, MainActivity::class.java).putExtra(MainActivity.EXTRA_OPEN_JOB, meta.id),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
            val done = meta.status == "done"
            val notification = Notification.Builder(context, RESULTS)
                .setSmallIcon(android.R.drawable.ic_media_play)
                .setContentTitle(if (done) "Your score is ready" else "Your score couldn't be made")
                .setContentText(if (done) meta.title else Reasons.plain(meta))
                .setContentIntent(open)
                .setAutoCancel(true)
                .build()
            manager.notify(meta.id, NOTIFICATION, notification)
        }

        fun dismiss(context: Context, id: String) =
            context.getSystemService(NotificationManager::class.java).cancel(id, NOTIFICATION)
    }
}

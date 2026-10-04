package dev.scorefromaudio.app

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** What the Record screen shows. [file] is the recording once it has been stopped. */
data class RecordingState(
    val recording: Boolean,
    val elapsedMs: Long,
    val bars: List<Float>,
    val hint: LevelHint,
    val file: File?,
    val error: String? = null,
)

/**
 * The microphone, recording to a 16-bit mono WAV at 44.1 kHz. One recording at a time, owned by this object so it
 * survives the Record screen being rebuilt; [RecordingService] keeps the app in the foreground meanwhile, so locking
 * the phone does not silence the microphone.
 */
object Recorder {
    private const val RATE = 44_100
    val state = MutableStateFlow<RecordingState?>(null)
    @Volatile private var thread: Thread? = null
    @Volatile private var stopping = false

    fun folder(context: Context) = File(context.filesDir, "recordings").apply { mkdirs() }

    @SuppressLint("MissingPermission") // The Record screen asks for RECORD_AUDIO before it calls this.
    fun start(context: Context) {
        if (thread != null) return
        folder(context).listFiles()?.forEach { it.delete() } // anything a killed process left behind
        val file = File(folder(context), "recording-${System.currentTimeMillis()}.wav")
        val minimum = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val record = try {
            AudioRecord(source(context), RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minimum, RATE))
                .takeIf { it.state == AudioRecord.STATE_INITIALIZED } ?: error("not initialised")
        } catch (e: Exception) {
            Log.w("Recorder", "The microphone could not be opened", e)
            state.value = RecordingState(false, 0, emptyList(), LevelHint.LISTENING, null, "The microphone could not be opened")
            return
        }
        stopping = false
        state.value = RecordingState(true, 0, emptyList(), LevelHint.LISTENING, null)
        context.startForegroundService(Intent(context, RecordingService::class.java))
        thread = Thread({ capture(record, file) }, "recorder").apply { start() }
    }

    /** Stops and finishes the file; blocks until it is written. */
    fun stop(context: Context) {
        val t = thread ?: return
        stopping = true
        t.join()
        thread = null
        context.stopService(Intent(context, RecordingService::class.java))
    }

    /** Deletes the recording, stopping it first. */
    fun discard(context: Context) {
        stop(context)
        state.value?.file?.delete()
        state.value = null
    }

    /** Hands the recording over to be made into a score: the file is no longer the recorder's to delete. */
    fun release(): File? = state.value?.file.also { state.value = null }

    /** The source with the least processing: unprocessed where the phone offers it, else voice recognition (no AGC). */
    private fun source(context: Context): Int {
        val audio = context.getSystemService(AudioManager::class.java)
        return if (audio.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true")
            MediaRecorder.AudioSource.UNPROCESSED else MediaRecorder.AudioSource.VOICE_RECOGNITION
    }

    private fun capture(record: AudioRecord, file: File) {
        val levels = Levels(RATE)
        val block = ShortArray(levels.blockSize)
        val bytes = ByteBuffer.allocate(block.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        var frames = 0L
        var error: String? = null
        RandomAccessFile(file, "rw").use { out ->
            out.setLength(0)
            out.write(ByteArray(44))
            try {
                record.startRecording()
                while (!stopping) {
                    var filled = 0
                    while (filled < block.size && !stopping) {
                        val n = record.read(block, filled, block.size - filled)
                        if (n < 0) error("read failed ($n)")
                        filled += n
                    }
                    if (filled == 0) continue
                    bytes.clear()
                    for (i in 0 until filled) bytes.putShort(block[i])
                    out.write(bytes.array(), 0, filled * 2)
                    frames += filled
                    if (filled == block.size) levels.add(block)
                    state.value = RecordingState(true, frames * 1000 / RATE, levels.bars(), levels.hint(), null)
                }
            } catch (e: Exception) {
                Log.w("Recorder", "Recording stopped by an error", e)
                error = "Recording stopped: the microphone stopped sending sound"
            } finally {
                runCatching { record.stop() }
                record.release()
                out.seek(0)
                out.write(header(frames))
            }
        }
        state.value = RecordingState(false, frames * 1000 / RATE, levels.bars(), levels.hint(), file, error)
        if (error != null) thread = null
    }

    private fun header(frames: Long): ByteArray {
        val data = (frames * 2).toInt()
        return ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + data); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(1); putInt(RATE); putInt(RATE * 2)
            putShort(2); putShort(16)
            put("data".toByteArray()); putInt(data)
        }.array()
    }
}

/** Keeps the app in the foreground while it records, as Android requires for the microphone in the background. */
class RecordingService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "Recording", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 1, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val notification = Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Recording")
            .setContentText("Tap to go back to the recording")
            .setContentIntent(open)
            .setOngoing(true)
            .build()
        startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        return START_NOT_STICKY
    }

    private companion object {
        const val CHANNEL = "recording"
        const val NOTIFICATION = 2 // apart from the transcription service's, which may be showing too
    }
}

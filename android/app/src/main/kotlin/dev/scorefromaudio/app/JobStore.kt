package dev.scorefromaudio.app

import android.content.Context
import dev.scorefromaudio.pipeline.score.Basics
import dev.scorefromaudio.pipeline.score.Repairs
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

/**
 * [status] is "queued", "running", "done" or "failed". [audio] names the recording kept in the job's folder, from
 * which a failed job can be tried again and "My recording" plays; [url] is the link it came from, if any.
 * [hint] is the meter picked while recording ("3/4"), [basics] what was set in "Fix the basics"; [meter], [fifths]
 * and [bpm] describe the score as written.
 */
@Serializable
data class JobMeta(
    val id: String,
    var title: String,
    val createdAt: Long,
    var status: String = "running",
    var stage: String = "",
    var error: String? = null,
    var seconds: Double = 0.0,
    var notes: Int = 0,
    var bars: Int = 0,
    var repairs: Repairs? = null,
    val timingsMs: MutableMap<String, Long> = mutableMapOf(),
    /** Stage name -> the most memory seen while it ran. A new name, so meta.json files holding the old single number still load. */
    val peakMemoryMbByStage: MutableMap<String, Long> = mutableMapOf(),
    var audio: String? = null,
    var url: String? = null,
    var hint: String? = null,
    var basics: Basics? = null,
    var meter: String? = null,
    var fifths: Int? = null,
    var bpm: Int? = null,
)

/**
 * One folder per transcription under `root`, holding meta.json and the files each stage writes. [scratch] is where a
 * fetch downloads to: outside [root], so a fetch in progress never shows up in list() or markInterrupted(). The app
 * puts it in the cache directory ([forApp]), so the system may reclaim what a killed fetch left behind.
 */
class JobStore(private val root: File, private val scratch: File = File(root.parentFile, "fetch")) {
    companion object {
        fun forApp(context: Context) = JobStore(File(context.filesDir, "jobs"), File(context.cacheDir, "fetch"))
    }

    private val json = Json { ignoreUnknownKeys = true; prettyPrint = true }

    fun create(title: String): JobMeta {
        val meta = fresh(title)
        dir(meta.id).mkdirs()
        save(meta)
        return meta
    }

    private fun fresh(title: String): JobMeta {
        val now = System.currentTimeMillis()
        return JobMeta("$now-${UUID.randomUUID().toString().take(8)}", title, now)
    }

    /**
     * A job waiting its turn, holding [file] (moved into its folder) and the meter [hint] picked while recording.
     * meta.json is written last, so a job never shows up queued without its recording.
     */
    fun enqueue(file: File, title: String, hint: String? = null): JobMeta {
        val meta = fresh(title)
        meta.status = "queued"
        meta.hint = hint
        meta.audio = keepAudio(meta.id, file)
        save(meta)
        return meta
    }

    fun dir(id: String) = File(root, id)

    fun scratchRoot(): File = scratch.apply { mkdirs() }

    /** Deletes what fetches left in the scratch area. Call only when no job is running, like [markInterrupted]. */
    fun sweepScratch() {
        scratch.listFiles()?.forEach { it.deleteRecursively() }
    }

    fun save(meta: JobMeta) = write(meta.id, "meta.json", json.encodeToString(JobMeta.serializer(), meta))

    /** Null when there is no meta.json, or it cannot be read (a file cut short by a crash before writes were atomic). */
    fun load(id: String): JobMeta? {
        val file = File(dir(id), "meta.json").takeIf { it.isFile } ?: return null
        return try {
            json.decodeFromString(JobMeta.serializer(), file.readText())
        } catch (e: IllegalArgumentException) { // SerializationException is one
            null
        } catch (e: IOException) {
            null
        }
    }

    fun list(): List<JobMeta> =
        (root.listFiles() ?: emptyArray()).mapNotNull { load(it.name) }.sortedByDescending { it.createdAt }

    /** Written beside the target and renamed over it, so a process killed mid-write never leaves a file cut short. */
    fun write(id: String, name: String, text: String) {
        dir(id).mkdirs()
        val temp = File(dir(id), "$name.tmp")
        temp.writeText(text)
        Files.move(temp.toPath(), File(dir(id), name).toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    }

    /** The oldest job waiting its turn, if any. */
    fun nextQueued(): JobMeta? = list().lastOrNull { it.status == "queued" }

    fun read(id: String, name: String): String = File(dir(id), name).readText()

    /**
     * Moves [file] into the job's folder as its recording (copying when a rename cannot cross file systems) and
     * returns the name it is kept under.
     */
    fun keepAudio(id: String, file: File): String {
        val name = "audio." + (file.extension.lowercase().takeIf { it.isNotEmpty() && it.length <= 5 } ?: "bin")
        val target = File(dir(id).apply { mkdirs() }, name)
        if (!file.renameTo(target)) {
            file.copyTo(target, overwrite = true)
            file.delete()
        }
        return name
    }

    fun delete(id: String) {
        dir(id).deleteRecursively()
    }

    /** Jobs a dead process left "running" become failed at the stage they reached. Call only when none is running. */
    fun markInterrupted() {
        for (meta in list()) if (meta.status == "running") {
            meta.status = "failed"
            meta.error = Reasons.INTERRUPTED
            save(meta)
        }
    }
}

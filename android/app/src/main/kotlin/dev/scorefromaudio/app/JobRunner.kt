package dev.scorefromaudio.app

import dev.scorefromaudio.pipeline.Audio
import dev.scorefromaudio.pipeline.Links
import dev.scorefromaudio.pipeline.Note
import dev.scorefromaudio.pipeline.NotesJson
import dev.scorefromaudio.pipeline.Resampler
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.CancellationException

class UnreadableAudio(message: String) : Exception(message)

private class Failure(val stage: Stage, message: String) : Exception(message)

/**
 * One transcription, start to finish: decode, resample, notes, notation, MusicXML.
 * Every stage's output is stored as it is produced, with its time and its peak
 * memory. The peak is sampled at every progress callback and meta.json is saved
 * every [saveEveryMs] as well, so a process killed for memory keeps what it saw.
 *
 * The recording is kept in the job's folder: "My recording" plays it, and a job that
 * failed or was stopped can be run again from it. A failure is recorded with its
 * stage; so is a stop, as "Stopped". Only a link stopped before its audio arrived
 * leaves nothing worth keeping, and is deleted.
 */
class JobRunner(
    private val store: JobStore,
    private val decode: (File, () -> Boolean) -> Audio,
    private val engines: Engines,
    private val fetcher: Fetcher? = null,
    private val saveEveryMs: Long = 2_000,
    private val memoryMb: () -> Long,
) {
    companion object {
        const val STOPPED = "Stopped"
    }

    /** Enqueues [input] and runs it straight away. */
    fun run(input: File, title: String, cancelled: () -> Boolean, progress: (Progress) -> Unit): JobMeta? =
        run(store.enqueue(input, title).id, cancelled, progress)

    fun run(source: Source, cancelled: () -> Boolean, progress: (Progress) -> Unit): JobMeta? = when (source) {
        is Source.Picked -> run(source.file, source.title, cancelled, progress)
        is Source.Link -> {
            val meta = store.create(Links.fallbackTitle(source.url, source.site))
            meta.url = source.url
            store.save(meta)
            run(meta.id, cancelled, progress)
        }
    }

    /** Runs a queued job, or again one that failed: from its kept recording, else by fetching its link. */
    fun run(id: String, cancelled: () -> Boolean, progress: (Progress) -> Unit): JobMeta? {
        val meta = store.load(id) ?: return null
        meta.status = "running"
        meta.error = null
        store.save(meta)
        val scratch = if (meta.audio == null) File(store.scratchRoot(), "fetch-${System.nanoTime()}") else null
        try {
            return run(meta, scratch, cancelled, progress)
        } finally {
            scratch?.deleteRecursively()
        }
    }

    private fun run(meta: JobMeta, scratch: File?, cancelled: () -> Boolean, progress: (Progress) -> Unit): JobMeta? {
        var stage = if (meta.audio == null) Stage.FETCHING else Stage.DECODING
        fun sample(s: Stage) {
            meta.peakMemoryMbByStage[s.name] = maxOf(meta.peakMemoryMbByStage[s.name] ?: 0, memoryMb())
        }
        fun <T> step(s: Stage, block: ((Float) -> Unit) -> T): T {
            stage = s
            meta.stage = s.name
            sample(s)
            store.save(meta)
            progress(Progress(meta.id, s, 0f))
            val started = System.currentTimeMillis()
            var saved = started
            val result = block { f ->
                sample(s)
                val now = System.currentTimeMillis()
                if (now - saved >= saveEveryMs) {
                    store.save(meta)
                    saved = now
                }
                progress(Progress(meta.id, s, f))
            }
            meta.timingsMs[s.name] = System.currentTimeMillis() - started
            sample(s)
            return result
        }
        try {
            if (meta.audio == null) {
                val url = checkNotNull(meta.url) { "This job has neither a recording nor a link" }
                val site = Links.site(url) ?: throw FetchFailed(Links.UNSUPPORTED)
                step(Stage.FETCHING) {
                    val f = checkNotNull(fetcher) { "no fetcher configured" }.fetch(url, site, scratch!!, it, cancelled)
                    f.title?.takeIf { t -> t.isNotBlank() }?.let { t -> meta.title = t }
                    meta.audio = store.keepAudio(meta.id, f.file)
                }
            }
            val input = File(store.dir(meta.id), meta.audio!!)
            val audio = step(Stage.DECODING) { Resampler.resample(decode(input, cancelled), 44_100) }
            meta.seconds = audio.seconds
            val notes = step(Stage.NOTES) { engines.notes(audio, it, cancelled) }
            store.write(meta.id, "notes.json", NotesJson.write(notes))
            meta.notes = notes.count { it.pitch >= 0 }
            if (meta.notes == 0) throw Failure(Stage.NOTES, "No notes were detected")
            val tokens = step(Stage.NOTATION) { engines.tokens(notes, it, cancelled) }
            store.write(meta.id, "tokens.json", Json.encodeToString(tokens.map { it.toList() }))
            step(Stage.WRITING) { ScoreFiles.write(store, meta, tokens, notes) }
            meta.status = "done"
        } catch (e: CancellationException) {
            if (meta.audio == null) {
                store.delete(meta.id)
                return null
            }
            fail(meta, stage, STOPPED)
        } catch (e: Failure) {
            fail(meta, e.stage, e.message)
        } catch (e: UnreadableAudio) {
            fail(meta, Stage.DECODING, e.message)
        } catch (e: FetchFailed) {
            fail(meta, Stage.FETCHING, e.message)
        } catch (e: OutOfMemoryError) {
            fail(meta, stage, "Ran out of memory while ${stage.label} (peak ${meta.peakMemoryMbByStage[stage.name] ?: 0} MB)")
        } catch (e: Exception) {
            fail(meta, stage, e.message ?: e.javaClass.simpleName)
        } catch (e: Throwable) {
            // Any other Error, e.g. an UnsatisfiedLinkError from a runtime whose native library will not load.
            fail(meta, stage, "${e.javaClass.simpleName}: ${e.message}")
        }
        store.save(meta)
        return meta
    }

    private fun fail(meta: JobMeta, stage: Stage, message: String?) {
        meta.status = "failed"
        meta.stage = stage.name
        meta.error = message
    }
}

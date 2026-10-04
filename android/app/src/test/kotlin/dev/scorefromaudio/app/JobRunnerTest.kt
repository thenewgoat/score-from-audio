package dev.scorefromaudio.app

import dev.scorefromaudio.pipeline.Audio
import dev.scorefromaudio.pipeline.Note
import dev.scorefromaudio.pipeline.Site
import dev.scorefromaudio.pipeline.Streams
import dev.scorefromaudio.pipeline.score.Basics
import java.io.File
import java.nio.file.Files
import java.util.concurrent.CancellationException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class JobRunnerTest {
    private val root = Files.createTempDirectory("runner").toFile()
    private val store = JobStore(File(root, "jobs"))
    private fun input() = File(root, "in.wav").apply { writeText("x") }

    /** Four quarter notes, one bar of 4/4. */
    private val tokens = List(4) { q ->
        IntArray(14).also {
            it[Streams.OFFSET] = q * 24; it[Streams.DURATION] = 24; it[Streams.PITCH] = 60 + q
            it[Streams.ACCIDENTAL] = 5; it[Streams.KEYSIGNATURE] = 7; it[Streams.VOICE] = 1
            it[Streams.STEM] = 3; it[Streams.PAD] = 1; it[Streams.VELOCITY] = 4
        }
    }

    private open inner class Fake(val notes: List<Note> = listOf(Note(0.5, 1.0, 60, 70))) : Engines {
        var sampleRateSeen = 0
        override fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note> {
            sampleRateSeen = audio.sampleRate; progress(1f); return notes
        }
        override fun tokens(notes: List<Note>, progress: (Float) -> Unit, cancelled: () -> Boolean) = tokens
    }

    private fun runner(engines: Engines, rate: Int = 44_100, decode: ((File, () -> Boolean) -> Audio)? = null,
                       memoryMb: () -> Long = { 100L }) =
        JobRunner(store, decode ?: { _, _ -> Audio(arrayOf(FloatArray(rate)), rate) }, engines, saveEveryMs = 0, memoryMb = memoryMb)

    @Test
    fun aJobWritesEveryFileAndIsDone() {
        val input = input()
        val meta = runner(Fake()).run(input, "piece", { false }) {}!!
        assertEquals("done", meta.status)
        for (name in listOf("meta.json", "notes.json", "tokens.json", "score.musicxml", "sync.json", "audio.wav")) {
            assertTrue(File(store.dir(meta.id), name).isFile, name)
        }
        assertEquals(Stage.entries.filter { it != Stage.FETCHING }.map { it.name }.toSet(), meta.timingsMs.keys)
        assertFalse(input.exists(), "the source audio is moved into the job")
        assertEquals("audio.wav", meta.audio)
        assertEquals("4/4", meta.meter)
        assertEquals(0, meta.fifths)
    }

    @Test
    fun otherSampleRatesAreResampled() {
        val fake = Fake()
        runner(fake, rate = 48_000).run(input(), "piece", { false }) {}
        assertEquals(44_100, fake.sampleRateSeen)
    }

    @Test
    fun noNotesFailsAtNotesStage() {
        val meta = runner(Fake(notes = listOf(Note(0.0, 5.0, -64, 100)))).run(input(), "silence", { false }) {}!!
        assertEquals("failed", meta.status)
        assertEquals(Stage.NOTES.name, meta.stage)
        assertEquals("No notes were detected", meta.error)
    }

    @Test
    fun anUnreadableFileFailsAtDecoding() {
        val input = input()
        val meta = runner(Fake(), decode = { _, _ -> throw UnreadableAudio("This file has no audio track") })
            .run(input, "video", { false }) {}!!
        assertEquals(Stage.DECODING.name, meta.stage)
        assertEquals("This file has no audio track", meta.error)
        assertTrue(File(store.dir(meta.id), "audio.wav").isFile, "the recording is kept to try again")
    }

    @Test
    fun theDecoderSeesCancellation() {
        val input = input()
        var asked = false
        val decode = { _: File, cancelled: () -> Boolean ->
            asked = true
            if (cancelled()) throw CancellationException("cancelled")
            Audio(arrayOf(FloatArray(10)), 44_100)
        }
        val meta = runner(Fake(), decode = decode).run(input, "stop", { true }) {}!!
        assertTrue(asked)
        assertEquals("failed", meta.status)
        assertEquals(JobRunner.STOPPED, meta.error)
        assertTrue(File(store.dir(meta.id), "audio.wav").isFile, "a stopped recording is kept")
    }

    @Test
    fun runningOutOfMemoryNamesTheStage() {
        val engines = object : Fake() {
            override fun tokens(notes: List<Note>, progress: (Float) -> Unit, cancelled: () -> Boolean): List<IntArray> =
                throw OutOfMemoryError()
        }
        var memory = 0L
        val meta = runner(engines) { memory += 10; memory }.run(input(), "big", { false }) {}!!
        assertEquals(Stage.NOTATION.name, meta.stage)
        val peak = meta.peakMemoryMbByStage.getValue(Stage.NOTATION.name)
        assertEquals("Ran out of memory while writing notation (peak $peak MB)", meta.error)
    }

    @Test
    fun anErrorFromTheRuntimeFailsTheJob() {
        val engines = object : Fake() {
            override fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note> =
                throw UnsatisfiedLinkError("no onnxruntime")
        }
        val input = input()
        val meta = runner(engines).run(input, "no runtime", { false }) {}!!
        assertEquals("failed", meta.status)
        assertEquals(Stage.NOTES.name, meta.stage)
        assertEquals("UnsatisfiedLinkError: no onnxruntime", meta.error)
        assertEquals("failed", store.load(meta.id)!!.status, "the failure is saved")
        assertFalse(input.exists())
    }

    @Test
    fun peakMemoryIsKeptPerStageAndSavedWhileRunning() {
        var memory = 100L
        var savedMidStage: Long? = null
        val engines = object : Fake() {
            override fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note> {
                memory = 900; progress(0.5f)
                memory = 300; progress(1f)
                savedMidStage = store.list().single().peakMemoryMbByStage[Stage.NOTES.name]
                return super.notes(audio, progress, cancelled)
            }
        }
        val meta = runner(engines) { memory }.run(input(), "piece", { false }) {}!!
        assertEquals(900L, savedMidStage, "a process killed mid-stage keeps the peak it saw")
        assertEquals(900L, meta.peakMemoryMbByStage[Stage.NOTES.name])
        assertEquals(300L, meta.peakMemoryMbByStage[Stage.NOTATION.name])
        assertEquals(100L, meta.peakMemoryMbByStage[Stage.DECODING.name])
        assertEquals(Stage.entries.filter { it != Stage.FETCHING }.map { it.name }.toSet(), store.load(meta.id)!!.peakMemoryMbByStage.keys)
    }

    @Test
    fun modelsThatCannotBePreparedFailAtNotes() {
        val engines = object : Fake() {
            override fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note> =
                throw IllegalStateException("models unavailable")
        }
        val meta = runner(engines).run(input(), "no models", { false }) {}!!
        assertEquals("failed", meta.status)
        assertEquals(Stage.NOTES.name, meta.stage)
        assertEquals("models unavailable", meta.error)
        assertEquals(meta.status, store.load(meta.id)!!.status, "the failure is saved")
    }

    @Test
    fun stoppingKeepsTheRecordingToTryAgain() {
        var stop = true
        val engines = object : Fake() {
            override fun notes(audio: Audio, progress: (Float) -> Unit, cancelled: () -> Boolean): List<Note> =
                if (stop) throw CancellationException("cancelled") else super.notes(audio, progress, cancelled)
        }
        val r = runner(engines)
        val stopped = r.run(input(), "stop", { true }) {}!!
        assertEquals(Stage.NOTES.name, stopped.stage)
        stop = false
        val again = r.run(stopped.id, { false }) {}!!
        assertEquals("done", again.status)
        assertNull(again.error)
        assertEquals(listOf(stopped.id), store.list().map { it.id }, "trying again reuses the job")
    }

    @Test
    fun queuedJobsWaitOldestFirst() {
        val r = runner(Fake())
        val first = store.enqueue(input(), "first")
        Thread.sleep(5)
        store.enqueue(File(root, "b.m4a").apply { writeText("x") }, "second")
        assertEquals(first.id, store.nextQueued()!!.id)
        r.run(first.id, { false }) {}
        assertEquals("second", store.nextQueued()!!.title)
        assertEquals("audio.m4a", store.nextQueued()!!.audio)
    }

    @Test
    fun aMeterPickedWhileRecordingReBarsTheScore() {
        val r = runner(Fake())
        val meta = r.run(store.enqueue(input(), "waltz", hint = "3/4").id, { false }) {}!!
        assertEquals("3/4", meta.meter)
        assertTrue("<beats>3</beats>" in File(store.dir(meta.id), "score.musicxml").readText())
    }

    @Test
    fun fixingTheBasicsRewritesTheScore() {
        val r = runner(Fake(notes = List(4) { Note(it * 0.5, it * 0.5 + 0.4, 60 + it, 70) }))
        val done = r.run(input(), "piece", { false }) {}!!
        assertEquals(120, done.bpm, "measured from the recording")
        File(store.dir(done.id), "export").mkdirs()
        val fixed = ScoreFiles.rewrite(store, done.id, Basics(3, 4, fifths = 2, bpm = 90))
        assertEquals("3/4", fixed.meter)
        assertEquals(2, fixed.fifths)
        assertEquals(90, fixed.bpm)
        assertEquals(Basics(3, 4, 2, 90), store.load(done.id)!!.basics)
        val xml = File(store.dir(done.id), "score.musicxml").readText()
        assertTrue("<fifths>2</fifths>" in xml && "<sound tempo=\"90\"/>" in xml)
        assertFalse(File(store.dir(done.id), "export").exists(), "old exports are dropped")
        assertEquals(4, ScoreFiles.readSync(File(store.dir(done.id), "sync.json").readText()).size)
    }

    private inner class FakeFetcher(val result: (File) -> Fetched) : Fetcher {
        var outDir: File? = null
        override fun fetch(url: String, site: Site, outDir: File, progress: (Float) -> Unit, cancelled: () -> Boolean): Fetched {
            this.outDir = outDir
            progress(1f)
            return result(outDir)
        }
    }

    private fun linkRunner(fetcher: Fetcher) =
        JobRunner(store, { _, _ -> Audio(arrayOf(FloatArray(44_100)), 44_100) }, Fake(), fetcher, saveEveryMs = 0) { 100L }

    @Test
    fun aLinkIsFetchedThenTranscribedAndItsDownloadDeleted() {
        val fetcher = FakeFetcher { dir -> Fetched(File(dir, "a.m4a").apply { parentFile.mkdirs(); writeText("x") }, "A Title") }
        val meta = linkRunner(fetcher).run(Source.Link("https://youtu.be/abc", Site.YOUTUBE), { false }) {}!!
        assertEquals("done", meta.status)
        assertEquals("A Title", meta.title)
        assertTrue(Stage.FETCHING.name in meta.timingsMs)
        assertFalse(fetcher.outDir!!.exists(), "the download's folder is deleted")
        assertTrue(File(store.dir(meta.id), "audio.m4a").isFile, "and the audio kept in the job")
        assertEquals("https://youtu.be/abc", meta.url)
    }

    @Test
    fun aFailedFetchIsRecordedAtFetching() {
        val fetcher = object : Fetcher {
            override fun fetch(url: String, site: Site, outDir: File, progress: (Float) -> Unit, cancelled: () -> Boolean): Fetched =
                throw FetchFailed("This post has no audio.")
        }
        val meta = linkRunner(fetcher).run(Source.Link("https://www.instagram.com/reel/x/", Site.INSTAGRAM), { false }) {}!!
        assertEquals("failed", meta.status)
        assertEquals(Stage.FETCHING.name, meta.stage)
        assertEquals("This post has no audio.", meta.error)
    }

    @Test
    fun aLinkWithNoFetchedTitleIsNamedBySiteAndId() {
        val titles = mutableListOf<String>()
        val fetcher = object : Fetcher {
            override fun fetch(url: String, site: Site, outDir: File, progress: (Float) -> Unit, cancelled: () -> Boolean): Fetched {
                titles += store.list().single().title
                return Fetched(File(outDir, "a.m4a").apply { parentFile.mkdirs(); writeText("x") }, null)
            }
        }
        val meta = linkRunner(fetcher).run(Source.Link("https://www.youtube.com/watch?v=jNQXAC9IVRw", Site.YOUTUBE), { false }) {}!!
        assertEquals(listOf("YouTube · jNQXAC9IVRw"), titles, "the job is named so while it fetches")
        assertEquals("YouTube · jNQXAC9IVRw", meta.title)
    }

    @Test
    fun cancelWhileFetchingDeletesTheJob() {
        val fetcher = object : Fetcher {
            override fun fetch(url: String, site: Site, outDir: File, progress: (Float) -> Unit, cancelled: () -> Boolean): Fetched =
                throw CancellationException("cancelled")
        }
        assertNull(linkRunner(fetcher).run(Source.Link("https://youtu.be/abc", Site.YOUTUBE), { true }) {})
        assertTrue(store.list().isEmpty())
    }

    @Test
    fun pickedFilesNeverFetch() {
        val meta = runner(Fake()).run(input(), "piece", { false }) {}!!
        assertFalse(Stage.FETCHING.name in meta.timingsMs)
    }
}

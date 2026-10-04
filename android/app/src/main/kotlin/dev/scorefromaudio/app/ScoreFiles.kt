package dev.scorefromaudio.app

import dev.scorefromaudio.pipeline.Bd1Input
import dev.scorefromaudio.pipeline.NotationInput
import dev.scorefromaudio.pipeline.Note
import dev.scorefromaudio.pipeline.NotesJson
import dev.scorefromaudio.pipeline.score.Anchor
import dev.scorefromaudio.pipeline.score.Basics
import dev.scorefromaudio.pipeline.score.ScoreWriter
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File

/** A job's score files: score.musicxml, and sync.json tying it to the recording. */
object ScoreFiles {
    private val json = Json { ignoreUnknownKeys = true }
    private val anchors = ListSerializer(Anchor.serializer())

    fun meterText(meter: Pair<Int, Int>) = "${meter.first}/${meter.second}"

    fun readSync(text: String): List<Anchor> = json.decodeFromString(anchors, text)

    /** The score, with the basics set by hand, else the meter hinted while recording; and its tie to the recording. */
    fun write(store: JobStore, meta: JobMeta, tokens: List<IntArray>, notes: List<Note>) {
        val basics = meta.basics ?: Basics.meter(meta.hint)?.let { (beats, type) -> Basics(beats, type) } ?: Basics()
        // Jobs from before bd1 kept Beyer & Dai's tokens, which index the notes in their own order.
        val onsets = if (ScoreWriter.isBd1(tokens)) Bd1Input.encode(notes).onsets else NotationInput.encode(notes).onsets
        val written = ScoreWriter.write(tokens, meta.title, basics, onsets)
        store.write(meta.id, "score.musicxml", written.xml)
        store.write(meta.id, "sync.json", json.encodeToString(anchors, written.sync))
        // The MIDI and PDF made from the old score are stale now.
        File(store.dir(meta.id), "export").deleteRecursively()
        meta.bars = written.bars
        meta.repairs = written.repairs
        meta.meter = meterText(written.meter)
        meta.fifths = written.fifths
        meta.bpm = written.bpm
    }

    /** Writes the score again with new [basics] ("Fix the basics"), from the notes and tokens the job kept. */
    fun rewrite(store: JobStore, id: String, basics: Basics): JobMeta {
        val meta = checkNotNull(store.load(id)) { "No such job" }
        val tokens = json.decodeFromString<List<List<Int>>>(store.read(id, "tokens.json")).map { it.toIntArray() }
        meta.basics = basics
        write(store, meta, tokens, NotesJson.read(store.read(id, "notes.json")))
        store.save(meta)
        return meta
    }
}

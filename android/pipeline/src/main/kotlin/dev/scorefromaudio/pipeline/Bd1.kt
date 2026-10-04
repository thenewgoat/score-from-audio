package dev.scorefromaudio.pipeline

import java.util.concurrent.CancellationException
import kotlin.math.ln1p

/**
 * Our own notation model (`notation.model`, e.g. bd1-synth) as the app runs it. Its output streams, in the order
 * the step graph takes and returns them, and the reserved pitch tokens (`notation.tokens`).
 */
object Bd1Streams {
    const val PITCH = 0
    const val ONSET = 1
    const val DURATION = 2
    const val BAR = 3
    const val HAND = 4
    const val VOICE = 5
    const val ACCIDENTAL = 6
    const val METRE = 7
    const val KEY = 8
    const val COUNT = 9

    /** A written row as the app stores it: the nine streams, then the input note it was written for and its velocity. */
    const val SOURCE = 9
    const val VELOCITY = 10
    const val ROW = 11

    val NAMES = listOf("pitch", "onset", "duration", "bar", "hand", "voice", "accidental", "metre", "key")

    const val PITCH_PAD = 128
    const val PITCH_BOS = 129
    const val PITCH_EOS = 130
    const val PITCH_REST = 131
    const val PITCH_SPACE = 132
    const val PITCH_UNPLAYED = 133
    val NEVER_WRITTEN = intArrayOf(PITCH_PAD, PITCH_BOS, PITCH_EOS, PITCH_UNPLAYED)

    /** The decoder's first input: BOS in the pitch stream, index 0 in every other. */
    fun start(): IntArray = IntArray(COUNT).also { it[PITCH] = PITCH_BOS }
}

/**
 * bd1's input streams, in its slot order: `notation.predict.performance_notes` (times rounded to float32, sorted by
 * onset, offset, pitch), each note's gap from the note before it and its length on `notation.tokens.gap_bucket`'s
 * 200 log slots, and its velocity in `notation.vocab.velocity_bucket`'s. Pedals are not notes here.
 */
class Bd1Input(val notes: List<Note>, val pitch: IntArray, val onset: IntArray, val duration: IntArray, val velocity: IntArray) {
    val size: Int get() = notes.size

    /** Each slot's onset in seconds: a written row's [Bd1Streams.SOURCE] indexes this. */
    val onsets: DoubleArray get() = DoubleArray(size) { notes[it].start }

    companion object {
        private const val GAP_LIMIT = 60.0
        private const val GAP_KNEE = 0.02
        private const val BUCKETS = 200

        fun gapBucket(seconds: Double): Int {
            val clamped = seconds.coerceIn(0.0, GAP_LIMIT)
            val squashed = ln1p(clamped / GAP_KNEE) / ln1p(GAP_LIMIT / GAP_KNEE)
            return minOf(BUCKETS - 1, (squashed * (BUCKETS - 1) + 0.5).toInt())
        }

        fun velocityBucket(velocity: Int): Int = 1 + minOf(6, velocity.coerceIn(0, 127) * 7 / 128)

        fun encode(detected: List<Note>): Bd1Input {
            val notes = detected.filter { it.pitch >= 0 }
                .map { Note(it.start.toFloat().toDouble(), it.end.toFloat().toDouble(), it.pitch, it.velocity) }
                .sortedWith(compareBy({ it.start }, { it.end }, { it.pitch }))
            val n = notes.size
            val onset = IntArray(n) { if (it == 0) gapBucket(0.0) else gapBucket(maxOf(notes[it].start - notes[it - 1].start, 0.0)) }
            val duration = IntArray(n) { gapBucket(maxOf(notes[it].end - notes[it].start, 0.0)) }
            return Bd1Input(notes, IntArray(n) { notes[it].pitch }, onset, duration, IntArray(n) { velocityBucket(notes[it].velocity) })
        }
    }
}

/** bd1 as exported by `export.bd1_export`: an encoder over a chunk of notes, and one cached decoder step. */
interface Bd1Model {
    fun encode(input: Bd1Input, from: Int, to: Int): Encoded
    fun newCache(): Cache
    /** Feed one row of the nine streams at position `cache.length`; returns the next slot's nine logit vectors. */
    fun step(encoded: Encoded, row: IntArray, cache: Cache): Array<FloatArray>
}

/**
 * bd1's decode loop, `notation.predict.aligned_greedy`: one slot per note, in chunks of 511 notes overlapping by 64,
 * each later chunk given the previous chunk's last 64 slots as decoder context. The pitch head never writes a
 * control or input-only token, and a space (a played note the score does not write) has its other streams written
 * as 0, as a `space_loss="pitch"` model was trained. Spaces are then dropped, a bar line one carried moving to the
 * next kept slot. `export/bd1_ref.py` is the reference and Bd1ParityTest holds this to it.
 */
class Bd1Notator(private val model: Bd1Model, private val chunk: Int = 512, private val overlap: Int = 64) {
    /** Every slot, spaces included: nine streams each. */
    fun slots(input: Bd1Input, progress: (Float) -> Unit = {}, cancelled: () -> Boolean = { false }): List<IntArray> {
        val n = input.size
        val out = ArrayList<IntArray>(n)
        for ((lo, hi) in chunkStarts(n)) {
            val prefix = out.subList(lo, out.size).map { it.copyOf() }
            out.addAll(decodeChunk(input, lo, hi, prefix, cancelled))
            progress(out.size.toFloat() / n)
        }
        check(out.size == n) { "decoded ${out.size} slots for $n notes" }
        return out
    }

    /** The written rows ([Bd1Streams.ROW] wide), spaces removed. */
    fun notate(input: Bd1Input, progress: (Float) -> Unit = {}, cancelled: () -> Boolean = { false }): List<IntArray> =
        withoutSpaces(slots(input, progress, cancelled), input)

    /** `notation.predict.chunk_starts`: `chunk - 1` notes each, consecutive chunks sharing `overlap`. */
    fun chunkStarts(count: Int): List<Pair<Int, Int>> {
        if (count == 0) return emptyList()
        val width = chunk - 1
        return (0 until maxOf(count - overlap, 1) step (width - overlap)).map { it to minOf(it + width, count) }
    }

    private fun decodeChunk(input: Bd1Input, lo: Int, hi: Int, prefix: List<IntArray>, cancelled: () -> Boolean): List<IntArray> =
        model.encode(input, lo, hi).use { encoded ->
            model.newCache().use { cache ->
                var logits = model.step(encoded, Bd1Streams.start(), cache)
                for (row in prefix) logits = model.step(encoded, row, cache)
                val count = (hi - lo) - prefix.size
                val rows = ArrayList<IntArray>(count)
                while (true) {
                    if (cancelled()) throw CancellationException("cancelled")
                    val row = choose(logits)
                    rows.add(row)
                    if (rows.size == count) break
                    logits = model.step(encoded, row, cache)
                }
                rows
            }
        }

    companion object {
        fun choose(logits: Array<FloatArray>): IntArray {
            val row = IntArray(Bd1Streams.COUNT)
            for (j in 0 until Bd1Streams.COUNT) {
                val scores = logits[j]
                var best = -1
                for (k in scores.indices) {
                    if (j == Bd1Streams.PITCH && k in Bd1Streams.NEVER_WRITTEN) continue
                    if (best < 0 || scores[k] > scores[best]) best = k
                }
                row[j] = best
            }
            if (row[Bd1Streams.PITCH] == Bd1Streams.PITCH_SPACE) for (j in 1 until Bd1Streams.COUNT) row[j] = 0
            return row
        }

        fun withoutSpaces(slots: List<IntArray>, input: Bd1Input): List<IntArray> {
            val kept = ArrayList<IntArray>(slots.size)
            var pending = 0
            for ((index, slot) in slots.withIndex()) {
                if (slot[Bd1Streams.PITCH] == Bd1Streams.PITCH_SPACE) {
                    if (slot[Bd1Streams.BAR] != 0) pending = slot[Bd1Streams.BAR]
                    continue
                }
                val row = slot.copyOf(Bd1Streams.ROW)
                if (pending != 0) {
                    if (row[Bd1Streams.BAR] == 0) row[Bd1Streams.BAR] = pending
                    pending = 0
                }
                row[Bd1Streams.SOURCE] = index
                row[Bd1Streams.VELOCITY] = input.notes[index].velocity
                kept.add(row)
            }
            return kept
        }
    }
}

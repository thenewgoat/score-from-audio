package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.Streams

/** One note as the writer sees it. Positions and durations in 24ths of a quarter; staff 1 upper, 2 lower. */
data class ScoreNote(
    val measure: Int, val offset: Int, val duration: Int, val pitch: Int, val alter: Int, val accidental: Int,
    val staff: Int, val voice: Int, val stem: String?, val grace: Boolean, val trill: Boolean,
    val staccato: Boolean, val velocity: Int,
    /** The token row this note came from, so its time in the recording can be looked up; -1 when made by hand. */
    val row: Int = -1,
)

data class Bar(val length: Int, val fifths: Int)

/** [meter] names the time signature of bars of its length, where the length alone is ambiguous (6/8 against 3/4). */
class Layout(val bars: List<Bar>, val notes: List<ScoreNote>, val pickup: Boolean, val repairs: Repairs,
             val meter: Pair<Int, Int>? = null) {
    /** Where each bar starts, in 24ths of a quarter from the start of the score. */
    val starts: IntArray by lazy { IntArray(bars.size).also { for (k in 1 until bars.size) it[k] = it[k - 1] + bars[k - 1].length } }

    fun position(note: ScoreNote): Int = starts[note.measure] + note.offset

    fun timeSignature(length: Int): Pair<Int, Int> =
        meter?.takeIf { (beats, type) -> beats * 96 / type == length } ?: Measures.timeSignature(length)

    /** The bar length most bars have, the pickup aside. */
    val usualLength: Int
        get() = bars.drop(if (pickup) 1 else 0).ifEmpty { bars }.groupingBy { it.length }.eachCount()
            .maxWith(compareBy<Map.Entry<Int, Int>> { it.value }.thenBy { it.key }).key
}

/**
 * Bars, keys and notes from the model's token rows.
 *
 * A note starts a new bar when the model marks it as a downbeat, or when its
 * offset falls below the previous note's (the model's own decoder forces a
 * downbeat there too). The downbeat token states the length of the bar that
 * just ended, so the last bar has no stated length and takes its neighbour's.
 */
object Measures {
    private const val QUARTER = 24

    fun timeSignature(length: Int): Pair<Int, Int> = when {
        length % 24 == 0 -> length / 24 to 4
        length % 12 == 0 -> length / 12 to 8
        else -> length / 6 to 16
    }

    fun layout(tokens: List<IntArray>, repairs: Repairs = Repairs()): Layout {
        val kept = tokens.indices.filter { i -> (tokens[i][Streams.PAD] == 1).also { if (!it) repairs.droppedByModel++ } }
        val rows = kept.map { tokens[it] }
        if (rows.isEmpty()) return Layout(listOf(Bar(4 * QUARTER, 0)), emptyList(), false, repairs)

        val barOf = IntArray(rows.size)
        val stated = ArrayList<Int>()
        for (i in 1 until rows.size) {
            val downbeat = rows[i][Streams.DOWNBEAT]
            if (downbeat >= 1 || rows[i][Streams.OFFSET] < rows[i - 1][Streams.OFFSET]) {
                stated.add(if (downbeat >= 1) downbeat - 1 else 0)
                barOf[i] = barOf[i - 1] + 1
            } else {
                barOf[i] = barOf[i - 1]
            }
        }
        val count = barOf.last() + 1
        val lengths = IntArray(count) { stated.getOrElse(it) { 0 } }
        for (k in 0 until count) if (lengths[k] <= 0) {
            if (k < stated.size) repairs.barLengthsRounded++
            lengths[k] = (k - 1 downTo 0).map { lengths[it] }.firstOrNull { it > 0 }
                ?: (k + 1 until stated.size).map { stated[it] }.firstOrNull { it > 0 }
                ?: 4 * QUARTER
        }
        for (k in 0 until count) if (lengths[k] % 6 != 0) {
            lengths[k] = maxOf(6, Math.round(lengths[k] / 6.0).toInt() * 6)
            repairs.barLengthsRounded++
        }
        val pickup = count > 1 && lengths[0] < lengths[1]

        val firstRow = IntArray(count) { -1 }
        for (i in rows.indices) if (firstRow[barOf[i]] < 0) firstRow[barOf[i]] = i
        val fifths = IntArray(count)
        var key = 0
        for (k in 0 until count) {
            val token = rows[firstRow[k]][Streams.KEYSIGNATURE]
            if (token in 0..14) key = token - 7
            fifths[k] = key
        }

        val notes = rows.mapIndexed { i, r ->
            val bar = barOf[i]
            var offset = r[Streams.OFFSET]
            if (offset >= lengths[bar]) {
                offset = maxOf(lengths[bar] - 6, 0)
                repairs.offsetsClamped++
            }
            val grace = r[Streams.GRACE] == 1 || r[Streams.DURATION] <= 0
            val spelled = Spelling.alter(r[Streams.PITCH], r[Streams.ACCIDENTAL], fifths[bar])
            if (spelled.overridden) repairs.spellingsFixed++
            ScoreNote(
                measure = bar, offset = offset, duration = if (grace) 0 else r[Streams.DURATION],
                pitch = r[Streams.PITCH], alter = spelled.alter, accidental = r[Streams.ACCIDENTAL],
                staff = when (r[Streams.HAND]) { 0 -> 1; 1 -> 2; else -> if (r[Streams.PITCH] >= 60) 1 else 2 },
                voice = if (r[Streams.VOICE] in 1..8) r[Streams.VOICE] else 1,
                stem = when (r[Streams.STEM]) { 0 -> "up"; 1 -> "down"; 2 -> "none"; else -> null },
                grace = grace, trill = r[Streams.TRILL] == 1, staccato = r[Streams.STACCATO] == 1,
                velocity = if (r[Streams.VELOCITY] >= 0) r[Streams.VELOCITY] * 16 + 8 else 64,
                row = kept[i],
            )
        }
        return Layout(lengths.indices.map { Bar(lengths[it], fifths[it]) }, notes, pickup, repairs)
    }
}

package dev.scorefromaudio.pipeline.score

import kotlinx.serialization.Serializable

/**
 * What a person can correct once the score is written ("Fix the basics"): beats per bar, key and tempo.
 * Null leaves the model's choice (or, for the tempo, the one measured from the recording).
 */
@Serializable
data class Basics(val beats: Int? = null, val beatType: Int? = null, val fifths: Int? = null, val bpm: Int? = null) {
    /** The bar length the meter asks for, in 24ths of a quarter; null when no meter is set. */
    val barLength: Int? get() = if (beats != null && beatType != null) beats * 96 / beatType else null

    companion object {
        /** "3/4" -> 3 and 4; null for anything else, such as "Not sure". */
        fun meter(text: String?): Pair<Int, Int>? {
            val parts = text?.split('/')?.mapNotNull { it.trim().toIntOrNull() } ?: return null
            return if (parts.size == 2 && parts[0] in 1..16 && parts[1] in setOf(2, 4, 8, 16)) parts[0] to parts[1] else null
        }
    }
}

/** Changes a layout to the [Basics]: new barlines for another meter, another key. Notes keep their places in time. */
object Rebar {
    /** Every bar but the pickup at the meter's length, or [layout] unchanged when it already is. */
    fun meter(layout: Layout, beats: Int, beatType: Int): Layout {
        val length = beats * 96 / beatType
        val full = layout.bars.drop(if (layout.pickup) 1 else 0)
        if (full.all { it.length == length }) return Layout(layout.bars, layout.notes, layout.pickup, layout.repairs, beats to beatType)
        val positions = layout.notes.map { layout.position(it) }
        // Bars run to the end of the last note, so no empty bar is left over from the model's last one.
        val end = layout.notes.indices.maxOfOrNull { positions[it] + maxOf(layout.notes[it].duration, 1) } ?: length
        val pickup = pickupLength(layout, positions, length, beatType)
        val starts = ArrayList<Int>()
        if (pickup > 0) starts.add(0)
        var start = pickup
        do { starts.add(start); start += length } while (start < end)
        val lengths = starts.indices.map { if (it == 0 && pickup > 0) pickup else length }
        fun barAt(position: Int) = starts.indexOfLast { it <= position }
        fun fifthsAt(position: Int) = layout.bars[layout.starts.indexOfLast { it <= position }].fifths
        val bars = starts.indices.map { Bar(lengths[it], fifthsAt(starts[it])) }
        val notes = layout.notes.mapIndexed { i, n ->
            val bar = barAt(positions[i])
            n.copy(measure = bar, offset = positions[i] - starts[bar])
        }
        return Layout(bars, notes, pickup > 0 && bars.size > 1, layout.repairs, beats to beatType)
    }

    /**
     * Where the first full bar starts: the beat that puts the most weight of notes on downbeats. Ties go to the model's
     * own first barline, then to no pickup.
     */
    private fun pickupLength(layout: Layout, positions: List<Int>, length: Int, beatType: Int): Int {
        val beat = if (beatType == 8 && length % 36 == 0) 36 else 96 / beatType
        val weight = IntArray(length / beat)
        layout.notes.forEachIndexed { i, n ->
            if (!n.grace && positions[i] % beat == 0) weight[(positions[i] % length) / beat] += n.velocity
        }
        val model = if (layout.pickup) layout.bars[0].length % length else 0
        return weight.indices.maxWith(compareBy<Int> { weight[it] }.thenBy { it * beat == model }.thenBy { it == 0 }) * beat
    }

    /** Every bar in the key of [fifths], with black keys spelled as its sharps or flats. */
    fun key(layout: Layout, fifths: Int): Layout {
        val notes = layout.notes.map { n ->
            val alter = if (Math.floorMod(n.pitch, 12) in setOf(1, 3, 6, 8, 10)) (if (fifths >= 0) 1 else -1) else 0
            n.copy(alter = alter)
        }
        return Layout(layout.bars.map { it.copy(fifths = fifths) }, notes, layout.pickup, layout.repairs, layout.meter)
    }
}

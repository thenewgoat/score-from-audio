package dev.scorefromaudio.pipeline.score

/** A writable note value: type, dots, whether it is a triplet member, and its length in 24ths of a quarter. */
data class NoteValue(val type: String, val dots: Int, val triplet: Boolean, val ticks: Int)

object Durations {
    private val PLAIN = listOf("whole" to 96, "half" to 48, "quarter" to 24, "eighth" to 12, "16th" to 6, "32nd" to 3)

    val VALUES: List<NoteValue> = buildList {
        for ((type, t) in PLAIN) {
            add(NoteValue(type, 0, false, t))
            if (t % 2 == 0) add(NoteValue(type, 1, false, t * 3 / 2))
            if (t % 4 == 0) add(NoteValue(type, 2, false, t * 7 / 4))
            if (t % 3 == 0) add(NoteValue(type, 0, true, t * 2 / 3))
        }
        add(NoteValue("64th", 0, true, 1))
    }.sortedByDescending { it.ticks }

    /**
     * One value when one exists; otherwise tied pieces chosen greedily, each the largest value that
     * still fits, so largest first (usually, not always, the fewest pieces).
     * A length divisible by 3 is always written without triplets.
     */
    fun split(ticks: Int): List<NoteValue> {
        require(ticks > 0) { "ticks must be positive, was $ticks" }
        VALUES.firstOrNull { it.ticks == ticks }?.let { return listOf(it) }
        val candidates = VALUES.filter { it.dots < 2 && (ticks % 3 != 0 || !it.triplet) }
        val out = ArrayList<NoteValue>()
        var left = ticks
        while (left > 0) {
            val v = candidates.first { it.ticks <= left }
            out.add(v)
            left -= v.ticks
        }
        return out
    }
}

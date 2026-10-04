package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.Bd1Streams
import kotlin.math.roundToInt

/**
 * Bars, keys and notes from bd1's written rows: `notation.build.build_score`'s reading of the streams, into the same
 * [Layout] Beyer & Dai's tokens become, so voicing and the MusicXML writer are shared.
 *
 * A non-zero bar slot opens a bar of that length on its row; everything after it belongs to that bar until the next.
 * Onsets are quarters from the bar line, durations quarters (slot 0 a grace note), both read off `notation.vocab`'s
 * tables and put on the writer's grid of 24 a quarter, which cannot write bd1's quintuplets or 64th-note fractions
 * of a quarter exactly: they land on the nearest 24th. A note is clamped into its bar, and a fragment that ends on
 * the bar line joins the same pitch opening the next bar in its staff and voice, which is how the streams say "tied"
 * (`build._tie_across_bars`); the writer splits and ties it again. Time signature and key are majority votes per bar,
 * the metre's even votes going to the compound reading (`build._metre_of`, `build._key_of`).
 */
object Bd1Layout {
    private const val QUARTER = 24
    private val DENOMINATORS = intArrayOf(1, 2, 3, 4, 6, 8, 10, 12, 16, 24)

    /** Every n/d up to [limit] quarters, sorted, without duplicates, in 24ths (rounded). */
    private fun family(limit: Int, zero: Boolean): List<Int> {
        val values = sortedSetOf<Pair<Long, Long>>(compareBy { it.first.toDouble() / it.second })
        val seen = HashSet<Pair<Long, Long>>()
        for (d in DENOMINATORS) {
            var n = if (zero) 0 else 1
            while (n <= limit * d) {
                val g = gcd(n.toLong(), d.toLong()).coerceAtLeast(1)
                val reduced = n / g to d / g
                if (seen.add(reduced)) values.add(reduced)
                n++
            }
        }
        return values.map { (it.first * QUARTER.toDouble() / it.second).roundToInt() }
    }

    private tailrec fun gcd(a: Long, b: Long): Long = if (b == 0L) a else gcd(b, a % b)

    /** `ONSETS`, `DURATIONS` (slot 0 grace, here 0) and `BAR_LENGTHS` (slot 0 "no bar line", here 0), in 24ths. */
    val ONSETS: List<Int> = family(6, zero = true)
    val DURATIONS: List<Int> = listOf(0) + family(4, zero = false)
    val BAR_LENGTHS: List<Int> = listOf(0) + family(6, zero = false)
    val TIME_SIGNATURES: List<Pair<Int, Int>?> = listOf(null, 1 to 2, 2 to 2, 2 to 4, 3 to 2, 3 to 4, 3 to 8, 4 to 4, 4 to 8,
        5 to 4, 5 to 8, 6 to 4, 6 to 8, 7 to 8, 9 to 8, 12 to 8, 12 to 16)

    private fun <T> slot(table: List<T>, index: Int): T = table[index.coerceIn(0, table.size - 1)]

    private fun compound(m: Pair<Int, Int>) = m.second in setOf(8, 16) && m.first > 3 && m.first % 3 == 0

    /** `build._metre_of`: the most voted signature, an even vote going to the compound one, then the lower slot. */
    fun metreOf(votes: List<Int>): Pair<Int, Int>? {
        val counts = votes.filter { it in 1 until TIME_SIGNATURES.size }.groupingBy { it }.eachCount()
        if (counts.isEmpty()) return null
        val best = counts.entries.maxWith(compareBy<Map.Entry<Int, Int>> { it.value }
            .thenBy { compound(TIME_SIGNATURES[it.key]!!) }.thenBy { -it.key })
        return TIME_SIGNATURES[best.key]
    }

    /** `build._key_of`: fifths by majority of real key votes; a tie keeps [previous] when it is tied, else the flatter. */
    fun keyOf(votes: List<Int>, previous: Int?): Int? {
        val counts = votes.filter { it != 0 }.map { it - 8 }.groupingBy { it }.eachCount()
        if (counts.isEmpty()) return previous
        val top = counts.values.max()
        val tied = counts.filterValues { it == top }.keys
        return if (previous != null && previous in tied) previous else tied.min()
    }

    private class Entry(val row: IntArray, val onset: Int, val duration: Int, val staff: Int, val voice: Int) {
        /** Its length with every later fragment folded in. */
        var total = duration
    }

    private class Measure(val length: Int) {
        val entries = ArrayList<Entry>()
        val metres = ArrayList<Int>()
        val keys = ArrayList<Int>()
    }

    fun layout(rows: List<IntArray>, repairs: Repairs = Repairs()): Layout {
        val measures = ArrayList<Measure>()
        for (row in rows) {
            val bar = slot(BAR_LENGTHS, row[Bd1Streams.BAR])
            if (bar != 0 || measures.isEmpty()) {
                var length = if (bar != 0) bar else 4 * QUARTER
                if (bar == 0 && measures.isNotEmpty()) length = measures.last().length
                if (length % 6 != 0) {
                    length = maxOf(6, Math.round(length / 6.0).toInt() * 6)
                    repairs.barLengthsRounded++
                }
                measures.add(Measure(length))
            }
            val m = measures.last()
            val pitch = row[Bd1Streams.PITCH]
            if (pitch !in 0..127 && pitch != Bd1Streams.PITCH_REST) {
                repairs.droppedByModel++
                continue
            }
            m.metres.add(row[Bd1Streams.METRE])
            m.keys.add(row[Bd1Streams.KEY])
            if (pitch == Bd1Streams.PITCH_REST) continue
            var onset = slot(ONSETS, row[Bd1Streams.ONSET])
            if (onset >= m.length) {
                onset = maxOf(m.length - 6, 0)
                repairs.offsetsClamped++
            }
            val written = slot(DURATIONS, row[Bd1Streams.DURATION])
            val duration = if (row[Bd1Streams.DURATION] <= 0) 0 else minOf(maxOf(written, 1), m.length - onset).coerceAtLeast(1)
            m.entries.add(Entry(row, onset, duration, if (row[Bd1Streams.HAND] <= 0) 1 else 2,
                                row[Bd1Streams.VOICE].coerceIn(0, 7) + 1))
        }
        if (measures.isEmpty()) return Layout(listOf(Bar(4 * QUARTER, 0)), emptyList(), false, repairs)

        val keys = ArrayList<Int?>()
        var voted: Int? = null
        for (m in measures) { voted = keyOf(m.keys, voted); keys.add(voted) }
        val firstReal = keys.indexOfFirst { it != null }
        for (k in 0 until maxOf(firstReal, 0)) keys[k] = keys[firstReal]
        val fifths = keys.map { it ?: 0 }

        // A note ending on the bar line and the same pitch opening the next bar, in the same staff and voice, are one
        // note: the later fragment is folded into the earlier one.
        val joined = HashSet<Entry>()
        val root = HashMap<Entry, Entry>()
        for (k in 0 until measures.size - 1) {
            val length = measures[k].length
            val opening = measures[k + 1].entries.filter { it.onset == 0 && it.duration > 0 }.toMutableList()
            for (e in measures[k].entries) {
                if (e.duration == 0 || e.onset + e.duration != length) continue
                val next = opening.firstOrNull {
                    it.row[Bd1Streams.PITCH] == e.row[Bd1Streams.PITCH] && it.staff == e.staff && it.voice == e.voice
                } ?: continue
                opening.remove(next)
                joined.add(next)
                val first = root[e] ?: e
                root[next] = first
                first.total += next.duration
            }
        }

        val notes = ArrayList<ScoreNote>()
        for ((k, m) in measures.withIndex()) for (e in m.entries) {
            if (e in joined) continue
            val pitch = e.row[Bd1Streams.PITCH]
            val token = when (e.row[Bd1Streams.ACCIDENTAL]) { 1 -> 1; 2 -> 0; 3 -> 3; 4 -> 4; 5 -> 2; else -> -1 }
            val spelled = Spelling.alter(pitch, token, fifths[k])
            if (spelled.overridden) repairs.spellingsFixed++
            notes.add(ScoreNote(
                measure = k, offset = e.onset, duration = e.total, pitch = pitch, alter = spelled.alter,
                accidental = token, staff = e.staff, voice = e.voice, stem = null, grace = e.total == 0,
                trill = false, staccato = false, velocity = e.row[Bd1Streams.VELOCITY],
                row = e.row[Bd1Streams.SOURCE],
            ))
        }
        val bars = measures.indices.map { Bar(measures[it].length, fifths[it]) }
        val pickup = bars.size > 1 && bars[0].length < bars[1].length
        val usual = bars.drop(if (pickup) 1 else 0).ifEmpty { bars }.groupingBy { it.length }.eachCount()
            .maxWith(compareBy<Map.Entry<Int, Int>> { it.value }.thenBy { it.key }).key
        val meter = metreOf(measures.filter { it.length == usual }.flatMap { it.metres })
            ?.takeIf { (beats, type) -> beats * 96 / type == usual }
        return Layout(bars, notes, pickup, repairs, meter)
    }
}

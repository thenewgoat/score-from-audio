package dev.scorefromaudio.pipeline.score

import java.util.TreeMap
import kotlin.math.abs

/** The part of one note that lies inside one bar; `next` is its continuation in the following bar. */
class Piece(val note: ScoreNote, val bar: Int, val offset: Int, var ticks: Int, var tieStart: Boolean, var tieStop: Boolean) {
    var next: Piece? = null

    /** A shortened piece no longer reaches the barline, so neither end of its tie survives. */
    fun cutTie() {
        tieStart = false
        next?.tieStop = false
    }
}

sealed class Item {
    abstract val ticks: Int
    var tupletStart = false
    var tupletStop = false
}

/** One written value of a chord. A chord longer than any single value is several NoteItems, tied. */
class NoteItem(val pieces: List<Piece>, val value: NoteValue, val first: Boolean, val last: Boolean) : Item() {
    override val ticks: Int get() = value.ticks
}

class GraceItem(val pieces: List<Piece>) : Item() {
    override val ticks: Int get() = 0
}

/** `value == null` is a whole-bar rest. */
class RestItem(val value: NoteValue?, override val ticks: Int) : Item()

class ForwardItem(override val ticks: Int) : Item()

class Line(val staff: Int, val voice: Int, val items: List<Item>)

class BarLines(val bar: Int, val length: Int, val lines: List<Line>)

/**
 * Lays each bar of each staff out as voices whose items sum to the bar.
 *
 * Notes crossing a barline are split and tied. Notes that share a voice,
 * a start and a length are a chord. A note that overlaps another in its voice
 * moves to the nearest free voice; when none is free, the earlier note is
 * shortened. The first voice fills its gaps with rests, the others with
 * invisible forwards.
 */
object Voices {
    private val WHOLE = NoteValue("whole", 0, false, 96)
    private val HALF = NoteValue("half", 0, false, 48)

    private class Chord(val start: Int, var ticks: Int, val pieces: MutableList<Piece>, var voice: Int, val requested: Int) {
        val end: Int get() = start + ticks
    }

    fun arrange(layout: Layout): List<BarLines> {
        val byBarStaff = split(layout).groupBy { it.bar to it.note.staff }
        return layout.bars.indices.map { bar ->
            val length = layout.bars[bar].length
            BarLines(bar, length, (1..2).flatMap { staff ->
                staffLines(byBarStaff[bar to staff].orEmpty(), staff, length, layout.pickup && bar == 0, layout.repairs)
            })
        }
    }

    private fun split(layout: Layout): List<Piece> {
        val out = ArrayList<Piece>()
        for (n in layout.notes) {
            if (n.grace) { out.add(Piece(n, n.measure, n.offset, 0, false, false)); continue }
            var left = n.duration
            var bar = n.measure
            var offset = n.offset
            var first = true
            var previous: Piece? = null
            while (left > 0) {
                val part = minOf(left, layout.bars[bar].length - offset)
                left -= part
                val continues = left > 0 && bar + 1 < layout.bars.size
                if (left > 0 && !continues) layout.repairs.tiesPastEnd++
                val piece = Piece(n, bar, offset, part, continues, !first)
                previous?.next = piece
                previous = piece
                out.add(piece)
                if (!continues) break
                bar++; offset = 0; first = false
            }
        }
        return out
    }

    private fun free(chords: List<Chord>, voice: Int, start: Int, end: Int) =
        chords.none { it.voice == voice && it.start < end && start < it.end }

    private fun staffLines(pieces: List<Piece>, staff: Int, length: Int, pickup: Boolean, repairs: Repairs): List<Line> {
        val graces = pieces.filter { it.note.grace }
        val chords = ArrayList<Chord>()
        for (p in pieces.filter { !it.note.grace }.sortedWith(compareBy({ it.offset }, { it.note.voice }, { it.note.pitch }))) {
            val requested = p.note.voice
            val same = chords.firstOrNull { it.requested == requested && it.start == p.offset && it.ticks == p.ticks }
            if (same != null) { same.pieces.add(p); continue }
            val chord = Chord(p.offset, p.ticks, mutableListOf(p), requested, requested)
            if (!free(chords, requested, chord.start, chord.end)) {
                val other = (1..8).filter { it != requested }.sortedWith(compareBy({ abs(it - requested) }, { it }))
                    .firstOrNull { free(chords, it, chord.start, chord.end) }
                if (other != null) {
                    chord.voice = other
                    repairs.overlapsMoved++
                } else {
                    val clash = chords.filter { it.voice == requested && it.start < chord.end && chord.start < it.end }
                    val joined = clash.firstOrNull { it.start == chord.start }
                    repairs.overlapsShortened++
                    if (joined != null) {
                        p.ticks = joined.ticks
                        p.cutTie()
                        joined.pieces.add(p)
                        continue
                    }
                    for (c in clash) {
                        c.ticks = chord.start - c.start
                        c.pieces.forEach { it.cutTie() }
                    }
                }
            }
            chords.add(chord)
        }

        val voices = (chords.map { it.voice } + graces.map { it.note.voice }).distinct().sorted()
        if (voices.isEmpty()) {
            // A pickup is shorter than its time signature, so its rests are written out, not "the whole bar".
            val rest = if (pickup) items(emptyList(), emptyList(), true, length) else listOf(RestItem(null, length))
            return listOf(Line(staff, 1, rest))
        }
        val primary = voices.first()
        return voices.map { v ->
            Line(staff, v, items(chords.filter { it.voice == v }.sortedBy { it.start },
                                 graces.filter { it.note.voice == v }, v == primary, length))
        }
    }

    private fun items(chords: List<Chord>, graces: List<Piece>, primary: Boolean, length: Int): List<Item> {
        val out = ArrayList<Item>()
        var cursor = 0
        val pending = TreeMap(graces.sortedBy { it.note.pitch }.groupBy { it.offset })

        fun gap(ticks: Int) {
            if (ticks <= 0) return
            if (!primary) { out.add(ForwardItem(ticks)); return }
            for (v in Durations.split(ticks)) {
                // A whole rest reads as "the whole bar"; in a bar of any other length, four beats are two halves.
                if (v == WHOLE && length != WHOLE.ticks) repeat(2) { out.add(RestItem(HALF, HALF.ticks)) }
                else out.add(RestItem(v, v.ticks))
            }
        }

        fun flushGraces(upTo: Int) {
            for (at in pending.headMap(upTo, true).keys.toList()) {
                if (at > cursor) { gap(at - cursor); cursor = at }
                out.add(GraceItem(pending.remove(at)!!))
            }
        }

        for (c in chords) {
            flushGraces(c.start)
            if (c.start > cursor) { gap(c.start - cursor); cursor = c.start }
            val values = Durations.split(c.ticks)
            values.forEachIndexed { i, v -> out.add(NoteItem(c.pieces, v, first = i == 0, last = i == values.lastIndex)) }
            cursor += c.ticks
        }
        flushGraces(length)
        gap(length - cursor)
        markTuplets(out)
        return out
    }

    /** Bracket runs of triplet values: close at three of the first value's length, or on a beat. */
    private fun markTuplets(items: List<Item>) {
        var open = false
        var first = 0
        var sum = 0
        var last: Item? = null
        for (item in items) {
            if (item is GraceItem) continue
            val value = when (item) { is NoteItem -> item.value; is RestItem -> item.value; else -> null }
            if (value == null || !value.triplet) {
                if (open) { last?.tupletStop = true; open = false }
                continue
            }
            if (!open) { open = true; item.tupletStart = true; first = value.ticks; sum = 0 }
            sum += value.ticks
            last = item
            if (sum % (3 * first) == 0 || sum % 24 == 0) { item.tupletStop = true; open = false }
        }
        if (open) last?.tupletStop = true
    }
}

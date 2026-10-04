package dev.scorefromaudio.pipeline.score

import java.util.Collections
import java.util.IdentityHashMap
import java.util.Locale

/**
 * MusicXML 4.0: one piano part, two staves, 24 divisions per quarter so every
 * token position is exact. Elements inside <note> follow the schema's order,
 * because readers that validate reject a file that does not.
 */
object MusicXml {
    private val ACCIDENTALS = mapOf(-2 to "flat-flat", -1 to "flat", 0 to "natural", 1 to "sharp", 2 to "double-sharp")

    /** [bpm], quarter notes a minute, is written as a metronome mark and a playback tempo on the first bar. */
    fun write(layout: Layout, bars: List<BarLines>, title: String, bpm: Int? = null): String {
        val x = StringBuilder()
        x.append("""<?xml version="1.0" encoding="UTF-8"?>""").append('\n')
        x.append("""<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">""").append('\n')
        x.append("""<score-partwise version="4.0">""")
        x.append("<work><work-title>").append(escape(title)).append("</work-title></work>")
        x.append("""<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>""")
        x.append("""<part id="P1">""")
        var shownLength = -1
        var shownFifths = Int.MIN_VALUE
        for (b in bars) {
            val bar = layout.bars[b.bar]
            val implicit = layout.pickup && b.bar == 0
            val number = if (layout.pickup) b.bar else b.bar + 1
            x.append("<measure number=\"$number\"").append(if (implicit) " implicit=\"yes\"" else "").append(">")
            val attributes = StringBuilder()
            if (b.bar == 0) attributes.append("<divisions>24</divisions>")
            if (bar.fifths != shownFifths) {
                attributes.append("<key><fifths>${bar.fifths}</fifths></key>")
                shownFifths = bar.fifths
            }
            val timeLength = if (implicit) layout.bars[1].length else bar.length
            if (timeLength != shownLength) {
                val (beats, type) = layout.timeSignature(timeLength)
                attributes.append("<time><beats>$beats</beats><beat-type>$type</beat-type></time>")
                shownLength = timeLength
            }
            if (b.bar == 0) {
                attributes.append("<staves>2</staves>")
                attributes.append("<clef number=\"1\"><sign>G</sign><line>2</line></clef>")
                attributes.append("<clef number=\"2\"><sign>F</sign><line>4</line></clef>")
            }
            if (attributes.isNotEmpty()) x.append("<attributes>").append(attributes).append("</attributes>")
            if (b.bar == 0 && bpm != null) {
                x.append("<direction placement=\"above\"><direction-type><metronome><beat-unit>quarter</beat-unit>")
                x.append("<per-minute>$bpm</per-minute></metronome></direction-type><staff>1</staff><sound tempo=\"$bpm\"/></direction>")
            }
            val shown = accidentals(b, bar.fifths)
            val numbers = voiceNumbers(b)
            for ((index, line) in b.lines.withIndex()) {
                if (index > 0) x.append("<backup><duration>${b.length}</duration></backup>")
                for (item in line.items) item(x, item, line.staff, numbers.getValue(line), shown)
            }
            x.append("</measure>")
        }
        x.append("</part></score-partwise>\n")
        return x.toString()
    }

    private fun voiceNumbers(bar: BarLines): Map<Line, Int> {
        val out = IdentityHashMap<Line, Int>()
        for (staff in 1..2) bar.lines.filter { it.staff == staff }.forEachIndexed { i, line -> out[line] = i + 1 + (staff - 1) * 8 }
        return out
    }

    /** Which pieces print an accidental: those whose alteration differs from the key or from earlier in the bar. */
    private fun accidentals(bar: BarLines, fifths: Int): Set<Piece> {
        val shown = Collections.newSetFromMap(IdentityHashMap<Piece, Boolean>())
        for (staff in 1..2) {
            val events = ArrayList<Triple<Int, Int, Piece>>()
            for (line in bar.lines.filter { it.staff == staff }) {
                var cursor = 0
                for (item in line.items) {
                    when (item) {
                        is GraceItem -> item.pieces.forEach { events.add(Triple(cursor, 0, it)) }
                        is NoteItem -> if (item.first) item.pieces.forEach { events.add(Triple(cursor, 1, it)) }
                        else -> {}
                    }
                    cursor += item.ticks
                }
            }
            val state = HashMap<String, Int>()
            for ((_, _, p) in events.sortedWith(compareBy({ it.first }, { it.second }))) {
                val n = p.note
                val step = Spelling.step(n.pitch, n.alter)
                val key = step + Spelling.octave(n.pitch, n.alter)
                val current = state[key] ?: Spelling.keyAlter(step, fifths)
                // A tied continuation prints no sign, so it leaves the bar's state as it was.
                if (p.tieStop) continue
                if (n.alter != current) shown.add(p)
                state[key] = n.alter
            }
        }
        return shown
    }

    private fun item(x: StringBuilder, item: Item, staff: Int, voice: Int, shown: Set<Piece>) {
        when (item) {
            is ForwardItem -> x.append("<forward><duration>${item.ticks}</duration><voice>$voice</voice><staff>$staff</staff></forward>")
            is RestItem -> {
                x.append("<note>")
                x.append(if (item.value == null) "<rest measure=\"yes\"/>" else "<rest/>")
                x.append("<duration>${item.ticks}</duration><voice>$voice</voice>")
                item.value?.let { typeAndDots(x, it); if (it.triplet) timeModification(x) }
                x.append("<staff>$staff</staff>")
                val notations = tuplets(item)
                if (notations.isNotEmpty()) x.append("<notations>").append(notations).append("</notations>")
                x.append("</note>")
            }
            is GraceItem -> item.pieces.sortedBy { it.note.pitch }.forEachIndexed { i, p -> note(x, p, null, i > 0, staff, voice, shown, item) }
            is NoteItem -> item.pieces.sortedBy { it.note.pitch }.forEachIndexed { i, p -> note(x, p, item, i > 0, staff, voice, shown, item) }
        }
    }

    private fun note(x: StringBuilder, p: Piece, item: NoteItem?, chord: Boolean, staff: Int, voice: Int,
                     shown: Set<Piece>, owner: Item) {
        val n = p.note
        x.append("<note dynamics=\"").append(String.format(Locale.ROOT, "%.2f", n.velocity / 90.0 * 100)).append("\">")
        if (item == null) x.append("<grace slash=\"yes\"/>")
        if (chord) x.append("<chord/>")
        x.append("<pitch><step>").append(Spelling.step(n.pitch, n.alter)).append("</step>")
        if (n.alter != 0) x.append("<alter>${n.alter}</alter>")
        x.append("<octave>").append(Spelling.octave(n.pitch, n.alter)).append("</octave></pitch>")
        val tieStop = item != null && (!item.first || p.tieStop)
        val tieStart = item != null && (!item.last || p.tieStart)
        if (item != null) x.append("<duration>${item.value.ticks}</duration>")
        if (tieStop) x.append("<tie type=\"stop\"/>")
        if (tieStart) x.append("<tie type=\"start\"/>")
        x.append("<voice>$voice</voice>")
        if (item != null) typeAndDots(x, item.value) else x.append("<type>eighth</type>")
        val firstValue = item == null || item.first
        if (firstValue && p in shown) x.append("<accidental>${ACCIDENTALS.getValue(n.alter)}</accidental>")
        if (item != null && item.value.triplet) timeModification(x)
        n.stem?.let { x.append("<stem>$it</stem>") }
        x.append("<staff>$staff</staff>")
        val notations = StringBuilder()
        if (tieStop) notations.append("<tied type=\"stop\"/>")
        if (tieStart) notations.append("<tied type=\"start\"/>")
        if (!chord) notations.append(tuplets(owner))
        if (firstValue && n.staccato) notations.append("<articulations><staccato/></articulations>")
        if (firstValue && n.trill) notations.append("<ornaments><trill-mark/></ornaments>")
        if (notations.isNotEmpty()) x.append("<notations>").append(notations).append("</notations>")
        x.append("</note>")
    }

    private fun tuplets(item: Item): String = buildString {
        if (item.tupletStart) append("<tuplet type=\"start\" bracket=\"yes\"/>")
        if (item.tupletStop) append("<tuplet type=\"stop\"/>")
    }

    private fun typeAndDots(x: StringBuilder, v: NoteValue) {
        x.append("<type>${v.type}</type>")
        repeat(v.dots) { x.append("<dot/>") }
    }

    private fun timeModification(x: StringBuilder) =
        x.append("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>")

    private fun escape(text: String) =
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
}

package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.Streams
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VoicesTest {
    private fun arrange(rows: List<IntArray>) = Measures.layout(rows).let { it to Voices.arrange(it) }

    private fun assertLinesFill(bars: List<BarLines>) {
        for (bar in bars) for (line in bar.lines) {
            assertEquals(bar.length, line.items.sumOf { it.ticks }, "bar ${bar.bar} staff ${line.staff} voice ${line.voice}")
        }
    }

    @Test
    fun gapsInTheFirstVoiceBecomeRests() {
        val (_, bars) = arrange(listOf(tok(24, duration = 24)))
        val line = bars[0].lines.single { it.staff == 1 }
        assertTrue(line.items.first() is RestItem)
        assertLinesFill(bars)
    }

    @Test
    fun anEmptyStaffGetsAWholeBarRest() {
        val (_, bars) = arrange(fourFour(1))
        val lower = bars[0].lines.single { it.staff == 2 }
        assertEquals(null, (lower.items.single() as RestItem).value)
    }

    @Test
    fun anEmptyStaffInAPickupGetsWrittenRests() {
        // A whole-bar rest would fill the pickup to a full bar of the time signature in music21.
        val (layout, bars) = arrange(listOf(tok(0)) + fourFour(2).mapIndexed { i, r -> if (i == 0) r.also { it[1] = 25 } else r })
        assertTrue(layout.pickup)
        val lower = bars[0].lines.single { it.staff == 2 }
        assertEquals(listOf("quarter"), lower.items.map { (it as RestItem).value?.type })
        assertEquals(null, (bars[1].lines.single { it.staff == 2 }.items.single() as RestItem).value)
    }

    @Test
    fun aSecondVoiceUsesForwardsNotRests() {
        // Inserted among bar 0's rows: appended after offset 72, offset 48 would start a new bar.
        val rows = fourFour(1).let { it.take(3) + listOf(tok(48, duration = 24, pitch = 55, voice = 2)) + it.drop(3) }
        val (_, bars) = arrange(rows)
        assertEquals(1, bars.size)
        val second = bars[0].lines.single { it.staff == 1 && it.voice == 2 }
        assertTrue(second.items.first() is ForwardItem)
        assertLinesFill(bars)
    }

    @Test
    fun overlappingNotesInOneVoiceMoveToAFreeVoice() {
        val (layout, bars) = arrange(listOf(tok(0, duration = 48, pitch = 60), tok(24, duration = 48, pitch = 64)))
        assertEquals(1, layout.repairs.overlapsMoved)
        assertEquals(2, bars[0].lines.count { it.staff == 1 })
        assertLinesFill(bars)
    }

    @Test
    fun aNoteCrossingTheBarlineIsTied() {
        val (_, bars) = arrange(listOf(tok(72, duration = 48), tok(24, downbeat = 97, duration = 72)))
        val first = bars[0].lines.single { it.staff == 1 }.items.filterIsInstance<NoteItem>().single()
        val second = bars[1].lines.single { it.staff == 1 }.items.filterIsInstance<NoteItem>().first()
        assertTrue(first.pieces.single().tieStart)
        assertTrue(second.pieces.single().tieStop)
        assertLinesFill(bars)
    }

    @Test
    fun aShortenedNoteLosesItsTieIntoTheNextBar() {
        // Voice 1 holds a note from beat 3 across the barline; voices 2..8 are busy for the rest of bar 0,
        // so a later voice-1 note has nowhere to go and the held note is shortened. Its continuation in
        // bar 1 must no longer stop a tie.
        fun note(voice: Int, offset: Int, duration: Int, pitch: Int) = ScoreNote(0, offset, duration, pitch, 0, 5, 1, voice,
            null, grace = false, trill = false, staccato = false, velocity = 64)
        val notes = listOf(note(1, 48, 96, 60)) + (2..8).map { note(it, 48, 48, 60 + 2 * it) } + note(1, 72, 24, 62)
        val layout = Layout(listOf(Bar(96, 0), Bar(96, 0)), notes, false, Repairs())
        val bars = Voices.arrange(layout)
        assertEquals(1, layout.repairs.overlapsShortened)
        val held = bars[0].lines.flatMap { it.items }.filterIsInstance<NoteItem>().flatMap { it.pieces }
            .single { it.note === notes[0] }
        assertEquals(false, held.tieStart)
        val continuation = bars[1].lines.flatMap { it.items }.filterIsInstance<NoteItem>().flatMap { it.pieces }
            .single { it.note === notes[0] }
        assertEquals(false, continuation.tieStop)
        assertLinesFill(bars)
    }

    @Test
    fun fourBeatsOfRestInAFiveFourBarAreTwoHalves() {
        // A whole rest means "the whole bar" to readers (music21 stretches it), so it is kept for 4/4 alone.
        val note = ScoreNote(0, 0, 24, 60, 0, 5, 1, 1, null, grace = false, trill = false, staccato = false, velocity = 64)
        val bars = Voices.arrange(Layout(listOf(Bar(120, 0)), listOf(note), false, Repairs()))
        val rests = bars[0].lines.single { it.staff == 1 }.items.filterIsInstance<RestItem>()
        assertEquals(listOf("half", "half"), rests.map { it.value?.type })
        assertLinesFill(bars)
    }

    @Test
    fun aTiePastTheLastBarIsCutAndCounted() {
        val (layout, bars) = arrange(listOf(tok(72, duration = 48)))
        assertEquals(1, layout.repairs.tiesPastEnd)
        assertLinesFill(bars)
    }

    @Test
    fun threeEighthTripletsGetOneBracket() {
        val (_, bars) = arrange(listOf(tok(0, duration = 8), tok(8, duration = 8, pitch = 62), tok(16, duration = 8, pitch = 64),
                                       tok(24, duration = 72, pitch = 65)))
        val notes = bars[0].lines.single { it.staff == 1 }.items.filterIsInstance<NoteItem>()
        assertEquals(listOf(true, false, false, false), notes.map { it.tupletStart })
        assertEquals(listOf(false, false, true, false), notes.map { it.tupletStop })
    }

    @Test
    fun graceNotesComeBeforeTheirNote() {
        val (_, bars) = arrange(listOf(tok(24, duration = 0, pitch = 62), tok(24, pitch = 60)))
        val items = bars[0].lines.single { it.staff == 1 }.items
        val grace = items.indexOfFirst { it is GraceItem }
        assertTrue(grace >= 0 && items[grace + 1] is NoteItem)
    }

    @Test
    fun randomTokensAlwaysFillEveryLine() {
        val random = Random(5)
        repeat(20) {
            val rows = List(200) {
                tok(offset = random.nextInt(145), duration = random.nextInt(97), pitch = random.nextInt(21, 109),
                    downbeat = if (random.nextInt(8) == 0) random.nextInt(146) else 0, accidental = random.nextInt(7),
                    key = random.nextInt(16), voice = random.nextInt(9), hand = random.nextInt(3),
                    grace = if (random.nextInt(20) == 0) 1 else 0, pad = if (random.nextInt(50) == 0) 0 else 1)
            }
            assertLinesFill(arrange(rows).second)
        }
    }
}

package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.Bd1Streams
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class Bd1LayoutTest {
    private val quarterSlot = Bd1Layout.DURATIONS.indexOf(24)
    private val halfSlot = Bd1Layout.DURATIONS.indexOf(48)
    private val fourQuarters = Bd1Layout.BAR_LENGTHS.indexOf(96)
    private val threeQuarters = Bd1Layout.BAR_LENGTHS.indexOf(72)

    private fun row(pitch: Int, onset: Int, duration: Int = quarterSlot, bar: Int = 0, hand: Int = 0, voice: Int = 0,
                    metre: Int = 0, key: Int = 8, source: Int = 0) = IntArray(Bd1Streams.ROW).also {
        it[Bd1Streams.PITCH] = pitch; it[Bd1Streams.ONSET] = Bd1Layout.ONSETS.indexOf(onset)
        it[Bd1Streams.DURATION] = duration; it[Bd1Streams.BAR] = bar; it[Bd1Streams.HAND] = hand
        it[Bd1Streams.VOICE] = voice; it[Bd1Streams.METRE] = metre; it[Bd1Streams.KEY] = key
        it[Bd1Streams.SOURCE] = source; it[Bd1Streams.VELOCITY] = 64
    }

    @Test
    fun tablesAreVocabsLength() {
        assertEquals(241, Bd1Layout.ONSETS.size)
        assertEquals(161, Bd1Layout.DURATIONS.size)
        assertEquals(241, Bd1Layout.BAR_LENGTHS.size)
        assertEquals(listOf(0, 1, 2, 2), Bd1Layout.ONSETS.take(4))
        assertEquals(144, Bd1Layout.BAR_LENGTHS.last())
    }

    @Test
    fun aBarSlotOpensABarOfThatLengthOnItsRow() {
        val layout = Bd1Layout.layout(listOf(
            row(60, 0, bar = threeQuarters), row(62, 24), row(64, 48),
            row(65, 0, bar = fourQuarters), row(67, 72)))
        assertEquals(listOf(72, 96), layout.bars.map { it.length })
        assertEquals(listOf(0, 0, 0, 1, 1), layout.notes.map { it.measure })
        assertEquals(listOf(0, 24, 48, 0, 72), layout.notes.map { it.offset })
        assertTrue(layout.pickup)
    }

    @Test
    fun aNoteEndingOnTheBarLineJoinsTheSamePitchOpeningTheNext() {
        val layout = Bd1Layout.layout(listOf(
            row(60, 48, duration = halfSlot, bar = fourQuarters), row(64, 0, bar = fourQuarters), row(60, 0), row(67, 48)))
        val c = layout.notes.filter { it.pitch == 60 }
        assertEquals(1, c.size)
        assertEquals(48 + 24, c[0].duration)
    }

    @Test
    fun graceAndRestSlots() {
        val layout = Bd1Layout.layout(listOf(row(62, 0, duration = 0, bar = fourQuarters), row(60, 0),
                                             row(Bd1Streams.PITCH_REST, 0, bar = fourQuarters)))
        assertEquals(listOf(true, false), layout.notes.map { it.grace })
        assertEquals(2, layout.bars.size)
    }

    @Test
    fun metreAndKeyAreVotedPerBar() {
        val sixEight = Bd1Layout.TIME_SIGNATURES.indexOf(6 to 8)
        val threeFour = Bd1Layout.TIME_SIGNATURES.indexOf(3 to 4)
        assertEquals(6 to 8, Bd1Layout.metreOf(listOf(sixEight, threeFour)))
        assertEquals(3 to 4, Bd1Layout.metreOf(listOf(sixEight, threeFour, threeFour)))
        assertNull(Bd1Layout.metreOf(listOf(0, 0)))
        assertEquals(-1, Bd1Layout.keyOf(listOf(7, 9), null))
        assertEquals(1, Bd1Layout.keyOf(listOf(7, 9), 1))
        val layout = Bd1Layout.layout(listOf(row(60, 0, bar = threeQuarters, metre = sixEight, key = 0),
                                             row(62, 0, bar = threeQuarters, metre = sixEight, key = 10)))
        assertEquals(6 to 8, layout.meter)
        assertEquals(listOf(2, 2), layout.bars.map { it.fifths })
    }

    @Test
    fun writesMusicXml() {
        val written = ScoreWriter.write(listOf(row(60, 0, bar = fourQuarters), row(64, 24, hand = 1)), onsets = doubleArrayOf(0.0))
        assertTrue("<step>E</step>" in written.xml)
        assertEquals(1, written.bars)
    }
}

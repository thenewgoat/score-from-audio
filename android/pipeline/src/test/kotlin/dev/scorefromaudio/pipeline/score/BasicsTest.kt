package dev.scorefromaudio.pipeline.score

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class BasicsTest {
    @Test
    fun meterTextIsReadOrRefused() {
        assertEquals(6 to 8, Basics.meter("6/8"))
        assertNull(Basics.meter("Not sure"))
        assertNull(Basics.meter(null))
        assertEquals(72, Basics(6, 8).barLength)
    }

    @Test
    fun fourFourBecomesThreeFourKeepingEveryNoteInPlace() {
        val before = Measures.layout(fourFour(3))
        val after = Rebar.meter(before, 3, 4)
        assertEquals(List(4) { 72 }, after.bars.map { it.length })
        assertFalse(after.pickup)
        assertEquals(before.notes.map { before.position(it) }, after.notes.map { after.position(it) })
        assertEquals(List(12) { it / 3 }, after.notes.map { it.measure })
    }

    @Test
    fun accentedBeatsDecideWhereTheFirstBarStarts() {
        // Nine quarters, loud on the second, fifth and eighth: 3/4 with a one-beat pickup.
        val rows = (0 until 9).map { q -> tok((q % 4) * 24, downbeat = if (q > 0 && q % 4 == 0) 97 else 0, velocity = if (q % 3 == 1) 7 else 2) }
        val after = Rebar.meter(Measures.layout(rows), 3, 4)
        assertTrue(after.pickup)
        assertEquals(listOf(24, 72, 72, 72), after.bars.map { it.length })
    }

    @Test
    fun barsAlreadyOfTheLengthOnlyTakeTheName() {
        val threeFour = (0 until 6).map { q -> tok((q % 3) * 24, downbeat = if (q == 3) 73 else 0) }
        val before = Measures.layout(threeFour)
        val after = Rebar.meter(before, 6, 8)
        assertSame(before.bars, after.bars)
        assertEquals(6 to 8, after.timeSignature(72))
        val xml = ScoreWriter.write(threeFour, basics = Basics(6, 8)).xml
        assertTrue("<beats>6</beats><beat-type>8</beat-type>" in xml)
    }

    @Test
    fun aKeyRespellsBlackKeysAndEveryBar() {
        val rows = listOf(tok(0, pitch = 61, accidental = 3), tok(24, pitch = 62), tok(0, downbeat = 97, pitch = 66, accidental = 3))
        val after = Rebar.key(Measures.layout(rows), -3)
        assertEquals(listOf(-3, -3), after.bars.map { it.fifths })
        assertEquals(listOf(-1, 0, -1), after.notes.map { it.alter })
    }

    @Test
    fun anchorsMedianEachPositionAndDropBackwardSteps() {
        // Two notes at the first beat, then one heard before them (a mistake) that would run time backwards.
        val rows = listOf(tok(0, pitch = 60), tok(0, pitch = 64), tok(24), tok(48), tok(72))
        val onsets = doubleArrayOf(1.0, 1.2, 1.6, 0.9, 2.6)
        val anchors = Timing.anchors(Measures.layout(rows), onsets)
        assertEquals(listOf(0.0, 1.0, 3.0), anchors.map { it.quarter })
        assertEquals(1.1, anchors[0].seconds, 1e-9)
    }

    @Test
    fun theTempoIsTheMedianPace() {
        val steady = List(16) { Anchor(it.toDouble(), it * 0.625) }
        assertEquals(96, Timing.bpm(steady))
        assertEquals(60, Timing.bpm(listOf(Anchor(0.0, 0.0), Anchor(2.0, 2.0))), "short pieces use first to last")
        assertNull(Timing.bpm(listOf(Anchor(0.0, 0.0))))
    }
}

package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.BdFixture
import dev.scorefromaudio.pipeline.Fixtures
import org.w3c.dom.Element
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScoreWriterTest {
    /** What a strict notation program checks: every voice fills its bar, and nothing runs past it. */
    private fun validate(xml: String, name: String) {
        val factory = DocumentBuilderFactory.newInstance()
        factory.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false)
        val doc = factory.newDocumentBuilder().parse(xml.byteInputStream())
        var barLength = 0
        val measures = doc.getElementsByTagName("measure")
        assertTrue(measures.length > 0, name)
        for (m in 0 until measures.length) {
            val measure = measures.item(m) as Element
            val time = measure.getElementsByTagName("time")
            if (time.length > 0) {
                val t = time.item(0) as Element
                val beats = t.getElementsByTagName("beats").item(0).textContent.toInt()
                val type = t.getElementsByTagName("beat-type").item(0).textContent.toInt()
                barLength = beats * 96 / type
            }
            val implicit = measure.getAttribute("implicit") == "yes"
            var cursor = 0
            var lineEnd = -1
            val staffOfVoice = HashMap<Int, Int>()
            fun sameStaff(child: Element) {
                val voice = child.getElementsByTagName("voice").item(0)?.textContent?.toInt() ?: return
                val staff = child.getElementsByTagName("staff").item(0).textContent.toInt()
                assertEquals(staff, staffOfVoice.getOrPut(voice) { staff }, "$name bar $m: voice $voice on both staves")
            }
            val children = measure.childNodes
            for (c in 0 until children.length) {
                val child = children.item(c) as? Element ?: continue
                fun duration() = child.getElementsByTagName("duration").item(0)?.textContent?.toInt() ?: 0
                when (child.tagName) {
                    "note" -> {
                        val chord = child.getElementsByTagName("chord").length > 0
                        val grace = child.getElementsByTagName("grace").length > 0
                        val voice = child.getElementsByTagName("voice").item(0).textContent.toInt()
                        assertTrue(voice >= 1, "$name bar $m voice $voice")
                        sameStaff(child)
                        if (!chord && !grace) cursor += duration()
                    }
                    "forward" -> { sameStaff(child); cursor += duration() }
                    "backup" -> {
                        if (lineEnd < 0) lineEnd = cursor else assertEquals(lineEnd, cursor, "$name bar $m: voices differ")
                        cursor -= duration()
                        assertEquals(0, cursor, "$name bar $m: backup must return to the bar start")
                    }
                }
            }
            if (lineEnd >= 0) assertEquals(lineEnd, cursor, "$name bar $m: last voice")
            if (implicit) assertTrue(cursor in 1..barLength, "$name pickup bar $m") else assertEquals(barLength, cursor, "$name bar $m")
        }
    }

    private fun save(name: String, xml: String) {
        Fixtures.writerOut.mkdirs()
        File(Fixtures.writerOut, "$name.musicxml").writeText(xml)
    }

    @Test
    fun theHeldOutPiecesWriteValidScores() {
        for (piece in BdFixture.PIECES) {
            val tokens = BdFixture.load(piece).tokens.map { it.toIntArray() }
            val written = ScoreWriter.write(tokens, piece)
            println("$piece: ${written.bars} bars, repairs ${written.repairs}")
            validate(written.xml, piece)
            save(piece, written.xml)
        }
    }

    @Test
    fun randomTokensWriteValidScores() {
        val random = Random(9)
        repeat(10) { k ->
            val rows = List(300) {
                tok(offset = random.nextInt(145), duration = random.nextInt(97), pitch = random.nextInt(21, 109),
                    downbeat = if (random.nextInt(6) == 0) random.nextInt(1, 146) else 0,
                    accidental = random.nextInt(7), key = random.nextInt(16), voice = random.nextInt(9),
                    stem = random.nextInt(4), hand = random.nextInt(3), grace = if (random.nextInt(25) == 0) 1 else 0,
                    staccato = random.nextInt(2), trill = if (random.nextInt(30) == 0) 1 else 0,
                    pad = if (random.nextInt(60) == 0) 0 else 1)
            }
            val written = ScoreWriter.write(rows, "random $k")
            validate(written.xml, "random $k")
            save("random_$k", written.xml)
        }
    }

    @Test
    fun accidentalsFollowTheKeyAndTheBar() {
        // G major: F sharp needs no sign; F natural does; the second F sharp after it needs one again.
        val rows = listOf(tok(0, pitch = 66, accidental = 3, key = 8), tok(24, pitch = 65, accidental = 2, key = 8),
                          tok(48, pitch = 66, accidental = 3, key = 8), tok(72, pitch = 66, accidental = 3, key = 8))
        val xml = ScoreWriter.write(rows).xml
        assertEquals(listOf("natural", "sharp"), Regex("<accidental>(\\w+)</accidental>").findAll(xml).map { it.groupValues[1] }.toList())
    }

    @Test
    fun anAccidentalIsRestatedAfterATiedNote() {
        // G major: F natural tied into bar 2 prints no sign there, so the next F natural in bar 2 still needs one.
        val rows = listOf(tok(72, duration = 48, pitch = 65, accidental = 2, key = 8),
                          tok(24, downbeat = 97, pitch = 65, accidental = 2, key = 8))
        val xml = ScoreWriter.write(rows).xml
        val bar2 = Regex("<measure number=\"2\".*?</measure>").find(xml)!!.value
        assertEquals(listOf("natural"), Regex("<accidental>(\\w+)</accidental>").findAll(bar2).map { it.groupValues[1] }.toList())
        validate(xml, "tied accidental")
    }

    @Test
    fun aPickupIsImplicitAndNumberedZero() {
        val rows = listOf(tok(0)) + fourFour(2).mapIndexed { i, r -> if (i == 0) r.also { it[1] = 25 } else r }
        val xml = ScoreWriter.write(rows).xml
        assertTrue("<measure number=\"0\" implicit=\"yes\">" in xml)
        validate(xml, "pickup")
    }

    @Test
    fun everyMeterAndKeyWritesValidScores() {
        val meters = listOf(2 to 4, 3 to 4, 4 to 4, 6 to 8)
        for (piece in BdFixture.PIECES) {
            val tokens = BdFixture.load(piece).tokens.map { it.toIntArray() }
            for ((beats, type) in meters) {
                val written = ScoreWriter.write(tokens, piece, Basics(beats, type, fifths = -3, bpm = 80))
                validate(written.xml, "$piece in $beats/$type")
                assertEquals(beats to type, written.meter)
                assertEquals(-3, written.fifths)
            }
        }
    }

    @Test
    fun theMeasuredTempoIsWrittenAndAnAskedOneWins() {
        val onsets = DoubleArray(8) { it * 0.5 }
        val measured = ScoreWriter.write(fourFour(2), onsets = onsets)
        assertEquals(120, measured.bpm)
        assertTrue("<sound tempo=\"120\"/>" in measured.xml)
        assertEquals(8, measured.sync.size)
        val asked = ScoreWriter.write(fourFour(2), basics = Basics(bpm = 72), onsets = onsets)
        assertTrue("<per-minute>72</per-minute>" in asked.xml && "<sound tempo=\"72\"/>" in asked.xml)
        assertEquals(null, ScoreWriter.write(fourFour(2)).bpm, "no recording, no measured tempo")
    }
}

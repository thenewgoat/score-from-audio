package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.Bd1Streams

/**
 * [meter] is the usual bar's time signature and [fifths] the first bar's key. [bpm] is the tempo written into the score:
 * the one asked for, else the one measured from the recording, else null (no mark; players assume 120).
 * [sync] ties score positions to times in the recording; empty without onsets.
 */
class WrittenScore(
    val xml: String, val repairs: Repairs, val bars: Int,
    val meter: Pair<Int, Int>, val fifths: Int, val bpm: Int?, val sync: List<Anchor>,
)

/** Notation tokens -> MusicXML, with a count of everything that had to be repaired to make it legal. */
object ScoreWriter {
    /**
     * [tokens] are Beyer & Dai's 14-wide rows or bd1's [Bd1Streams.ROW]-wide ones; [onsets] is each input note's onset
     * in the recording, in seconds, in that model's own note order (NotationInput.onsets, Bd1Input.onsets).
     */
    fun write(tokens: List<IntArray>, title: String = "Transcription", basics: Basics = Basics(),
              onsets: DoubleArray? = null): WrittenScore {
        val repairs = Repairs()
        var layout = if (isBd1(tokens)) Bd1Layout.layout(tokens, repairs) else Measures.layout(tokens, repairs)
        if (basics.beats != null && basics.beatType != null) layout = Rebar.meter(layout, basics.beats, basics.beatType)
        if (basics.fifths != null) layout = Rebar.key(layout, basics.fifths)
        val sync = onsets?.let { Timing.anchors(layout, it) }.orEmpty()
        val bpm = basics.bpm ?: Timing.bpm(sync)
        val bars = Voices.arrange(layout)
        return WrittenScore(MusicXml.write(layout, bars, title, bpm), repairs, layout.bars.size,
            layout.timeSignature(layout.usualLength), layout.bars.first().fifths, bpm, sync)
    }

    fun isBd1(tokens: List<IntArray>): Boolean = tokens.firstOrNull()?.size == Bd1Streams.ROW
}

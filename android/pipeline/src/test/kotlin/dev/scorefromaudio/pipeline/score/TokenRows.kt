package dev.scorefromaudio.pipeline.score

import dev.scorefromaudio.pipeline.Streams

/** A token row as the notation model would emit it; positions in 24ths of a quarter. */
fun tok(offset: Int, duration: Int = 24, pitch: Int = 60, downbeat: Int = 0, accidental: Int = 5, key: Int = 7,
        voice: Int = 1, stem: Int = 3, hand: Int = 0, grace: Int = 0, staccato: Int = 0, trill: Int = 0,
        velocity: Int = 4, pad: Int = 1): IntArray {
    val row = IntArray(14)
    row[Streams.OFFSET] = offset; row[Streams.DOWNBEAT] = downbeat; row[Streams.DURATION] = duration
    row[Streams.PITCH] = pitch; row[Streams.ACCIDENTAL] = accidental; row[Streams.KEYSIGNATURE] = key
    row[Streams.VELOCITY] = velocity; row[Streams.GRACE] = grace; row[Streams.TRILL] = trill
    row[Streams.STACCATO] = staccato; row[Streams.VOICE] = voice; row[Streams.STEM] = stem
    row[Streams.HAND] = hand; row[Streams.PAD] = pad
    if (pad == 0) for (j in 0 until 13) row[j] = -1
    return row
}

/** Four quarter notes per bar for `bars` bars of 4/4. */
fun fourFour(bars: Int): List<IntArray> = (0 until bars).flatMap { b ->
    (0 until 4).map { q -> tok(q * 24, downbeat = if (b > 0 && q == 0) 97 else 0) }
}

package dev.scorefromaudio.pipeline

import kotlin.math.ln

/**
 * Beyer & Dai's input streams as bucket indices, in their note order.
 *
 * Computed in float32 as their torch code computes them: onset gaps from the
 * previous note (the first from time zero) and durations are log-squashed and
 * bucketed into 200, pitch into 128, velocity into 8. Pedals are not notes to
 * the notation model and are dropped here.
 */
class NotationInput(
    val notes: List<Note>,
    val onset: IntArray,
    val duration: IntArray,
    val pitch: IntArray,
    val velocity: IntArray,
) {
    val size: Int get() = notes.size

    /** Each note's onset in seconds: the notation model writes one token row per note, in this order. */
    val onsets: DoubleArray get() = DoubleArray(size) { notes[it].start }

    companion object {
        private val onsetScale = ln(33.0).toFloat()
        private val durationScale = ln(65.0).toFloat()

        private fun bucket(v: Float, lo: Float, hi: Float, buckets: Int) =
            ((v - lo) / (hi - lo) * buckets).toInt().coerceIn(0, buckets - 1)

        fun encode(detected: List<Note>): NotationInput {
            val notes = detected.filter { it.pitch >= 0 }
                .sortedWith(compareBy({ it.start }, { it.pitch }, { it.end - it.start }))
            val n = notes.size
            val onset = IntArray(n)
            var previous = 0f
            for (i in 0 until n) {
                val start = notes[i].start.toFloat()
                val gap = start - previous
                previous = start
                onset[i] = bucket(ln(4f * gap + 1f) * 4f / onsetScale, 0f, 4f, 200)
            }
            // Their tokenizer takes end - start in double and only then converts to
            // float32; onsets are converted first and differenced in float32.
            val duration = IntArray(n) {
                val d = (notes[it].end - notes[it].start).toFloat()
                bucket(ln(4f * d + 1f) * 4f / durationScale, 0f, 4f, 200)
            }
            val pitch = IntArray(n) { bucket(notes[it].pitch.toFloat(), 0f, 127f, 128) }
            val velocity = IntArray(n) { bucket(notes[it].velocity.toFloat(), 0f, 127f, 8) }
            return NotationInput(notes, onset, duration, pitch, velocity)
        }
    }
}

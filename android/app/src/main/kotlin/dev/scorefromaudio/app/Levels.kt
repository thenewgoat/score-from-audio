package dev.scorefromaudio.app

import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt

enum class LevelHint(val text: String) {
    LISTENING("Listening…"),
    QUIET("Too quiet: move the phone closer to the piano"),
    GOOD("Sound level good"),
    CLIPPING("Too loud: move the phone further away"),
}

/**
 * The live level meter's arithmetic, apart from the microphone so it can be tested. Samples arrive in blocks of
 * [blockSize]; each block's loudness (RMS, as 0..1 over -60..0 dBFS) is kept for the meter's bars, and its peak for
 * the hint: clipping if any block in the last 1.5 s touched full scale, too quiet if nothing in the last 3 s rose
 * above -30 dBFS.
 */
class Levels(sampleRate: Int, private val bars: Int = 36) {
    val blockSize = sampleRate / 10
    private val loudness = ArrayDeque<Float>()
    private val peaks = ArrayDeque<Float>()
    private var blocks = 0

    fun add(samples: ShortArray, count: Int = samples.size) {
        var sum = 0.0
        var peak = 0
        for (i in 0 until count) {
            val v = samples[i].toInt()
            sum += v.toDouble() * v
            peak = maxOf(peak, abs(v))
        }
        val rms = sqrt(sum / maxOf(count, 1)) / 32768.0
        val db = if (rms > 0) 20 * log10(rms) else -120.0
        push(loudness, ((db + 60) / 60).toFloat().coerceIn(0f, 1f), bars)
        push(peaks, peak / 32768f, 30)
        blocks++
    }

    /** The newest [bars] loudness values, oldest first. */
    fun bars(): List<Float> = loudness.toList()

    fun hint(): LevelHint = when {
        blocks < 10 -> LevelHint.LISTENING
        peaks.takeLast(15).any { it >= 0.98f } -> LevelHint.CLIPPING
        peaks.all { it < 0.0316f } -> LevelHint.QUIET
        else -> LevelHint.GOOD
    }

    private fun push(queue: ArrayDeque<Float>, value: Float, limit: Int) {
        queue.addLast(value)
        while (queue.size > limit) queue.removeFirst()
    }
}

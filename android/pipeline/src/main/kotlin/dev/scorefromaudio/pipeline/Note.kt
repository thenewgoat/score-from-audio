package dev.scorefromaudio.pipeline

/** A detected note. Times in seconds. Pedals are notes too: sustain is pitch -64, una corda -67. */
data class Note(val start: Double, val end: Double, val pitch: Int, val velocity: Int)

/** Decoded audio, one array per channel, samples in [-1, 1). */
class Audio(val channels: Array<FloatArray>, val sampleRate: Int) {
    val frames: Int get() = channels.firstOrNull()?.size ?: 0
    val seconds: Double get() = frames.toDouble() / sampleRate
}

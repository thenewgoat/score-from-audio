package dev.scorefromaudio.pipeline

import java.nio.FloatBuffer

object Transkun {
    const val FS = 44_100
    const val HOP = 1024
    const val FRAMES = 691
    const val CONTEXT = 256
    val TRACKS: IntArray = intArrayOf(-64, -67) + (21..108).toList().toIntArray()
}

/** Transkun's network split as exported: backbone + scorer, and the two interval heads. */
interface TranskunModel {
    /**
     * features [691][229][6] -> interval scores [end][begin][track] and context [track][frame][256].
     * The output may live in the model's own memory: it is valid only until the next core() call.
     */
    fun core(features: FloatArray): CoreOutput
    /** attr [n][768] (context at begin, at end, and their product) -> per-interval attributes. */
    fun heads(attr: FloatArray, n: Int): HeadsOutput
}

/** Read with absolute get(index) only; the buffers' positions are not meaningful. */
class CoreOutput(val score: FloatBuffer, val ctx: FloatBuffer)

class HeadsOutput(val velocity: IntArray, val ofValue: FloatArray, val ofPresence: BooleanArray)

package dev.scorefromaudio.pipeline

/** Beyer & Dai's model as exported: an encoder over a window of notes, and one cached decoder step. */
interface NotationModel {
    fun encode(input: NotationInput, from: Int, to: Int): Encoded
    fun newCache(): Cache
    /** Feed one token row at position `cache.length`; returns the next step's logits: 13 streams, then pad (size 1). */
    fun step(encoded: Encoded, row: IntArray, cache: Cache): Array<FloatArray>
}

interface Encoded : AutoCloseable

interface Cache : AutoCloseable {
    val length: Int
}

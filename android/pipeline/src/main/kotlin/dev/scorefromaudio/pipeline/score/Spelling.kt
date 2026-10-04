package dev.scorefromaudio.pipeline.score

/** Pitch spelling from a MIDI number and the model's accidental token, and key-signature arithmetic. */
object Spelling {
    class Spelled(val alter: Int, val overridden: Boolean)

    private val WHITE = setOf(0, 2, 4, 5, 7, 9, 11)
    private val STEPS = mapOf(0 to "C", 2 to "D", 4 to "E", 5 to "F", 7 to "G", 9 to "A", 11 to "B")
    private const val SHARPS = "FCGDAEB"
    private const val FLATS = "BEADGCF"

    /** The token's alteration when it names a real spelling of `pitch`; otherwise the key's default. */
    fun alter(pitch: Int, accidental: Int, fifths: Int): Spelled {
        val wanted = when (accidental) { 0 -> -2; 1 -> -1; 2 -> 0; 3 -> 1; 4 -> 2; else -> null }
        if (wanted != null && Math.floorMod(pitch - wanted, 12) in WHITE) return Spelled(wanted, false)
        val default = if (Math.floorMod(pitch, 12) in WHITE) 0 else if (fifths >= 0) 1 else -1
        return Spelled(default, wanted != null)
    }

    fun step(pitch: Int, alter: Int): String = STEPS.getValue(Math.floorMod(pitch - alter, 12))

    fun octave(pitch: Int, alter: Int): Int = Math.floorDiv(pitch - alter, 12) - 1

    fun keyAlter(step: String, fifths: Int): Int = when {
        fifths > 0 && SHARPS.indexOf(step[0]) in 0 until fifths -> 1
        fifths < 0 && FLATS.indexOf(step[0]) in 0 until -fifths -> -1
        else -> 0
    }
}

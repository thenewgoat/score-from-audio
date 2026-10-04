package dev.scorefromaudio.pipeline

/** Beyer & Dai's 13 output streams plus pad, in the order their decoder chooses them. */
object Streams {
    const val OFFSET = 0
    const val DOWNBEAT = 1
    const val DURATION = 2
    const val PITCH = 3
    const val ACCIDENTAL = 4
    const val KEYSIGNATURE = 5
    const val VELOCITY = 6
    const val GRACE = 7
    const val TRILL = 8
    const val STACCATO = 9
    const val VOICE = 10
    const val STEM = 11
    const val HAND = 12
    const val PAD = 13

    val NAMES = listOf("offset", "downbeat", "duration", "pitch", "accidental", "keysignature", "velocity",
                       "grace", "trill", "staccato", "voice", "stem", "hand")
    val SIZES = intArrayOf(145, 146, 97, 128, 7, 16, 8, 2, 2, 2, 9, 4, 3)

    /** The start token: every stream a zero vector, pad 0. */
    fun start(): IntArray = IntArray(14) { -1 }.also { it[PAD] = 0 }
}

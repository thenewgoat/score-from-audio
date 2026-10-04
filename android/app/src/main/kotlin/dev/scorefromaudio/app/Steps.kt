package dev.scorefromaudio.app

/** The four steps a person sees while a score is made. The last happens when the score is first shown. */
enum class Step(val label: String) {
    LISTENING("Listening for notes"),
    BEAT("Finding the beat"),
    WRITING("Writing the notes"),
    LAYOUT("Laying out the pages"),
}

object Steps {
    fun of(stage: Stage): Step = when (stage) {
        Stage.FETCHING, Stage.DECODING, Stage.NOTES -> Step.LISTENING
        Stage.NOTATION -> Step.BEAT
        Stage.WRITING -> Step.WRITING
    }

    fun of(stageName: String): Step = Stage.entries.firstOrNull { it.name == stageName }?.let { of(it) } ?: Step.LISTENING

    /** The whole job's progress, each stage given its rough share of the time the phone takes. */
    fun overall(stage: Stage, fraction: Float): Float {
        val (start, end) = when (stage) {
            Stage.FETCHING -> 0f to 0.10f
            Stage.DECODING -> 0.10f to 0.15f
            Stage.NOTES -> 0.15f to 0.60f
            Stage.NOTATION -> 0.60f to 0.93f
            Stage.WRITING -> 0.93f to 0.97f
        }
        return start + (end - start) * fraction.coerceIn(0f, 1f)
    }
}

/** Why a job failed, in words for the person who recorded it. */
object Reasons {
    const val INTERRUPTED = "Stopped unexpectedly"

    fun plain(meta: JobMeta): String {
        val error = meta.error.orEmpty()
        return when {
            error == JobRunner.STOPPED -> "You stopped it before it finished."
            error == INTERRUPTED -> "The app was closed before your score was finished."
            error == "No notes were detected" -> "No piano notes were heard. Try recording closer to the piano, in a quieter room."
            error.startsWith("Ran out of memory") -> "The phone ran out of memory. Close other apps and try again, or use a shorter recording."
            meta.stage == Stage.FETCHING.name && error.isNotEmpty() -> error
            meta.stage == Stage.DECODING.name -> if (error.startsWith("This ")) error else "This audio couldn't be read."
            else -> "Something went wrong while ${Steps.of(meta.stage).label.lowercase()}."
        }
    }
}

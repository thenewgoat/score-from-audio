package dev.scorefromaudio.pipeline.score

import kotlinx.serialization.Serializable

/** Every place the writer had to change what the model said to make a legal score. Stored in a job's meta.json. */
@Serializable
data class Repairs(
    var droppedByModel: Int = 0,
    var offsetsClamped: Int = 0,
    var barLengthsRounded: Int = 0,
    var overlapsMoved: Int = 0,
    var overlapsShortened: Int = 0,
    var spellingsFixed: Int = 0,
    var tiesPastEnd: Int = 0,
) {
    val total: Int
        get() = droppedByModel + offsetsClamped + barLengthsRounded + overlapsMoved + overlapsShortened +
            spellingsFixed + tiesPastEnd
}

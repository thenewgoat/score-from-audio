package dev.scorefromaudio.app

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StepsTest {
    private fun failed(stage: Stage, error: String) = JobMeta("id", "t", 0, "failed", stage.name, error)

    @Test
    fun stagesFallIntoTheFourStepsInOrder() {
        assertEquals(Step.LISTENING, Steps.of(Stage.NOTES))
        assertEquals(Step.BEAT, Steps.of(Stage.NOTATION))
        assertEquals(Step.WRITING, Steps.of(Stage.WRITING))
        val points = Stage.entries.flatMap { listOf(Steps.overall(it, 0f), Steps.overall(it, 1f)) }
        assertEquals(points.sorted(), points, "overall progress never runs backwards")
        assertTrue(points.last() < 1f, "the layout step is left")
    }

    @Test
    fun failuresAreToldPlainly() {
        assertEquals("No piano notes were heard. Try recording closer to the piano, in a quieter room.",
            Reasons.plain(failed(Stage.NOTES, "No notes were detected")))
        assertTrue(Reasons.plain(failed(Stage.NOTATION, "Ran out of memory while writing notation (peak 900 MB)")).startsWith("The phone ran out of memory"))
        assertEquals("This file has no audio track", Reasons.plain(failed(Stage.DECODING, "This file has no audio track")))
        assertEquals("This audio couldn't be read.", Reasons.plain(failed(Stage.DECODING, "IllegalStateException")))
        assertEquals("Something went wrong while finding the beat.", Reasons.plain(failed(Stage.NOTATION, "ORT error 6")))
        assertEquals("You stopped it before it finished.", Reasons.plain(failed(Stage.NOTES, JobRunner.STOPPED)))
    }
}

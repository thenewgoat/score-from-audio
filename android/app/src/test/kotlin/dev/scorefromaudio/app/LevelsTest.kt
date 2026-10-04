package dev.scorefromaudio.app

import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals

class LevelsTest {
    private fun tone(amplitude: Double, n: Int) = ShortArray(n) { (amplitude * 32767 * sin(2 * PI * 440 * it / 44_100.0)).toInt().toShort() }

    private fun after(amplitude: Double, seconds: Int = 3): Levels {
        val levels = Levels(44_100)
        repeat(seconds * 10) { levels.add(tone(amplitude, levels.blockSize)) }
        return levels
    }

    @Test
    fun theHintFollowsTheLevel() {
        assertEquals(LevelHint.LISTENING, Levels(44_100).also { it.add(tone(0.3, 4410)) }.hint())
        assertEquals(LevelHint.GOOD, after(0.3).hint())
        assertEquals(LevelHint.QUIET, after(0.01).hint())
        assertEquals(LevelHint.CLIPPING, after(1.0).hint())
    }

    @Test
    fun barsKeepTheNewestAndScaleToOne() {
        val levels = after(0.5, seconds = 5)
        assertEquals(36, levels.bars().size)
        levels.bars().forEach { assert(it in 0.8f..1f) { "$it" } }
    }
}

package dev.scorefromaudio.pipeline

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class NpyTest {
    @Test
    fun readsTheFrontendFixture() {
        val array = Fixtures.stream("frontend_features.npy").use { Npy.read(it) }
        assertContentEquals(intArrayOf(88, 229, 6), array.shape)
        assertEquals(88 * 229 * 6, array.floats!!.size)
    }
}

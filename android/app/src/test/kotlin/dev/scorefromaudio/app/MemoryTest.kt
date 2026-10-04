package dev.scorefromaudio.app

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class MemoryTest {
    @Test
    fun readsResidentMemoryFromProcStatus() {
        val status = """
            Name:	dev.scorefromaudio.app
            VmPeak:	 9876543 kB
            VmHWM:	 2400000 kB
            VmRSS:	 2359296 kB
            Threads:	31
        """.trimIndent()
        assertEquals(2304, Memory.residentMb(status))
    }

    @Test
    fun noResidentLineGivesNull() {
        assertNull(Memory.residentMb("Name:\tx\nVmRSS:\tlots kB"))
        assertNull(Memory.residentMb(""))
    }
}

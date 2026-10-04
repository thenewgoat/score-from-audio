package dev.scorefromaudio.app

/**
 * The process's resident memory, from the text of /proc/self/status. Resident memory counts
 * the runtime's native allocations too, which the Java heap and the native-heap counters miss.
 * VmRSS rather than VmHWM: the high-water mark is the process's lifetime peak, so a later
 * stage would inherit an earlier stage's peak.
 */
object Memory {
    /** VmRSS in MB, or null when the text has no such line. */
    fun residentMb(status: String): Long? = kb(status, "VmRSS")?.let { it / 1024 }

    private fun kb(status: String, field: String): Long? =
        status.lineSequence().firstOrNull { it.startsWith("$field:") }
            ?.substringAfter(':')?.trim()?.substringBefore(' ')?.toLongOrNull()
}

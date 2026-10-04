package dev.scorefromaudio.app

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/** Words and numbers as the screens show them. */
object Format {
    private val MAJOR = listOf("C♭", "G♭", "D♭", "A♭", "E♭", "B♭", "F", "C", "G", "D", "A", "E", "B", "F♯", "C♯")
    private val MINOR = listOf("A♭", "E♭", "B♭", "F", "C", "G", "D", "A", "E", "B", "F♯", "C♯", "G♯", "D♯", "A♯")

    /** A key signature by both keys it serves: "E♭ major / C minor". */
    fun key(fifths: Int): String = "${MAJOR[fifths.coerceIn(-7, 7) + 7]} major / ${MINOR[fifths.coerceIn(-7, 7) + 7]} minor"

    fun majorKey(fifths: Int): String = "${MAJOR[fifths.coerceIn(-7, 7) + 7]} major"

    fun length(seconds: Double): String {
        val s = seconds.toInt()
        return "%d:%02d".format(s / 60, s % 60)
    }

    fun clock(ms: Long): String = length(ms / 1000.0)

    /** "Today", "Yesterday", or the day and month. */
    fun day(millis: Long, zone: ZoneId = ZoneId.systemDefault(), today: LocalDate = LocalDate.now(zone)): String {
        val date = Instant.ofEpochMilli(millis).atZone(zone).toLocalDate()
        return when (date) {
            today -> "Today"
            today.minusDays(1) -> "Yesterday"
            else -> date.format(DateTimeFormatter.ofPattern(if (date.year == today.year) "d MMM" else "d MMM yyyy", Locale.getDefault()))
        }
    }

    /** A new recording's name, until the person gives it another: "Recording 28 Sep, 22:05". */
    fun recordingTitle(millis: Long, zone: ZoneId = ZoneId.systemDefault()): String =
        "Recording " + Instant.ofEpochMilli(millis).atZone(zone).format(DateTimeFormatter.ofPattern("d MMM, HH:mm", Locale.getDefault()))

    /** What the library says about a finished score: "Today · 3:12 · 4/4". */
    fun summary(meta: JobMeta): String = listOfNotNull(day(meta.createdAt), length(meta.seconds), meta.meter).joinToString(" · ")

    /** The score's header line: "4/4 · C major / A minor · 96 bpm". */
    fun basics(meter: String?, fifths: Int?, bpm: Int?): String =
        listOfNotNull(meter, fifths?.let { key(it) }, bpm?.let { "$it bpm" }).joinToString(" · ")
}

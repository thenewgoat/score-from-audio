package dev.scorefromaudio.app

import java.time.LocalDate
import java.time.ZoneOffset
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertEquals

class FormatTest {
    private val utc = ZoneOffset.UTC
    private val sep28 = LocalDate.of(2026, 9, 28).atStartOfDay(utc).toInstant().toEpochMilli() + 22 * 3_600_000L + 5 * 60_000L

    @Test
    fun keysAreNamedBothWays() {
        assertEquals("C major / A minor", Format.key(0))
        assertEquals("E♭ major / C minor", Format.key(-3))
        assertEquals("F♯ major / D♯ minor", Format.key(6))
    }

    @Test
    fun daysAreRelativeWhenRecent() {
        Locale.setDefault(Locale.US)
        val today = LocalDate.of(2026, 9, 28)
        assertEquals("Today", Format.day(sep28, utc, today))
        assertEquals("Yesterday", Format.day(sep28, utc, today.plusDays(1)))
        assertEquals("28 Sep", Format.day(sep28, utc, today.plusDays(5)))
        assertEquals("28 Sep 2026", Format.day(sep28, utc, today.plusYears(1)))
        assertEquals("Recording 28 Sep, 22:05", Format.recordingTitle(sep28, utc))
    }

    @Test
    fun lengthsAndHeaders() {
        assertEquals("3:12", Format.length(192.4))
        assertEquals("4/4 · C major / A minor · 96 bpm", Format.basics("4/4", 0, 96))
        assertEquals("3/4", Format.basics("3/4", null, null))
    }
}

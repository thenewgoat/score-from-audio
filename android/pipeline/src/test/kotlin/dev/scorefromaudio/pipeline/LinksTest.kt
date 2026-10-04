package dev.scorefromaudio.pipeline

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class LinksTest {
    @Test
    fun urlInsideText() {
        assertEquals("https://youtu.be/abc123?si=x", Links.extractUrl("Watch this! https://youtu.be/abc123?si=x 🎹"))
        assertEquals("https://www.instagram.com/reel/CDUMkliABpa/", Links.extractUrl("https://www.instagram.com/reel/CDUMkliABpa/."))
        assertEquals("https://youtu.be/abc", Links.extractUrl("(https://youtu.be/abc)"))
        assertNull(Links.extractUrl("no link here"))
        assertEquals("https://youtu.be/abc123", Links.extractUrl("https://youtu.be/abc123 🎹"))
        assertEquals("https://youtu.be/abc123", Links.extractUrl("[https://youtu.be/abc123]"))
        assertEquals("https://youtu.be/abc123", Links.extractUrl("https://youtu.be/abc123 extra"))
        assertEquals("https://youtu.be/aaa", Links.extractUrl("See https://youtu.be/aaa and also https://youtu.be/bbb"))
    }

    @Test
    fun acceptsTheSupportedShapes() {
        for (u in listOf("https://www.youtube.com/watch?v=jNQXAC9IVRw", "https://youtube.com/shorts/abc", "https://m.youtube.com/watch?v=x",
                         "https://music.youtube.com/watch?v=x", "https://www.youtube.com/live/abc", "https://youtu.be/jNQXAC9IVRw",
                         "https://www.youtube.com/watch?list=x&v=abc")) {
            assertEquals(Site.YOUTUBE, Links.site(u), u)
        }
        for (u in listOf("https://www.instagram.com/reel/CDUMkliABpa/", "https://instagram.com/reels/Cop84x6u7CP/",
                         "https://www.instagram.com/p/abc/", "https://www.instagram.com/tv/abc/",
                         "https://www.instagram.com/someone/reel/CDUMkliABpa/", "https://www.instagram.com/someone/p/CDUMkliABpa/?igsh=x",
                         "https://www.instagram.com/someone/reels/Cop84x6u7CP/")) {
            assertEquals(Site.INSTAGRAM, Links.site(u), u)
        }
    }

    @Test
    fun rejectsEverythingElse() {
        for (u in listOf("https://vimeo.com/1", "https://www.youtube.com/", "https://www.youtube.com/@channel", "https://www.instagram.com/someone/",
                         "https://evil.com/youtube.com/watch?v=x", "https://youtube.com.evil.com/watch?v=x", "ftp://youtu.be/x", "not a url",
                         "https://www.youtube.com/watch?prev=1", "https://www.instagram.com/reel/",
                         // yt-dlp's Instagram extractor does not take share links, so neither do we.
                         "https://www.instagram.com/share/reel/BAabc123/", "https://www.instagram.com/share/p/BAabc123/",
                         "https://www.instagram.com/someone/stories/123/", "https://www.instagram.com/reels/audio/123/")) {
            assertNull(Links.site(u), u)
        }
    }

    @Test
    fun errorMessages() {
        val cases = mapOf(
            "ERROR: [youtube] x: Unable to download webpage: <urlopen error [Errno -3] Temporary failure in name resolution>" to
                "Couldn't reach the site. Check your connection and try again.",
            "ERROR: Read timed out." to "Couldn't reach the site. Check your connection and try again.",
            "ERROR: [youtube] x: Private video. Sign in if you've been granted access to this video" to
                "This video isn't available (private, removed or age-restricted).",
            "ERROR: [youtube] x: Video unavailable" to "This video isn't available (private, removed or age-restricted).",
            "ERROR: [youtube] x: Sign in to confirm your age. This video may be inappropriate for some users." to
                "This video isn't available (private, removed or age-restricted).",
            "ERROR: [Instagram] x: Instagram API is not granting access" to
                "This Instagram post needs a login — log in under Settings, or save the video and share the file.",
            "ERROR: [Instagram] x: empty media response ... use --cookies" to
                "This Instagram post needs a login — log in under Settings, or save the video and share the file.",
            "ERROR: [Instagram] Chunk8-jurw: Requested format is not available. Use --list-formats for a list of available formats" to
                "This post has no audio.",
            "ERROR: [youtube] x: Sign in to confirm you're not a bot" to
                "YouTube refused the download. Try again later, or save the video and share the file.",
            "ERROR: [youtube] x: No video formats found!" to
                "YouTube refused the download. Try again later, or save the video and share the file.",
            "ERROR: something odd happened\nmore lines" to "Couldn't fetch the audio: something odd happened",
            "ERROR: [youtube] x: Unable to download webpage: HTTP Error 403: Forbidden (caused by <HTTPError 403: Forbidden>)" to
                "YouTube refused the download. Try again later, or save the video and share the file.",
            "ERROR: [youtube] x: Unable to download webpage: <urlopen error HTTP Error 403: Forbidden>" to
                "YouTube refused the download. Try again later, or save the video and share the file.",
            "ERROR: [Instagram] CDUMkliABpa: Unable to download webpage: HTTP Error 401: Unauthorized (caused by <HTTPError 401: Unauthorized>)" to
                "This Instagram post needs a login — log in under Settings, or save the video and share the file.",
            "ERROR: [Instagram] CDUMkliABpa: Unable to download JSON metadata: HTTP Error 429: Too Many Requests (caused by <HTTPError 429: Too Many Requests>)" to
                "This Instagram post needs a login — log in under Settings, or save the video and share the file.",
            "ERROR: [Instagram] CDUMkliABpa: Unable to download webpage: HTTP Error 403: Forbidden (caused by <HTTPError 403: Forbidden>)" to
                "This Instagram post needs a login — log in under Settings, or save the video and share the file.",
            "ERROR: [Instagram] CDUMkliABpa: Unable to download webpage: <urlopen error [Errno -3] Temporary failure in name resolution>" to
                "Couldn't reach the site. Check your connection and try again.",
        )
        for ((error, message) in cases) {
            val site = if ("[Instagram]" in error) Site.INSTAGRAM else Site.YOUTUBE
            assertEquals(message, Links.errorMessage(site, error), error)
        }
    }

    @Test
    fun fallbackTitles() {
        assertEquals("YouTube · jNQXAC9IVRw", Links.fallbackTitle("https://www.youtube.com/watch?v=jNQXAC9IVRw&t=3", Site.YOUTUBE))
        assertEquals("YouTube · jNQXAC9IVRw", Links.fallbackTitle("https://youtu.be/jNQXAC9IVRw?si=x", Site.YOUTUBE))
        assertEquals("YouTube · jNQXAC9IVRw", Links.fallbackTitle("https://youtube.com/shorts/jNQXAC9IVRw", Site.YOUTUBE))
        assertEquals("YouTube · jNQXAC9IVRw", Links.fallbackTitle("https://www.youtube.com/live/jNQXAC9IVRw?feature=share", Site.YOUTUBE))
        assertEquals("Instagram · CDUMkliABpa", Links.fallbackTitle("https://www.instagram.com/reel/CDUMkliABpa/", Site.INSTAGRAM))
        assertEquals("Instagram · CDUMkliABpa", Links.fallbackTitle("https://www.instagram.com/someone/p/CDUMkliABpa/?igsh=x", Site.INSTAGRAM))
        assertEquals("YouTube", Links.fallbackTitle("not a url", Site.YOUTUBE))
    }

    @Test
    fun onlyExtractionFailuresRetryWithJs() {
        assertTrue(Links.isRetryableWithJs("ERROR: [youtube] x: No video formats found!"))
        assertTrue(Links.isRetryableWithJs("ERROR: [youtube] x: Requested format is not available"))
        assertFalse(Links.isRetryableWithJs("ERROR: Read timed out."))
        assertFalse(Links.isRetryableWithJs("ERROR: [youtube] x: Video unavailable"))
        assertFalse(Links.isRetryableWithJs("ERROR: [youtube] x: Private video"))
        assertTrue(Links.isRetryableWithJs("ERROR: [youtube] x: Unable to download webpage: HTTP Error 403: Forbidden (caused by <HTTPError 403: Forbidden>)"))
        assertTrue(Links.isRetryableWithJs("ERROR: [youtube] x: Unable to download webpage: <urlopen error HTTP Error 403: Forbidden>"))
    }

    @Test
    fun netscapeCookieFile() {
        val text = Links.netscapeCookies("sessionid=abc; csrftoken=def", ".instagram.com", 1_900_000_000)
        assertEquals(
            "# Netscape HTTP Cookie File\n" +
            ".instagram.com\tTRUE\t/\tTRUE\t1900000000\tsessionid\tabc\n" +
            ".instagram.com\tTRUE\t/\tTRUE\t1900000000\tcsrftoken\tdef\n", text)
    }

    @Test
    fun anyUnicodeSpaceEndsTheUrl() {
        assertEquals("https://youtu.be/abc", Links.extractUrl("https://youtu.be/abc\u00A0🎹"))
        assertEquals("https://youtu.be/abc", Links.extractUrl("https://youtu.be/abc\u2003next"))
        assertEquals("https://youtu.be/abc", Links.extractUrl("https://youtu.be/abc\u3000次"))
    }

    /** The JVM accepts regex syntax Android's ICU engine rejects; a pattern ICU cannot compile crashed the app on a share. */
    @Test
    fun patternsAvoidJavaOnlyRegexSyntax() {
        for (p in Links.PATTERNS.map { it.pattern }) {
            for (javaOnly in listOf("(?U)", "(?u)", "\\p{java", "\\R", "\\X")) {
                assertFalse(javaOnly in p, "Java-only $javaOnly in $p")
            }
        }
    }
}

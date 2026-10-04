package dev.scorefromaudio.pipeline

import java.net.URI

enum class Site { YOUTUBE, INSTAGRAM }

/**
 * Links shared into the app: finding the URL in shared text, deciding whether
 * it is a site the fetcher handles, and turning yt-dlp's error text into
 * something a person can act on. Pure, so it is tested on the JVM with the
 * error strings the on-phone spike actually saw.
 */
object Links {
    const val UNSUPPORTED = "Only YouTube and Instagram links are supported"

    private const val NETWORK = "Couldn't reach the site. Check your connection and try again."
    private const val UNAVAILABLE = "This video isn't available (private, removed or age-restricted)."
    private const val LOGIN = "This Instagram post needs a login — log in under Settings, or save the video and share the file."
    private const val NO_AUDIO = "This post has no audio."
    private const val REFUSED = "YouTube refused the download. Try again later, or save the video and share the file."

    // Stops at any space, \s for ASCII and \p{Z} for the Unicode separators (U+00A0 and the like), and at
    // bracket characters a shared caption might wrap the link in. Not (?U): Android's ICU regex engine
    // rejects that Java-only flag, and a pattern it cannot compile crashes the app when this object loads.
    private val URL = Regex("""https?://[^\s\p{Z}<>"'(){}\[\]]+""")
    private val YOUTUBE_HOSTS = setOf("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com")
    private val INSTAGRAM_HOSTS = setOf("instagram.com", "www.instagram.com")
    // The paths yt-dlp's Instagram extractor takes (its _VALID_URL): an optional user segment, then p, tv, reel or
    // reels and the id. It refuses /share/ links, so this does too: fetcher.py allows no other extractor to take them.
    private val INSTAGRAM_PATH = Regex("""^(?:/(?!share/)[^/]+)?/(?:p|tv|reels?(?!/audio/))/([^/?#&]+)""")
    private val INSTAGRAM_HTTP_4XX = Regex("""http ?error (?:401|403|429)""")
    private val PREFIX = Regex("""^\[[^\]]+\] [^:]+: """)

    /** Every pattern above, for the test that keeps them to syntax Android's ICU engine also accepts. */
    internal val PATTERNS: List<Regex> get() = listOf(URL, INSTAGRAM_PATH, INSTAGRAM_HTTP_4XX, PREFIX)

    fun extractUrl(text: String): String? =
        URL.find(text)?.value?.trimEnd('.', ',', ';', ':', '!', '?', ')', ']', '}')

    private fun queryParams(query: String?): Map<String, String> =
        (query ?: "").split('&').filter { it.isNotEmpty() }.associate {
            val eq = it.indexOf('=')
            if (eq < 0) it to "" else it.substring(0, eq) to it.substring(eq + 1)
        }

    fun site(url: String): Site? {
        val uri = runCatching { URI(url) }.getOrNull() ?: return null
        if (uri.scheme !in setOf("http", "https")) return null
        val host = uri.host?.lowercase() ?: return null
        val path = uri.path ?: ""
        return when {
            host == "youtu.be" && path.length > 1 -> Site.YOUTUBE
            host in YOUTUBE_HOSTS && (path == "/watch" && !queryParams(uri.query)["v"].isNullOrEmpty() ||
                path.startsWith("/shorts/") || path.startsWith("/live/")) -> Site.YOUTUBE
            host in INSTAGRAM_HOSTS && INSTAGRAM_PATH.containsMatchIn(path) -> Site.INSTAGRAM
            else -> null
        }
    }

    fun errorMessage(site: Site, error: String): String {
        val e = error.lowercase()
        return when {
            listOf("private video", "video unavailable", "confirm your age", "has been removed", "is not available in your country",
                   "this video is private", "members-only").any { it in e } -> UNAVAILABLE
            // Checked before the network patterns: a YouTube 403/bot-check often also says
            // "Unable to download webpage", but it is a refusal, not a network failure.
            site == Site.YOUTUBE && listOf("not a bot", "no video formats", "po token", "requested format is not available",
                   "page needs to be reloaded", "http error 403").any { it in e } -> REFUSED
            // Instagram answers a request it wants logged in with 401, 403 or 429, wrapped in "Unable to download
            // webpage": that is a login problem, not the network.
            site == Site.INSTAGRAM && INSTAGRAM_HTTP_4XX.containsMatchIn(e) -> LOGIN
            listOf("name resolution", "timed out", "network is unreachable", "connection refused", "connection reset",
                   "unable to download webpage", "urlopen error", "failed to establish").any { it in e } -> NETWORK
            site == Site.INSTAGRAM && listOf("not granting access", "login required", "use --cookies", "rate-limit",
                   "login_required").any { it in e } -> LOGIN
            site == Site.INSTAGRAM && "requested format is not available" in e -> NO_AUDIO
            else -> "Couldn't fetch the audio: " + error.lineSequence().first().removePrefix("ERROR: ")
                .replace(PREFIX, "").trim()
        }
    }

    /** "YouTube · <video id>" or "Instagram · <post id>": a job's title until, or unless, the fetch finds the real one. */
    fun fallbackTitle(url: String, site: Site): String {
        val name = when (site) { Site.YOUTUBE -> "YouTube"; Site.INSTAGRAM -> "Instagram" }
        val uri = runCatching { URI(url) }.getOrNull() ?: return name
        val path = uri.path ?: ""
        val id = when (site) {
            Site.YOUTUBE -> when {
                uri.host?.lowercase() == "youtu.be" -> path.removePrefix("/").substringBefore('/')
                path == "/watch" -> queryParams(uri.query)["v"]
                else -> path.split('/').getOrNull(2)
            }
            Site.INSTAGRAM -> INSTAGRAM_PATH.find(path)?.groupValues?.get(1)
        }
        return if (id.isNullOrEmpty()) name else "$name · $id"
    }

    /** A YouTube failure that a JavaScript runtime might fix; not network, availability or cancellation. */
    fun isRetryableWithJs(error: String): Boolean {
        val e = error.lowercase()
        val http403 = "http error 403" in e
        val hardStop = listOf("timed out", "name resolution", "private video", "video unavailable",
                   "confirm your age", "cancelled").any { it in e } ||
                   ("unable to download webpage" in e && !http403)
        if (hardStop) return false
        return listOf("no video formats", "requested format is not available", "not a bot", "po token", "signature",
                      "nsig", "challenge", "page needs to be reloaded", "http error 403").any { it in e }
    }

    /** The cookies of one `Cookie:` header as a Netscape cookie file, the format yt-dlp's `cookiefile` reads. */
    fun netscapeCookies(cookieHeader: String, domain: String, expiresEpochSeconds: Long): String = buildString {
        append("# Netscape HTTP Cookie File\n")
        val includeSubdomains = if (domain.startsWith(".")) "TRUE" else "FALSE"
        for (pair in cookieHeader.split(';')) {
            val trimmed = pair.trim()
            val eq = trimmed.indexOf('=')
            if (eq <= 0) continue
            append("$domain\t$includeSubdomains\t/\tTRUE\t$expiresEpochSeconds\t${trimmed.substring(0, eq)}\t${trimmed.substring(eq + 1)}\n")
        }
    }
}

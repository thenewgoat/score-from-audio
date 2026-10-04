package dev.scorefromaudio.pipeline

object AudioTracks {
    /** The track to decode from a container's MIME types, or null when it has no audio. */
    fun pick(mimes: List<String>): Int? = mimes.indexOfFirst { it.startsWith("audio/") }.takeIf { it >= 0 }
}

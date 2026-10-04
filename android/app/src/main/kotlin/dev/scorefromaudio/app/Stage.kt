package dev.scorefromaudio.app

enum class Stage(val label: String) {
    FETCHING("fetching audio"),
    DECODING("decoding audio"),
    NOTES("finding notes"),
    NOTATION("writing notation"),
    WRITING("laying out the score"),
}

data class Progress(val jobId: String, val stage: Stage, val fraction: Float)

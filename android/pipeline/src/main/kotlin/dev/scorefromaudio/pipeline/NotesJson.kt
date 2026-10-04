package dev.scorefromaudio.pipeline

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** Notes as `{"notes": [[start, end, pitch, velocity], ...]}` -- what a job stores and the CLI reads. */
object NotesJson {
    @Serializable
    private class File(val notes: List<List<Double>>)

    private val json = Json { ignoreUnknownKeys = true }

    fun write(notes: List<Note>): String =
        json.encodeToString(File.serializer(), File(notes.map { listOf(it.start, it.end, it.pitch.toDouble(), it.velocity.toDouble()) }))

    fun read(text: String): List<Note> =
        json.decodeFromString(File.serializer(), text).notes.map { Note(it[0], it[1], it[2].toInt(), it[3].toInt()) }
}

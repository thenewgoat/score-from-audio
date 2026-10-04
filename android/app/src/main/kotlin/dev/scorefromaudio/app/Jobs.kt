package dev.scorefromaudio.app

import kotlinx.coroutines.flow.MutableStateFlow

/**
 * What the UI observes: the running job's progress, and a counter bumped whenever the library changes.
 * [watching] is the job whose progress is on screen, so its "ready" notification is not needed.
 */
object Jobs {
    val progress = MutableStateFlow<Progress?>(null)
    val changed = MutableStateFlow(0)
    val watching = MutableStateFlow<String?>(null)
}

package dev.scorefromaudio.app.ui

import android.media.MediaPlayer
import java.io.File

/** A file through MediaPlayer: the score's MIDI (Android's built-in synthesiser) or the original recording. */
class Player {
    private var player: MediaPlayer? = null

    fun load(file: File, startMs: Int) {
        release()
        player = MediaPlayer().apply {
            setDataSource(file.path)
            prepare()
            seekTo(startMs.coerceIn(0, duration))
        }
    }

    val loaded: Boolean get() = player != null
    val position: Int get() = player?.currentPosition ?: 0
    val duration: Int get() = player?.duration ?: 0
    val playing: Boolean get() = player?.isPlaying == true

    fun play() { player?.start() }
    fun pause() { if (playing) player?.pause() }
    fun seekTo(ms: Int) { player?.let { it.seekTo(ms.coerceIn(0, it.duration)) } }

    /** Time-stretched speed, pitch kept. Setting a speed starts a paused MediaPlayer, so it is paused again. */
    fun setSpeed(speed: Float) {
        val p = player ?: return
        val was = p.isPlaying
        runCatching { p.playbackParams = p.playbackParams.setSpeed(speed).setPitch(1f) }
        if (!was && p.isPlaying) p.pause()
    }

    fun release() {
        player?.release()
        player = null
    }
}

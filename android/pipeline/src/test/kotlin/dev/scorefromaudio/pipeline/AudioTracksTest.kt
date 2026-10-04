package dev.scorefromaudio.pipeline

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class AudioTracksTest {
    @Test
    fun picksTheFirstAudioTrack() = assertEquals(1, AudioTracks.pick(listOf("video/avc", "audio/mp4a-latm", "audio/opus")))

    @Test
    fun aVideoWithoutSoundHasNone() = assertNull(AudioTracks.pick(listOf("video/avc")))

    @Test
    fun anEmptyContainerHasNone() = assertNull(AudioTracks.pick(emptyList()))
}

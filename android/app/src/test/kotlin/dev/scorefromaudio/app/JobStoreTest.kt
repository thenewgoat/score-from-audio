package dev.scorefromaudio.app

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull

class JobStoreTest {
    private val store = JobStore(Files.createTempDirectory("jobs").toFile())

    @Test
    fun scratchIsWhereItIsToldAndSweptApartFromTheJobs() {
        val base = Files.createTempDirectory("app").toFile()
        val jobs = JobStore(File(base, "jobs"), File(base, "cache/fetch"))
        val job = jobs.create("kept")
        assertEquals(File(base, "cache/fetch"), jobs.scratchRoot())
        File(jobs.scratchRoot(), "fetch-1").mkdirs()
        File(jobs.scratchRoot(), "fetch-1/a.m4a.part").writeText("x")
        jobs.sweepScratch()
        assertEquals(0, jobs.scratchRoot().listFiles()!!.size)
        assertEquals(listOf(job.id), jobs.list().map { it.id })
    }

    @Test
    fun newestFirst() {
        val a = store.create("first")
        Thread.sleep(5)
        val b = store.create("second")
        assertEquals(listOf(b.id, a.id), store.list().map { it.id })
    }

    @Test
    fun savesAndLoads() {
        val job = store.create("piece")
        job.status = "done"; job.bars = 12
        store.save(job)
        assertEquals(12, store.load(job.id)!!.bars)
    }

    @Test
    fun aJobLeftRunningByACrashBecomesFailed() {
        val job = store.create("crashed")
        job.stage = "NOTES"
        store.save(job)
        store.markInterrupted()
        val after = store.load(job.id)!!
        assertEquals("failed", after.status)
        assertEquals("NOTES", after.stage)
    }

    @Test
    fun deleteRemovesTheFolder() {
        val job = store.create("gone")
        store.write(job.id, "notes.json", "{}")
        store.delete(job.id)
        assertFalse(store.dir(job.id).exists())
        assertNull(store.load(job.id))
    }

    @Test
    fun aTruncatedMetaIsSkippedNotFatal() {
        val good = store.create("good")
        val cut = store.create("cut")
        val meta = File(store.dir(cut.id), "meta.json")
        meta.writeText(meta.readText().take(20))
        assertNull(store.load(cut.id))
        assertEquals(listOf(good.id), store.list().map { it.id })
        store.markInterrupted()
        assertEquals("failed", store.load(good.id)!!.status)
    }

    @Test
    fun writesLeaveNoTemporaryFiles() {
        val job = store.create("piece")
        store.write(job.id, "notes.json", "[]")
        store.write(job.id, "notes.json", "[1]")
        assertEquals("[1]", File(store.dir(job.id), "notes.json").readText())
        assertEquals(setOf("meta.json", "notes.json"), store.dir(job.id).list()!!.toSet())
    }
}

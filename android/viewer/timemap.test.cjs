const test = require("node:test");
const assert = require("node:assert");
const { activeAt, durationOf } = require("../app/src/main/assets/viewer/timemap.js");

const map = [
  { tstamp: 0, on: ["a", "b"] },
  { tstamp: 500, off: ["a"], on: ["c"] },
  { tstamp: 1000, off: ["b", "c"] },
];

test("nothing sounds before the first event", () => assert.deepStrictEqual([...activeAt(map, -1)], []));
test("notes sound from their on to their off", () => {
  assert.deepStrictEqual([...activeAt(map, 250)].sort(), ["a", "b"]);
  assert.deepStrictEqual([...activeAt(map, 500)].sort(), ["b", "c"]);
});
test("everything is off at the end", () => assert.deepStrictEqual([...activeAt(map, 2000)], []));
test("duration is the last event", () => assert.strictEqual(durationOf(map), 1000));

const { measures, msToQ, qToMs, audioToQ, qToAudio, barAt } = require("../app/src/main/assets/viewer/timemap.js");

// Two bars of 2/4 at 60 bpm, then the tempo doubles in the second bar.
const withBars = [
  { tstamp: 0, qstamp: 0, measureOn: "m1", on: ["a"] },
  { tstamp: 1000, qstamp: 1, off: ["a"], on: ["b"] },
  { tstamp: 2000, qstamp: 2, measureOn: "m2", off: ["b"], on: ["c"] },
  { tstamp: 3000, qstamp: 4, off: ["c"] },
];

test("bars run from their first event to the next bar", () => {
  assert.deepStrictEqual(measures(withBars), [
    { id: "m1", ms: 0, q: 0, end: 2000, qEnd: 2 },
    { id: "m2", ms: 2000, q: 2, end: 3000, qEnd: 4 },
  ]);
  assert.strictEqual(barAt(measures(withBars), 2500), 1);
  assert.strictEqual(barAt(measures(withBars), 0), 0);
});

test("milliseconds and quarters follow tempo changes both ways", () => {
  assert.strictEqual(msToQ(withBars, 2500), 3);
  assert.strictEqual(qToMs(withBars, 3), 2500);
  assert.strictEqual(qToMs(withBars, 0.5), 500);
});

test("the recording maps through its anchors and carries on at its pace past them", () => {
  const anchors = [{ quarter: 0, seconds: 1.0 }, { quarter: 1, seconds: 1.5 }, { quarter: 4, seconds: 3.5 }];
  assert.strictEqual(audioToQ(anchors, 1.25), 0.5);
  assert.strictEqual(qToAudio(anchors, 2.5), 2.5);
  assert.strictEqual(qToAudio(anchors, 5), 3.5 + 2.5 / 4);
  assert.strictEqual(qToAudio(anchors, -10), 0, "never before the recording starts");
  assert.ok(Math.abs(audioToQ(anchors, qToAudio(anchors, 3.2)) - 3.2) < 1e-9);
});

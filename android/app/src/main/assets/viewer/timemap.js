(function (root) {
  /** Ids of the notes sounding at `ms`, from Verovio's timemap ([{tstamp, on, off}], in time order). */
  function activeAt(timemap, ms) {
    const on = new Set();
    for (const event of timemap) {
      if (event.tstamp > ms) break;
      (event.off || []).forEach((id) => on.delete(id));
      (event.on || []).forEach((id) => on.add(id));
    }
    return on;
  }

  function durationOf(timemap) {
    return timemap.length ? timemap[timemap.length - 1].tstamp : 0;
  }

  /** Bars in order from a timemap taken with includeMeasures: [{id, ms, end, q, qEnd}]. */
  function measures(timemap) {
    const out = [];
    for (const event of timemap) if (event.measureOn) out.push({ id: event.measureOn, ms: event.tstamp, q: event.qstamp });
    const last = timemap.length ? timemap[timemap.length - 1] : { tstamp: 0, qstamp: 0 };
    out.forEach((bar, i) => {
      bar.end = i + 1 < out.length ? out[i + 1].ms : last.tstamp;
      bar.qEnd = i + 1 < out.length ? out[i + 1].q : last.qstamp;
    });
    return out;
  }

  /** The last index whose key is <= x (0 when none is), by binary search over points sorted by key. */
  function floorIndex(points, x, key) {
    let lo = 0, hi = points.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (points[mid][key] <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  /**
   * y at x, linear between points sorted by `xKey`. Past the ends it carries on at `slope` (y per x) when given,
   * else it holds the end value.
   */
  function interpolate(points, x, xKey, yKey, slope) {
    if (!points.length) return 0;
    const first = points[0], last = points[points.length - 1];
    if (x <= first[xKey]) return slope === undefined ? first[yKey] : first[yKey] + (x - first[xKey]) * slope;
    if (x >= last[xKey]) return slope === undefined ? last[yKey] : last[yKey] + (x - last[xKey]) * slope;
    const i = floorIndex(points, x, xKey);
    const a = points[i], b = points[i + 1];
    const span = b[xKey] - a[xKey];
    return span > 0 ? a[yKey] + (b[yKey] - a[yKey]) * (x - a[xKey]) / span : a[yKey];
  }

  /** Written-tempo milliseconds <-> quarter notes, through the timemap's own events. */
  const msToQ = (timemap, ms) => interpolate(timemap, ms, "tstamp", "qstamp");
  const qToMs = (timemap, q) => interpolate(timemap, q, "qstamp", "tstamp");

  /** Seconds per quarter over the whole recording, for times before the first note or after the last. */
  function pace(anchors) {
    if (anchors.length < 2) return 0.5;
    const a = anchors[0], b = anchors[anchors.length - 1];
    return b.quarter > a.quarter ? (b.seconds - a.seconds) / (b.quarter - a.quarter) : 0.5;
  }

  /** Recording seconds <-> quarter notes, through the anchors ([{quarter, seconds}]) the app wrote in sync.json. */
  const audioToQ = (anchors, seconds) => interpolate(anchors, seconds, "seconds", "quarter", 1 / pace(anchors));
  const qToAudio = (anchors, q) => Math.max(0, interpolate(anchors, q, "quarter", "seconds", pace(anchors)));

  /** The bar sounding at written-tempo `ms` (index into `bars` from measures()). */
  const barAt = (bars, ms) => (bars.length ? floorIndex(bars, ms, "ms") : -1);

  const api = { activeAt, durationOf, measures, interpolate, msToQ, qToMs, audioToQ, qToAudio, barAt };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Timemap = api;
})(this);

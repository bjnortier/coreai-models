// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAISpeech

/// Whisper's window: 3000 mel frames at hop 160 = 480_000 samples = 30 s @ 16 kHz.
private let window = 480_000
/// 2 s @ 16 kHz.
private let overlap = 32_000

@Suite("Long-form window boundaries")
struct LongFormWindowStartsTests {
    @Test("Audio shorter than one window is a single window")
    func shorterThanWindow() {
        #expect(
            LongForm.windowStarts(
                pcmCount: 16_000, windowSamples: window, overlapSamples: overlap) == [0])
    }

    @Test("Audio exactly one window long is a single window")
    func exactlyOneWindow() {
        #expect(
            LongForm.windowStarts(
                pcmCount: window, windowSamples: window, overlapSamples: overlap) == [0])
    }

    @Test("Windows advance by window minus overlap")
    func advancesByStride() {
        // 62.3 s — the case that motivated this: three windows, not one.
        let starts = LongForm.windowStarts(
            pcmCount: 996_800, windowSamples: window, overlapSamples: overlap)
        #expect(starts == [0, 448_000, 896_000])
    }

    @Test("Windows cover the whole waveform")
    func coversEverything() {
        for count in [window + 1, 700_000, 996_800, 3_000_000] {
            let starts = LongForm.windowStarts(
                pcmCount: count, windowSamples: window, overlapSamples: overlap)
            #expect(starts.first == 0)
            let lastEnd = min(starts.last! + window, count)
            #expect(lastEnd == count, "window \(count) ends at \(lastEnd)")
        }
    }

    @Test("Consecutive windows really do overlap")
    func consecutiveWindowsOverlap() {
        let starts = LongForm.windowStarts(
            pcmCount: 2_000_000, windowSamples: window, overlapSamples: overlap)
        #expect(starts.count > 2)
        for (a, b) in zip(starts, starts.dropFirst()) {
            #expect(a + window - b == overlap)
        }
    }

    @Test("The trailing window is never shorter than the overlap")
    func trailingWindowHasSubstance() {
        // A sliver of near-silence is exactly what Whisper hallucinates on, so the
        // last window must carry real audio.
        for count in [window + 1, window + 5_000, 2 * window - 1, 1_500_000] {
            let starts = LongForm.windowStarts(
                pcmCount: count, windowSamples: window, overlapSamples: overlap)
            let lastLength = min(starts.last! + window, count) - starts.last!
            #expect(lastLength >= overlap, "count \(count) gave a \(lastLength)-sample tail")
        }
    }

    @Test("Zero overlap butts windows together")
    func zeroOverlap() {
        let starts = LongForm.windowStarts(
            pcmCount: 996_800, windowSamples: window, overlapSamples: 0)
        #expect(starts == [0, 480_000, 960_000])
    }
}

@Suite("Long-form seam stitching")
struct LongFormStitchTests {
    @Test("A single window passes through unchanged")
    func singleWindow() {
        #expect(LongForm.stitch(["hello there"]) == "hello there")
    }

    @Test("An exact repeat at the seam is dropped once")
    func exactRepeat() {
        let joined = LongForm.stitch(["the quick brown fox", "brown fox jumps over"])
        #expect(joined == "the quick brown fox jumps over")
    }

    @Test("The longest repeat wins, not the shortest")
    func longestRepeat() {
        // "over" alone also matches; dropping only that would duplicate "jumps".
        let joined = LongForm.stitch(["a b jumps over", "jumps over the lazy dog"])
        #expect(joined == "a b jumps over the lazy dog")
    }

    @Test("Case and punctuation differences still match")
    func looseMatching() {
        // The same word ends a sentence in one window and starts one in the next.
        let joined = LongForm.stitch(["and then he left.", "He left the building"])
        #expect(joined == "and then he left. the building")
    }

    @Test("No match concatenates rather than guessing")
    func noMatch() {
        // Losing speech is worse than repeating it, so nothing is dropped.
        let joined = LongForm.stitch(["alpha beta", "gamma delta"])
        #expect(joined == "alpha beta gamma delta")
    }

    @Test("Empty windows are skipped")
    func emptyWindows() {
        #expect(LongForm.stitch(["alpha", "", "  ", "beta"]) == "alpha beta")
        #expect(LongForm.stitch([]) == "")
        #expect(LongForm.stitch(["", ""]) == "")
    }

    @Test("A repeat longer than the search limit goes undetected, losing nothing")
    func respectsSearchLimit() {
        // The limit caps how far back a seam is compared, and a longer repeat does
        // not partially match: comparing the last k words against the first k lands
        // on different parts of the run for every k below its true length. So the
        // result is a full concatenation, not a partial trim — the safe direction.
        let words = (1...10).map(String.init).joined(separator: " ")
        #expect(
            LongForm.stitch([words, words], maxOverlapWords: 4)
                == "1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10")
        // The same input is fully de-duplicated once the limit can reach it.
        #expect(LongForm.stitch([words, words], maxOverlapWords: 16) == words)
    }

    @Test("overlapLength reports the matched run length")
    func overlapLengthDirectly() {
        #expect(
            LongForm.overlapLength(
                tail: ["a", "b", "c"], head: ["b", "c", "d"], maxOverlapWords: 16) == 2)
        #expect(
            LongForm.overlapLength(
                tail: ["a", "b"], head: ["x", "y"], maxOverlapWords: 16) == 0)
        #expect(
            LongForm.overlapLength(tail: [], head: ["a"], maxOverlapWords: 16) == 0)
    }
}

@Suite("Long-form stats merging")
struct LongFormStatsTests {
    @Test("A single window's stats pass through untouched")
    func singleWindow() {
        let one = DecodeStats(stepTimesMs: [1, 2, 3])
        #expect(DecodeStats.merged([one]).stepCount == 3)
        #expect(DecodeStats.merged([one]).windowCount == 1)
    }

    @Test("Step times concatenate and window count reflects the split")
    func concatenates() {
        let merged = DecodeStats.merged([
            DecodeStats(stepTimesMs: [1, 2]),
            DecodeStats(stepTimesMs: [3]),
            DecodeStats(stepTimesMs: [4, 5]),
        ])
        #expect(merged.stepCount == 5)
        #expect(merged.windowCount == 3)
        #expect(merged.minLatencyMs == 1)
        #expect(merged.maxLatencyMs == 5)
    }

    @Test("Coverage counters sum across windows")
    func coverageSums() {
        var a = DecodeStats.Coverage()
        a.blankSkipReuses = 2
        a.multiTokenSteps = 1
        var b = DecodeStats.Coverage()
        b.blankSkipReuses = 3
        b.blankOnlySteps = 4
        let merged = DecodeStats.merged([
            DecodeStats(stepTimesMs: [1], coverage: a),
            DecodeStats(stepTimesMs: [2], coverage: b),
        ])
        #expect(merged.coverage.blankSkipReuses == 5)
        #expect(merged.coverage.multiTokenSteps == 1)
        #expect(merged.coverage.blankOnlySteps == 4)
    }

    @Test("Stage timings sum across windows")
    func timingSums() {
        var a = DecodeStats.Timing()
        a.preprocessMs = 1
        a.encodeMs = 10
        a.prefixMs = 4
        a.prefixSteps = 4
        a.samplingMs = 2
        var b = DecodeStats.Timing()
        b.preprocessMs = 2
        b.encodeMs = 20
        b.prefixMs = 5
        b.prefixSteps = 4
        b.samplingMs = 3
        let merged = DecodeStats.merged([
            DecodeStats(stepTimesMs: [1], timing: a),
            DecodeStats(stepTimesMs: [2], timing: b),
        ])
        #expect(merged.timing.preprocessMs == 3)
        #expect(merged.timing.encodeMs == 30)
        #expect(merged.timing.prefixMs == 9)
        #expect(merged.timing.prefixSteps == 8)
        #expect(merged.timing.samplingMs == 5)
        #expect(merged.timing.languageDetectionMs == 0)
    }
}

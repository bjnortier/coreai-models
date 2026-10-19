// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAIDiarization

/// Activity at 10 ms for `seconds`, with each speaker fully active over their spans.
private func activity(seconds: Double, speakers: Int = 2, spans: [(speaker: Int, start: Double, end: Double)])
    -> SpeakerActivity
{
    let numFrames = Int(seconds * 100)
    var probabilities = [Float](repeating: 0, count: numFrames * speakers)
    for span in spans {
        for frame in Int(span.start * 100)..<min(Int(span.end * 100), numFrames) {
            probabilities[frame * speakers + span.speaker] = 1
        }
    }
    return SpeakerActivity(probabilities: probabilities, numFrames: numFrames, numSpeakers: speakers, frameSeconds: 0.01)
}

@Suite("Speaker turns")
struct SpeakerSegmentTests {
    @Test("Turns bridge short pauses and drop blips")
    func segments() {
        let act = activity(seconds: 5, spans: [(0, 0, 1), (0, 1.2, 2), (0, 3, 3.1), (1, 2.5, 4)])
        #expect(act.segments() == [
            SpeakerSegment(speaker: 0, start: 0, end: 2),
            SpeakerSegment(speaker: 1, start: 2.5, end: 4),
        ])
    }

    @Test("A longer bridge merges a turn the default splits")
    func settings() {
        let act = activity(seconds: 3, spans: [(0, 0, 1), (0, 1.5, 2.5)])
        #expect(act.segments().count == 2)
        var settings = SpeakerActivity.Segmentation()
        settings.bridgeSeconds = 0.6
        #expect(act.segments(settings) == [SpeakerSegment(speaker: 0, start: 0, end: 2.5)])
    }
}

@Suite("Speaker cache")
struct SpeakerCacheTests {
    private let policy = SpeakerCachePolicy(
        speakerCacheLength: 8, speakerCacheSilenceFramesPerSpeaker: 1, predictionScoreThreshold: 0.25,
        latestFramesScoreBoost: 0.05, minPositiveScoresRate: 0.5, strongBoostRate: 0.75, weakBoostRate: 1.5)

    /// One step of `frames` new frames whose embeddings encode their index, all spoken by `speaker`.
    private func push(_ cache: inout SpeakerCache, frames: Int, first: Int, speaker: Int) {
        let hidden = cache.hiddenSize
        let cached = cache.cachedEmbeds
        let chunk = (0..<frames).flatMap { [Float](repeating: Float(first + $0), count: hidden) }
        var logits = [Float]()
        for _ in 0..<((cache.numCachedFrames + frames) * cache.subsampling) {
            logits += (0..<cache.numSpeakers).map { $0 == speaker ? 10 : -10 }
        }
        cache.update(
            inputEmbeds: cached + chunk, logits: logits, silenceEmbeds: [Float](repeating: -1, count: hidden),
            numChunkFrames: frames, mask: [Bool](repeating: true, count: cache.numCachedFrames + frames))
    }

    @Test("Frames stay in the FIFO until it overflows, then the oldest move to the cache")
    func fifo() {
        var cache = SpeakerCache(policy: policy, fifoLength: 4, updatePeriod: 3, numSpeakers: 2, hiddenSize: 2, subsampling: 2)
        push(&cache, frames: 4, first: 0, speaker: 0)
        #expect(cache.numCacheFrames == 0 && cache.numFifoFrames == 4)
        push(&cache, frames: 2, first: 4, speaker: 0)
        // 6 frames overflow a 4-frame FIFO: max(update period 3, overflow 2) = 3 move
        #expect(cache.numCacheFrames == 3 && cache.numFifoFrames == 3)
        #expect(cache.cacheEmbeds == [0, 0, 1, 1, 2, 2])
    }

    @Test("A full cache compresses to its capacity, keeping each speaker's silence slot")
    func compress() {
        var cache = SpeakerCache(policy: policy, fifoLength: 2, updatePeriod: 2, numSpeakers: 2, hiddenSize: 2, subsampling: 2)
        for step in 0..<8 { push(&cache, frames: 2, first: 2 * step, speaker: step < 4 ? 0 : 1) }
        #expect(cache.numCacheFrames == 8)
        // two silence slots (one per speaker), whose embedding is all -1
        let rows = stride(from: 0, to: cache.cacheEmbeds.count, by: 2).map { cache.cacheEmbeds[$0] }
        #expect(rows.filter { $0 == -1 }.count == 2)
    }
}

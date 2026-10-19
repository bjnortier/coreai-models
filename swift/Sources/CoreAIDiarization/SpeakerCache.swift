// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The speaker-cache policy that metadata.json's `speaker_cache` block configures.
public struct SpeakerCachePolicy: Sendable, Codable {
    public let speakerCacheLength: Int
    public let speakerCacheSilenceFramesPerSpeaker: Int
    public let predictionScoreThreshold: Float
    public let latestFramesScoreBoost: Float
    public let minPositiveScoresRate: Float
    public let strongBoostRate: Float
    public let weakBoostRate: Float
}

/// Streaming state of Nemotron-3-Diarization: the Arrival-Order Speaker Cache and the FIFO of
/// recent encoder frames every step attends to.
///
/// A port of HF `Nemotron3DiarizationSpeakerCache` (transformers 5.19), one batch row, kept
/// operation for operation so the two can be compared step by step. Frames are encoder inputs
/// (the embedder's output, `hiddenSize` wide), not encoder outputs: every step re-encodes them.
struct SpeakerCache {
    let fifoLength: Int
    let updatePeriod: Int
    let cacheLength: Int
    let numSilenceFrames: Int
    let numSpeakers: Int
    let hiddenSize: Int
    let subsampling: Int
    private let scoreThreshold: Float
    private let latestFramesBoost: Float
    private let minPositiveScores: Int
    private let numStrongBoosted: Int
    private let numWeakBoosted: Int

    /// `[numCacheFrames, hiddenSize]` and `[numCacheFrames, numSpeakers]`, row-major.
    private(set) var cacheEmbeds: [Float] = []
    private var cacheProbs: [Float] = []
    /// `[numFifoFrames, hiddenSize]`.
    private(set) var fifo: [Float] = []
    private var isCompressed = false

    var numCacheFrames: Int { cacheEmbeds.count / hiddenSize }
    var numFifoFrames: Int { fifo.count / hiddenSize }
    var numCachedFrames: Int { numCacheFrames + numFifoFrames }

    init(policy: SpeakerCachePolicy, fifoLength: Int, updatePeriod: Int, numSpeakers: Int, hiddenSize: Int,
         subsampling: Int) {
        self.fifoLength = fifoLength
        self.updatePeriod = updatePeriod
        self.cacheLength = policy.speakerCacheLength
        self.numSilenceFrames = policy.speakerCacheSilenceFramesPerSpeaker
        self.numSpeakers = numSpeakers
        self.hiddenSize = hiddenSize
        self.subsampling = subsampling
        self.scoreThreshold = policy.predictionScoreThreshold
        self.latestFramesBoost = policy.latestFramesScoreBoost
        // each speaker's share of the cache, less its reserved silence slots
        let budget = Float(policy.speakerCacheLength / numSpeakers - policy.speakerCacheSilenceFramesPerSpeaker)
        self.minPositiveScores = Int((budget * policy.minPositiveScoresRate).rounded(.down))
        self.numStrongBoosted = Int((budget * policy.strongBoostRate).rounded(.down))
        self.numWeakBoosted = Int((budget * policy.weakBoostRate).rounded(.down))
    }

    /// The frames the next step attends to before its chunk: speaker cache, then FIFO.
    var cachedEmbeds: [Float] { cacheEmbeds + fifo }

    /// Pushes a processed step into the FIFO, moving its oldest frames into the speaker cache
    /// when it overflows and compressing the cache when that overflows.
    ///
    /// - Parameters:
    ///   - inputEmbeds: the step's encoder input without padding, `[numCachedFrames + n, hiddenSize]`.
    ///   - logits: the step's speaker logits for those frames at 10 ms, `[(numCachedFrames + n) * subsampling, numSpeakers]`.
    ///   - numChunkFrames: how many frames after the cached ones join the FIFO (the look-ahead does not).
    ///   - mask: valid frames of `inputEmbeds`; padding frames get zero probability.
    mutating func update(
        inputEmbeds: [Float], logits: [Float], silenceEmbeds: [Float], numChunkFrames: Int, mask: [Bool]
    ) {
        let numCache = numCacheFrames
        let numFifo = numFifoFrames
        let probs = poolProbs(logits: logits, mask: mask)

        let chunkStart = numCache + numFifo
        var fifoEmbeds = fifo
        fifoEmbeds += inputEmbeds[(chunkStart * hiddenSize)..<((chunkStart + numChunkFrames) * hiddenSize)]
        let fifoCount = fifoEmbeds.count / hiddenSize

        let numPopped = numPoppedFrames(fifoCount)
        if numPopped > 0 {
            let fifoProbs = probs[(numCache * numSpeakers)..<((numCache + fifoCount) * numSpeakers)]
            // an uncompressed cache still holds plain chunk frames, whose probabilities this step
            // re-estimates; a compressed one is out of order, so only its stored probs apply
            let stored = isCompressed ? cacheProbs : Array(probs[0..<(numCache * numSpeakers)])
            var embeds = cacheEmbeds + fifoEmbeds[0..<(numPopped * hiddenSize)]
            var newProbs = stored + fifoProbs.prefix(numPopped * numSpeakers)
            fifoEmbeds = Array(fifoEmbeds[(numPopped * hiddenSize)...])

            if embeds.count / hiddenSize > cacheLength {
                (embeds, newProbs) = compress(embeds: embeds, probs: newProbs, silenceEmbeds: silenceEmbeds)
                isCompressed = true
            }
            cacheEmbeds = embeds
            cacheProbs = newProbs
        }
        fifo = fifoEmbeds
    }

    /// Speaker probabilities at the encoder frame rate: sigmoid, averaged over each frame's
    /// `subsampling` logit rows, zeroed on padding frames.
    private func poolProbs(logits: [Float], mask: [Bool]) -> [Float] {
        let numFrames = logits.count / (numSpeakers * subsampling)
        var probs = [Float](repeating: 0, count: numFrames * numSpeakers)
        for frame in 0..<numFrames where frame >= mask.count || mask[frame] {
            for row in 0..<subsampling {
                let base = (frame * subsampling + row) * numSpeakers
                for s in 0..<numSpeakers {
                    probs[frame * numSpeakers + s] += 1 / (1 + exp(-logits[base + s]))
                }
            }
            for s in 0..<numSpeakers { probs[frame * numSpeakers + s] /= Float(subsampling) }
        }
        return probs
    }

    /// No frames move until the FIFO overflows; then at least `updatePeriod` of the oldest do,
    /// plus whatever else restores its capacity.
    private func numPoppedFrames(_ numFifoFrames: Int) -> Int {
        if numFifoFrames <= fifoLength { return 0 }
        return min(max(updatePeriod, numFifoFrames - fifoLength), numFifoFrames)
    }

    private func frameScores(_ probs: [Float], numFrames: Int) -> [Float] {
        let logHalf = log(Float(0.5))
        var scores = [Float](repeating: 0, count: probs.count)
        for t in 0..<numFrames {
            var complementSum: Float = 0
            for s in 0..<numSpeakers {
                complementSum += log(max(1 - probs[t * numSpeakers + s], scoreThreshold))
            }
            for s in 0..<numSpeakers {
                let p = probs[t * numSpeakers + s]
                let score = log(max(p, scoreThreshold)) - log(max(1 - p, scoreThreshold)) + complementSum - logHalf
                scores[t * numSpeakers + s] = p > 0.5 ? score : -.infinity
            }
        }
        // a speaker with enough confident frames keeps only those
        for s in 0..<numSpeakers {
            var positives = 0
            for t in 0..<numFrames where scores[t * numSpeakers + s] > 0 { positives += 1 }
            guard positives >= minPositiveScores else { continue }
            for t in 0..<numFrames where probs[t * numSpeakers + s] > 0.5 && !(scores[t * numSpeakers + s] > 0) {
                scores[t * numSpeakers + s] = -.infinity
            }
        }
        return scores
    }

    /// Adds `boost` to each speaker's `count` highest-scoring frames.
    private func boostScores(_ scores: inout [Float], numFrames: Int, count: Int, boost: Float) {
        for s in 0..<numSpeakers {
            let order = (0..<numFrames).sorted { scores[$0 * numSpeakers + s] > scores[$1 * numSpeakers + s] }
            for t in order.prefix(count) { scores[t * numSpeakers + s] += boost }
        }
    }

    /// Keeps the `cacheLength` most important frames, grouped by speaker and in their original
    /// order within a speaker, with `numSilenceFrames` slots per speaker given to `silenceEmbeds`.
    private func compress(embeds: [Float], probs: [Float], silenceEmbeds: [Float]) -> ([Float], [Float]) {
        let numFrames = embeds.count / hiddenSize
        var scores = frameScores(probs, numFrames: numFrames)
        // frames past the cache's capacity are the ones just popped from the FIFO
        for t in cacheLength..<numFrames {
            for s in 0..<numSpeakers { scores[t * numSpeakers + s] += latestFramesBoost }
        }
        let logHalf = log(Float(0.5))
        boostScores(&scores, numFrames: numFrames, count: numStrongBoosted, boost: -2 * logHalf)
        boostScores(&scores, numFrames: numFrames, count: numWeakBoosted, boost: -logHalf)

        // silence slots score +inf so every speaker keeps them; they map to the silence embedding
        let numScored = numFrames + numSilenceFrames
        var flat = [(score: Float, index: Int)]()
        flat.reserveCapacity(numScored * numSpeakers)
        for s in 0..<numSpeakers {
            for t in 0..<numScored {
                flat.append((t < numFrames ? scores[t * numSpeakers + s] : .infinity, s * numScored + t))
            }
        }
        let sentinel = numScored * numSpeakers
        let top = flat.sorted { $0.score > $1.score }.prefix(cacheLength)
        let indices = top.map { $0.score == -.infinity ? sentinel : $0.index }.sorted()

        var keptEmbeds = [Float]()
        var keptProbs = [Float]()
        keptEmbeds.reserveCapacity(indices.count * hiddenSize)
        keptProbs.reserveCapacity(indices.count * numSpeakers)
        for index in indices {
            let frame = index == sentinel ? numFrames : min(index % numScored, numFrames)
            if frame == numFrames {
                keptEmbeds += silenceEmbeds
                keptProbs += [Float](repeating: 0, count: numSpeakers)
            } else {
                keptEmbeds += embeds[(frame * hiddenSize)..<((frame + 1) * hiddenSize)]
                keptProbs += probs[(frame * numSpeakers)..<((frame + 1) * numSpeakers)]
            }
        }
        return (keptEmbeds, keptProbs)
    }
}

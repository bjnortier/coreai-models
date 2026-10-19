// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

// MARK: - DecoderResources

/// Architecture-specific assets handed to a `SpeechDecoder` per call.
public enum DecoderResources: Sendable {
    case whisper(decoder: AIModel, generationConfig: GenerationConfig)
    case parakeetTDT(decoderStep: AIModel, joint: AIModel, config: ParakeetTDTConfig)
}

// MARK: - DecodeStats

/// Per-step timing collected during the autoregressive decode loop, plus enough
/// diagnostics to tell which branches of that loop an input actually reached.
public struct DecodeStats: Sendable {
    public let stepTimesMs: [Double]

    /// How often each branch of the TDT loop fired.
    ///
    /// The token sequence is a discrete, all-or-nothing check: argmax absorbs numeric
    /// drift, so "tokens matched" says little about whether the blank-skip and duration
    /// bookkeeping were exercised at all. These counts let a verification harness assert
    /// that a given input reaches a given branch instead of assuming it does. Zero for
    /// architectures without a transducer loop.
    public struct Coverage: Sendable, Equatable {
        /// Inner-loop iterations that reused the cached decoder output instead of
        /// re-running the LSTM. This is how a blank input avoids advancing the state.
        public var blankSkipReuses = 0
        /// Iterations that ran the LSTM and adopted its new hidden/cell state.
        public var lstmStateAdvances = 0
        /// `isBlank && dur == 0` — a blank choosing duration 0, forced forward one frame.
        public var blankZeroDurationBreaks = 0
        /// `dur > 0` — the ordinary exit from the inner symbol loop.
        public var positiveDurationBreaks = 0
        /// `maxSymbolsPerStep` exhausted without any duration > 0 being chosen.
        public var symbolCapExhaustions = 0
        /// Outer steps that emitted more than one non-blank token.
        public var multiTokenSteps = 0
        /// Outer steps that emitted no token at all.
        public var blankOnlySteps = 0
        /// Times the predictor was zeroed after a long silence, which only happens when a
        /// caller opts in with `resetAfterSilenceFrames`. 0 on the offline path.
        public var predictorResets = 0

        public init() {}
    }

    public let coverage: Coverage

    /// The first `decoder_step` and `joint` outputs of the decode.
    ///
    /// Retained so a harness can compare those graphs tensor-to-tensor as the runtime
    /// actually drives them, rather than only through the token sequence. Costs one
    /// retained copy of values already computed on that step.
    public struct FirstStep: Sendable {
        public let decoderOutput: [Float]
        public let newHiddenState: [Float]
        public let newCellState: [Float]
        public let jointLogits: [Float]
    }

    public let firstStep: FirstStep?

    /// Wall-clock time spent outside the timed decoder steps, for benchmarking.
    ///
    /// `stepTimesMs` covers only the generated-token graph runs. These account for
    /// the rest of a transcription, so a caller can see whether time goes to the
    /// encoder, to the per-token runs, or to host-side work between them. Zero
    /// where a stage did not run (Parakeet has no forced prefix or language step).
    public struct Timing: Sendable, Equatable {
        /// Mel spectrogram and encoder input preparation, on the CPU.
        public var preprocessMs: Double = 0
        /// Encoder graph runs.
        public var encodeMs: Double = 0
        /// Whisper language detection: its own encoder pass plus one decoder step.
        public var languageDetectionMs: Double = 0
        /// Whisper's forced-prefix decoder steps, run one token at a time.
        public var prefixMs: Double = 0
        /// Forced-prefix decoder steps taken.
        public var prefixSteps = 0
        /// Copying logits out and choosing the next token, after each generated step.
        public var samplingMs: Double = 0

        public init() {}
    }

    public var timing: Timing

    /// Where one emitted token sits in the encoder output.
    ///
    /// `frame` is the encoder frame the token was emitted at and `duration` the frame count
    /// TDT predicted alongside it, both relative to the decoded window. HF
    /// `ParakeetProcessor.decode(..., durations=...)` builds word timestamps from exactly
    /// these: start `frame`, end `frame + duration`.
    public struct TokenTiming: Sendable, Equatable {
        public let token: Int32
        public let frame: Int
        public let duration: Int
    }

    /// One entry per emitted token, in order. Empty for Whisper, and for stats merged across
    /// long-form windows, whose frames would each be relative to a different window.
    public var tokenTimings: [TokenTiming]

    /// Whisper's alignment-head scores for every generated step, when the decode was asked to
    /// keep them. Whisper's counterpart to `tokenTimings`: its tokens carry no frame, so word
    /// times are warped out of these instead (see `WhisperAlignment`). `nil` otherwise, and
    /// for stats merged across windows.
    package var alignment: WhisperAlignment.Scores?

    /// How many audio windows this transcription decoded.
    ///
    /// A static encoder (Whisper's is fixed at 30 s) can only see one window per
    /// pass, so longer audio is decoded as several. Surfaced so a caller can tell
    /// a long-form result from a single-window one.
    public let windowCount: Int

    public init(
        stepTimesMs: [Double], coverage: Coverage = Coverage(), firstStep: FirstStep? = nil,
        windowCount: Int = 1, timing: Timing = Timing(), tokenTimings: [TokenTiming] = []
    ) {
        self.stepTimesMs = stepTimesMs
        self.coverage = coverage
        self.firstStep = firstStep
        self.windowCount = windowCount
        self.timing = timing
        self.tokenTimings = tokenTimings
    }

    /// Combines per-window stats from a long-form transcription.
    ///
    /// Step times concatenate and coverage counters sum, so the aggregate reads as
    /// one decode. `firstStep` keeps the first window's tensors — it exists for
    /// parity checking, which only ever inspects the first step of the first window.
    package static func merged(_ parts: [DecodeStats]) -> DecodeStats {
        if parts.count == 1 { return parts[0] }
        var times: [Double] = []
        var coverage = Coverage()
        var timing = Timing()
        for part in parts {
            times.append(contentsOf: part.stepTimesMs)
            timing.preprocessMs += part.timing.preprocessMs
            timing.encodeMs += part.timing.encodeMs
            timing.languageDetectionMs += part.timing.languageDetectionMs
            timing.prefixMs += part.timing.prefixMs
            timing.prefixSteps += part.timing.prefixSteps
            timing.samplingMs += part.timing.samplingMs
            coverage.blankSkipReuses += part.coverage.blankSkipReuses
            coverage.lstmStateAdvances += part.coverage.lstmStateAdvances
            coverage.blankZeroDurationBreaks += part.coverage.blankZeroDurationBreaks
            coverage.positiveDurationBreaks += part.coverage.positiveDurationBreaks
            coverage.symbolCapExhaustions += part.coverage.symbolCapExhaustions
            coverage.multiTokenSteps += part.coverage.multiTokenSteps
            coverage.blankOnlySteps += part.coverage.blankOnlySteps
        }
        return DecodeStats(
            stepTimesMs: times, coverage: coverage,
            firstStep: parts.first?.firstStep, windowCount: max(1, parts.count), timing: timing)
    }

    public var stepCount: Int { stepTimesMs.count }
    public var avgLatencyMs: Double {
        guard !stepTimesMs.isEmpty else { return 0 }
        return stepTimesMs.reduce(0, +) / Double(stepTimesMs.count)
    }
    public var minLatencyMs: Double { stepTimesMs.min() ?? 0 }
    public var maxLatencyMs: Double { stepTimesMs.max() ?? 0 }
    public var stepsPerSecond: Double { avgLatencyMs > 0 ? 1000 / avgLatencyMs : 0 }
}

// MARK: - SpeechDecoder protocol

/// Model-specific decode logic.
public protocol SpeechDecoder: Sendable {
    func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources
    ) async throws -> (tokens: [Int32], stats: DecodeStats)
}

// MARK: - Helpers

/// Greedy decoder for Whisper (encoder-decoder, cross-attention, KV cache).
public struct WhisperDecoder: SpeechDecoder {
    public init() {}

    /// Predicts the spoken language from one decoder step.
    ///
    /// Whisper's first prediction after `<|startoftranscript|>` is the language
    /// token, so detection is a single step: prime the cache with SOT, then take
    /// the argmax over the language tokens only. Restricting the argmax matters —
    /// unrestricted, the highest-scoring token at that position is frequently a
    /// text token rather than a language tag.
    ///
    /// Returns `nil` when the candidate set is empty, leaving the caller to fall
    /// back to the bundle's pinned language.
    public func detectLanguageToken(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        resources: DecoderResources,
        candidates: Set<Int32>
    ) async throws -> Int32? {
        guard !candidates.isEmpty else { return nil }
        guard case .whisper(let decoderModel, let config) = resources else {
            throw SpeechError.incompatibleResources("WhisperDecoder requires .whisper resources")
        }
        guard let startToken = config.forcedPrefix.first else { return nil }
        guard let decFn = try decoderModel.loadFunction(named: "main"),
            let decDesc = decoderModel.functionDescriptor(for: "main")
        else { throw SpeechError.missingModel("No 'main' function in decoder") }

        guard case .ndArray(let inputIdsNDDesc) = decDesc.inputDescriptor(of: "input_ids"),
            case .ndArray(let posIdsNDDesc) = decDesc.inputDescriptor(of: "position_ids"),
            case .ndArray(let keyCacheNDDesc) = decDesc.stateDescriptor(of: "keyCache"),
            case .ndArray(let valCacheNDDesc) = decDesc.stateDescriptor(of: "valueCache"),
            case .ndArray(let logitsNDDesc) = decDesc.outputDescriptor(of: "logits")
        else { throw SpeechError.missingModel("Unexpected decoder descriptors") }

        let vocabSize = logitsNDDesc.shape.last!
        let maxTargetPos = 448
        var keyCache = NDArray(
            descriptor: keyCacheNDDesc.resolvingDynamicDimensions(
                keyCacheNDDesc.shape.map { $0 < 0 ? maxTargetPos : $0 }))
        var valueCache = NDArray(
            descriptor: valCacheNDDesc.resolvingDynamicDimensions(
                valCacheNDDesc.shape.map { $0 < 0 ? maxTargetPos : $0 }))
        var logitsArray = NDArray(
            descriptor: logitsNDDesc.resolvingDynamicDimensions([1, 1, vocabSize]))

        var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
        var posIds = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, 1]))
        fillNDArray(&ids, as: Int32.self, with: [startToken])
        fillNDArray(&posIds, as: Int32.self, count: 1) { Int32($0) }
        var st = InferenceFunction.MutableViews()
        st.insert(&keyCache, for: "keyCache")
        st.insert(&valueCache, for: "valueCache")
        var out = InferenceFunction.MutableViews()
        out.insert(&logitsArray, for: "logits")
        _ = try await decFn.run(
            inputs: [
                "input_ids": ids, "position_ids": posIds,
                "encoder_hidden_states": encoderOutput,
            ],
            states: consume st, outputViews: consume out)

        let logits = flattenAsFloat(logitsArray)
        return candidates
            .filter { $0 >= 0 && Int($0) < logits.count }
            .max { logits[Int($0)] < logits[Int($1)] }
    }

    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources
    ) async throws -> (tokens: [Int32], stats: DecodeStats) {
        try await decode(
            encoderOutput: encoderOutput, encoderOutputShape: encoderOutputShape,
            validEncoderFrames: validEncoderFrames, resources: resources, onToken: nil)
    }

    /// Greedy decode of one window, handing each generated token to `onToken` as it is chosen.
    ///
    /// The forced prefix is not reported, and neither is the closing `eotToken`. Cancellation
    /// is checked before every decoder step, so a cancelled task stops within one step rather
    /// than decoding the rest of the window.
    ///
    /// - Parameter capturesAlignment: also keep each generated step's `alignment_logits` in
    ///   `stats.alignment`, for word timestamps. Throws if the decoder was exported without
    ///   them.
    public func decode(
        encoderOutput: NDArray,
        encoderOutputShape: [Int],
        validEncoderFrames: Int,
        resources: DecoderResources,
        onToken: (@Sendable (Int32) async throws -> Void)?,
        capturesAlignment: Bool = false
    ) async throws -> (tokens: [Int32], stats: DecodeStats) {
        // `validEncoderFrames` is ignored: Whisper is encoder-decoder and stops on
        // `eotToken`, not on encoder-frame exhaustion, so padded encoder frames don't
        // produce trailing tokens the way the Parakeet TDT frame loop does.
        guard case .whisper(let decoderModel, let config) = resources else {
            throw SpeechError.incompatibleResources("WhisperDecoder requires .whisper resources")
        }
        guard let decFn = try decoderModel.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in decoder")
        }
        let decDesc = decoderModel.functionDescriptor(for: "main")!

        guard case .ndArray(let inputIdsNDDesc) = decDesc.inputDescriptor(of: "input_ids"),
            case .ndArray(let posIdsNDDesc) = decDesc.inputDescriptor(of: "position_ids"),
            case .ndArray(_) = decDesc.inputDescriptor(of: "encoder_hidden_states"),
            case .ndArray(let keyCacheNDDesc) = decDesc.stateDescriptor(of: "keyCache"),
            case .ndArray(let valCacheNDDesc) = decDesc.stateDescriptor(of: "valueCache"),
            case .ndArray(let logitsNDDesc) = decDesc.outputDescriptor(of: "logits")
        else { throw SpeechError.missingModel("Unexpected decoder descriptors") }

        let vocabSize = logitsNDDesc.shape.last!
        let maxTargetPos = 448
        let kcShape = keyCacheNDDesc.shape.map { $0 < 0 ? maxTargetPos : $0 }
        let vcShape = valCacheNDDesc.shape.map { $0 < 0 ? maxTargetPos : $0 }
        var keyCache = NDArray(descriptor: keyCacheNDDesc.resolvingDynamicDimensions(kcShape))
        var valueCache = NDArray(descriptor: valCacheNDDesc.resolvingDynamicDimensions(vcShape))
        var logitsArray = NDArray(descriptor: logitsNDDesc.resolvingDynamicDimensions([1, 1, vocabSize]))

        // `[1, heads, frames]`, bound only when asked for: the graph computes it every step
        // regardless, but copying it out is wasted work for a plain transcription.
        // A placeholder until then, since a view has to be bound to a non-optional array.
        var alignmentArray = NDArray(shape: [1], scalarType: .float32)
        var alignment: WhisperAlignment.Scores?
        if capturesAlignment {
            guard decDesc.outputNames.contains(Self.alignmentOutputName),
                case .ndArray(let desc) = decDesc.outputDescriptor(of: Self.alignmentOutputName),
                desc.shape.count == 3, desc.shape.allSatisfy({ $0 > 0 })
            else {
                throw SpeechError.incompatibleResources(
                    "This Whisper decoder has no '\(Self.alignmentOutputName)' output, so it cannot "
                        + "time words. Re-export the bundle with models/whisper/export_kv.py.")
            }
            alignmentArray = NDArray(descriptor: desc)
            alignment = WhisperAlignment.Scores(heads: desc.shape[1], frames: desc.shape[2])
        }

        // A `--static-positions` export fixes `position_ids` at [1, 1] and takes the current
        // position alone; the dynamic export takes 0...pos, whose length is the cache length.
        let staticPositions = posIdsNDDesc.shape.last == 1

        func step(_ tok: Int32, pos: Int) async throws {
            try Task.checkCancellation()
            var ids = NDArray(descriptor: inputIdsNDDesc.resolvingDynamicDimensions([1, 1]))
            fillNDArray(&ids, as: Int32.self, with: [tok])
            var posIds: NDArray
            if staticPositions {
                posIds = NDArray(descriptor: posIdsNDDesc)
                fillNDArray(&posIds, as: Int32.self, with: [Int32(pos)])
            } else {
                posIds = NDArray(descriptor: posIdsNDDesc.resolvingDynamicDimensions([1, pos + 1]))
                fillNDArray(&posIds, as: Int32.self, count: pos + 1) { Int32($0) }
            }
            let inputs: [String: NDArray] = [
                "input_ids": ids, "position_ids": posIds,
                "encoder_hidden_states": encoderOutput,
            ]
            var st = InferenceFunction.MutableViews()
            st.insert(&keyCache, for: "keyCache")
            st.insert(&valueCache, for: "valueCache")
            var out = InferenceFunction.MutableViews()
            out.insert(&logitsArray, for: "logits")
            // two calls rather than a conditional insert, which the views' lifetime check rejects
            guard alignment != nil else {
                _ = try await decFn.run(inputs: inputs, states: consume st, outputViews: consume out)
                return
            }
            out.insert(&alignmentArray, for: Self.alignmentOutputName)
            _ = try await decFn.run(inputs: inputs, states: consume st, outputViews: consume out)
        }

        var timing = DecodeStats.Timing()

        // Prime KV cache with forced prefix
        var tokens: [Int32] = config.forcedPrefix
        let prefixStart = ContinuousClock.now
        for (i, tok) in config.forcedPrefix.enumerated() {
            try await step(tok, pos: i)
        }
        timing.prefixMs = (ContinuousClock.now - prefixStart).inMilliseconds
        timing.prefixSteps = config.forcedPrefix.count

        // Greedy decode
        var stepTimesMs: [Double] = []
        var pos = config.forcedPrefix.count
        while tokens.count - config.forcedPrefix.count < config.maxDecodeSteps {
            let t0 = ContinuousClock.now
            try await step(tokens.last!, pos: pos)
            let t1 = ContinuousClock.now
            stepTimesMs.append((t1 - t0).inMilliseconds)
            let logits = flattenAsFloat(logitsArray)
            let next = Int32(logits.indices.max(by: { logits[$0] < logits[$1] })!)
            // one row per generated token, eot included: it is what ends the last word
            if alignment != nil { alignment!.values += flattenAsFloat(alignmentArray) }
            timing.samplingMs += (ContinuousClock.now - t1).inMilliseconds
            tokens.append(next)
            pos += 1
            if next == config.eotToken { break }
            try await onToken?(next)
        }

        var stats = DecodeStats(stepTimesMs: stepTimesMs, timing: timing)
        stats.alignment = alignment
        return (tokens: tokens, stats: stats)
    }

    /// The decoder output `export_kv.py` writes the alignment heads' cross-attention scores to.
    package static let alignmentOutputName = "alignment_logits"
}

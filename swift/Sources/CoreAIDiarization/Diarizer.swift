// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAISpeech
import Foundation

/// Each speaker's activity probability every `frameSeconds`, as `[frames, speakers]` row-major.
/// Speakers are numbered in order of first arrival.
public struct SpeakerActivity: Sendable {
    public let probabilities: [Float]
    public let numFrames: Int
    public let numSpeakers: Int
    public let frameSeconds: Double

    public init(probabilities: [Float], numFrames: Int, numSpeakers: Int, frameSeconds: Double) {
        self.probabilities = probabilities
        self.numFrames = numFrames
        self.numSpeakers = numSpeakers
        self.frameSeconds = frameSeconds
    }

    public func probability(frame: Int, speaker: Int) -> Float {
        probabilities[frame * numSpeakers + speaker]
    }
}

/// What a diarization run cost, for benchmarking.
/// One step's worth of a diarization in progress: the speaker activity of the frames it settled, which
/// follow the frames of every update before it. Concatenated, the updates are exactly the activity
/// ``Diarizer/diarize(pcm:onUpdate:)`` returns: a frame is never revised once reported.
public struct DiarizationUpdate: Sendable {
    /// The new frames' activity.
    public let activity: SpeakerActivity
    /// The index of its first frame in the whole recording's activity.
    public let startFrame: Int

    public init(activity: SpeakerActivity, startFrame: Int) {
        self.activity = activity
        self.startFrame = startFrame
    }

    /// The frames reported so far, this update's included.
    public var endFrame: Int { startFrame + activity.numFrames }
    /// How far into the audio the reported frames reach, in seconds.
    public var endSeconds: Double { Double(endFrame) * activity.frameSeconds }
}

public struct DiarizationTiming: Sendable, Codable {
    public var featuresMs: Double = 0
    /// Core AI step runs.
    public var stepsMs: Double = 0
    /// Padding, output reads and the speaker-cache policy, on the CPU.
    public var hostMs: Double = 0
    public var steps = 0
}

/// The exported bundle's `metadata.json`, as `export.py` writes it.
public struct DiarizationMetadata: Sendable, Codable {
    public struct Mode: Sendable, Codable {
        public let asset: String
        public let chunkFrames: Int
        public let lookaheadFrames: Int
        public let fifoLength: Int
        public let speakerCacheUpdatePeriod: Int
        public let cacheFrames: Int
        public let windowFrames: Int
    }

    public let model: String
    public let dtype: String
    public let subsamplingFactor: Int
    public let hiddenSize: Int
    public let numSpeakers: Int
    public let silenceEmbeds: String
    public let speakerCache: SpeakerCachePolicy
    public let modes: [String: Mode]
}

public enum DiarizationError: LocalizedError {
    case unknownMode(String)
    case noLoadableAsset(String, underlying: Error?)
    case missingFunction(String)
    /// ``Diarizer/repeatStep()`` was called before ``Diarizer/runStep(cachedEmbeds:chunkFeatures:keyMask:)``.
    case noStepToRepeat
    /// A loading step failed; `stage` says which, since some framework errors carry no detail of their own.
    case loadFailed(stage: String, underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .unknownMode(let mode): "The bundle has no '\(mode)' mode."
        case .noLoadableAsset(let mode, let error): "No asset for '\(mode)' loaded: \(error.map { "\($0)" } ?? "none found")"
        case .missingFunction(let name): "The step asset is missing '\(name)'."
        case .noStepToRepeat: "No step has run yet, so there is none to repeat."
        case .loadFailed(let stage, let error):
            "Loading the diarizer failed at \(stage): \(error) (\((error as NSError).domain) \((error as NSError).code))"
        }
    }
}

/// Runs Nemotron-3-Diarization over audio with the exported static-shape step, keeping the
/// streaming loop and speaker-cache policy on the host, as `verify.py`'s `coreai_session` does.
public actor Diarizer {
    public let metadata: DiarizationMetadata
    public let mode: String
    public let geometry: DiarizationMetadata.Mode
    /// File name of the asset that loaded, e.g. `step_low_latency.h16p.aimodelc`.
    public let assetName: String

    private let step: StepFunction
    private let silenceEmbeds: [Float]
    private let melBasis: MelSpectrogram.Basis

    /// Loads one mode of the bundle at `url`.
    ///
    /// Tries the mode's ahead-of-time compiled `<asset>.<arch>.aimodelc` files first (a wrong
    /// architecture fails to load cleanly), then the source `.aimodel`. Ship compiled assets
    /// on iOS where you can: on iPhone 15 Pro the compiled step is 2.5x faster than the source
    /// specialized on device.
    ///
    /// `allowsCompiledAssets: false` skips the compiled assets and specializes the source on this
    /// device, with `options`.
    public init(
        bundleAt url: URL, mode: String, options: SpecializationOptions? = nil, allowsCompiledAssets: Bool = true
    ) async throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let metadata = try Self.stage("metadata") {
            try decoder.decode(DiarizationMetadata.self, from: Data(contentsOf: url.appending(path: "metadata.json")))
        }
        guard let geometry = metadata.modes[mode] else { throw DiarizationError.unknownMode(mode) }

        let base = geometry.asset.replacingOccurrences(of: ".aimodel", with: "")
        let files = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        let candidates =
            files.filter {
                allowsCompiledAssets && $0.pathExtension == "aimodelc" && $0.lastPathComponent.hasPrefix(base + ".")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            + files.filter { $0.lastPathComponent == geometry.asset }
        var loaded: (AIModel, URL)?
        var lastError: Error?
        for candidate in candidates {
            do {
                loaded = (try await Self.loadModel(candidate, options: options), candidate)
                break
            } catch {
                lastError = error
            }
        }
        guard case (var model, let assetURL)? = loaded else { throw DiarizationError.noLoadableAsset(mode, underlying: lastError) }
        let name = assetURL.lastPathComponent
        let step: StepFunction
        do {
            step = try StepFunction(model: model, mode: mode)
        } catch {
            // A cached specialization left behind by an interrupted one loads instantly and then returns no
            // function (`nilError`) — seen on iPhone after an app was killed mid-specialization. Drop it and
            // specialize afresh, once.
            try? AIModelCache.default.deleteEntries(for: assetURL)
            model = try await Self.loadModel(assetURL, options: options)
            step = try Self.stage("function \(mode) of \(name)") { try StepFunction(model: model, mode: mode) }
        }
        let silencePath = url.appending(path: metadata.silenceEmbeds)
        let data = try Self.stage("silence embeddings") { try Data(contentsOf: silencePath) }

        self.metadata = metadata
        self.mode = mode
        self.geometry = geometry
        self.assetName = name
        self.step = step
        self.silenceEmbeds = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        self.melBasis = MelSpectrogram.Basis(config: .nemotronDiarization)
    }

    private static func loadModel(_ url: URL, options: SpecializationOptions?) async throws -> AIModel {
        if let options { try await AIModel(contentsOf: url, options: options) } else { try await AIModel(contentsOf: url) }
    }

    /// Runs one loading step, naming it in the error: some framework errors carry no detail of their own.
    private static func stage<T>(_ name: String, _ body: () throws -> T) throws -> T {
        do { return try body() } catch let error as DiarizationError { throw error } catch {
            throw DiarizationError.loadFailed(stage: name, underlying: error)
        }
    }

    /// Diarizes 16 kHz mono `pcm`.
    ///
    /// The whole recording's features are computed once and cut into the mode's chunks. For
    /// `offline` this is exactly the HF forward; for a streaming mode it replays the HF
    /// streaming loop over full-pass features, which the HF processor reproduces frame for
    /// frame, so the steps, cache and outputs are what a live session would compute.
    ///
    /// - Parameter onUpdate: receives each step's frames as soon as the step settles them, in order,
    ///   for a caller that shows speakers as they are found. A chunk's frames are final once its step
    ///   has run, so nothing reported is revised. One step covers the mode's chunk: 30 s offline, about
    ///   1 s at low latency.
    public func diarize(
        pcm: [Float], onUpdate: (@Sendable (DiarizationUpdate) -> Void)? = nil
    ) async throws -> (activity: SpeakerActivity, timing: DiarizationTiming) {
        var timing = DiarizationTiming()
        let clock = ContinuousClock()
        var mark = clock.now

        let config = MelConfig.nemotronDiarization
        let features = MelSpectrogram.fromPCM(pcm, config: config, basis: melBasis)
        let numMelBins = config.nMelBins
        let totalMelFrames = features.count / numMelBins
        let validMelFrames = MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: config)
        timing.featuresMs = (clock.now - mark).milliseconds

        let subsampling = metadata.subsamplingFactor
        let hidden = metadata.hiddenSize
        let numSpeakers = metadata.numSpeakers
        let numEmbeds = (totalMelFrames + subsampling - 1) / subsampling
        // an encoder frame is valid when the first mel frame it stacks is
        let embedValid = (0..<numEmbeds).map { $0 * subsampling < validMelFrames }

        var cache = SpeakerCache(
            policy: metadata.speakerCache, fifoLength: geometry.fifoLength,
            updatePeriod: geometry.speakerCacheUpdatePeriod, numSpeakers: numSpeakers, hiddenSize: hidden,
            subsampling: subsampling)
        let windowChunk = geometry.chunkFrames + geometry.lookaheadFrames
        var output = [Float]()
        output.reserveCapacity(totalMelFrames * numSpeakers)
        let frameSeconds = Double(config.hopLength) / config.sampleRate
        var reportedFrames = 0

        for (start, numChunkFrames, numLookahead) in chunks(numEmbeds: numEmbeds) {
            // a streaming mode takes thousands of steps over a long meeting
            try Task.checkCancellation()
            let numStepEmbeds = numChunkFrames + numLookahead
            mark = clock.now
            let cached = cache.cachedEmbeds
            let numCached = cache.numCachedFrames

            // left-pad the cache, right-pad the chunk; see export.py for why left
            var cachedInput = [Float](repeating: 0, count: geometry.cacheFrames * hidden)
            cachedInput.replaceSubrange(((geometry.cacheFrames - numCached) * hidden)..., with: cached)
            var chunkInput = [Float](repeating: 0, count: windowChunk * subsampling * numMelBins)
            let melStart = start * subsampling
            let melEnd = min((start + numStepEmbeds) * subsampling, totalMelFrames)
            chunkInput.replaceSubrange(
                0..<((melEnd - melStart) * numMelBins),
                with: features[(melStart * numMelBins)..<(melEnd * numMelBins)])
            var keyMask = [Float](repeating: 0, count: geometry.cacheFrames + windowChunk)
            for i in (geometry.cacheFrames - numCached)..<geometry.cacheFrames { keyMask[i] = 1 }
            let chunkValid = Array(embedValid[start..<(start + numStepEmbeds)])
            for (i, valid) in chunkValid.enumerated() where valid { keyMask[geometry.cacheFrames + i] = 1 }
            timing.hostMs += (clock.now - mark).milliseconds

            mark = clock.now
            try await step.run(cachedEmbeds: cachedInput, chunkFeatures: chunkInput, keyMask: keyMask)
            timing.stepsMs += (clock.now - mark).milliseconds
            timing.steps += 1

            mark = clock.now
            let (logits, chunkEmbeds) = step.readOutputs()
            // drop the cache's left padding and the chunk's right padding
            let offset = geometry.cacheFrames - numCached
            let stepLogits = Array(
                logits[(offset * subsampling * numSpeakers)..<((geometry.cacheFrames + numStepEmbeds) * subsampling * numSpeakers)])
            let inputEmbeds = cached + chunkEmbeds[0..<(numStepEmbeds * hidden)]
            cache.update(
                inputEmbeds: inputEmbeds, logits: stepLogits, silenceEmbeds: silenceEmbeds,
                numChunkFrames: numChunkFrames,
                mask: [Bool](repeating: true, count: numCached) + chunkValid)

            let first = numCached * subsampling
            let last = min((numCached + numChunkFrames) * subsampling, first + (melEnd - melStart))
            output += stepLogits[(first * numSpeakers)..<(last * numSpeakers)]
            if let onUpdate {
                // the same frames, and the same sigmoid, as the activity returned at the end
                let settled = min(output.count / numSpeakers, totalMelFrames)
                if settled > reportedFrames {
                    let probabilities = output[(reportedFrames * numSpeakers)..<(settled * numSpeakers)]
                        .map { 1 / (1 + exp(-$0)) }
                    onUpdate(
                        DiarizationUpdate(
                            activity: SpeakerActivity(
                                probabilities: probabilities, numFrames: settled - reportedFrames,
                                numSpeakers: numSpeakers, frameSeconds: frameSeconds),
                            startFrame: reportedFrames))
                    reportedFrames = settled
                }
            }
            timing.hostMs += (clock.now - mark).milliseconds
        }

        let numFrames = min(output.count / numSpeakers, totalMelFrames)
        let probabilities = output.prefix(numFrames * numSpeakers).map { 1 / (1 + exp(-$0)) }
        let activity = SpeakerActivity(
            probabilities: probabilities, numFrames: numFrames, numSpeakers: numSpeakers, frameSeconds: frameSeconds)
        return (activity, timing)
    }

    // MARK: - Single steps

    /// Runs one step on inputs already padded to the mode's static window, as `export.py` lays
    /// them out, and returns both outputs as float32 over the whole window.
    ///
    /// For checking the compiled step against `verify.py --fixture-dir` fixtures, whose inputs are
    /// exactly these; ``diarize(pcm:)`` drives the streaming loop itself.
    public func runStep(cachedEmbeds: [Float], chunkFeatures: [Float], keyMask: [Float]) async throws
        -> (logits: [Float], chunkEmbeds: [Float])
    {
        try await step.run(cachedEmbeds: cachedEmbeds, chunkFeatures: chunkFeatures, keyMask: keyMask)
        return step.readOutputs()
    }

    /// Runs the most recent step's inputs again, leaving the outputs unread: what a step latency
    /// measurement should time, since filling the inputs and reading the outputs are host work.
    public func repeatStep() async throws {
        try await step.rerun()
    }

    /// `(first encoder frame, chunk frames, look-ahead frames)` of each step.
    ///
    /// Offline follows the HF forward: chunks of `chunkFrames` taking up to `lookaheadFrames`
    /// from the next. Streaming follows the HF processor: full chunks while a chunk plus its
    /// look-ahead fits, then one last chunk of whatever remains, scored whole.
    private func chunks(numEmbeds: Int) -> [(Int, Int, Int)] {
        let (chunk, lookahead) = (geometry.chunkFrames, geometry.lookaheadFrames)
        var result: [(Int, Int, Int)] = []
        if mode == "offline" {
            for start in stride(from: 0, to: numEmbeds, by: chunk) {
                let end = min(start + chunk, numEmbeds)
                result.append((start, end - start, min(end + lookahead, numEmbeds) - end))
            }
            return result
        }
        var start = 0
        while start + chunk + lookahead <= numEmbeds {
            result.append((start, chunk, lookahead))
            start += chunk
        }
        if start < numEmbeds { result.append((start, numEmbeds - start, 0)) }
        return result
    }
}

/// One mode's step function, with its output buffers allocated once and bound on every run.
/// Letting the runtime allocate outputs per call left hundreds of MB unreclaimed per step on iPhone.
///
/// `@unchecked Sendable`: only the owning `Diarizer` touches it, one step at a time, but its
/// nonisolated `run` hands it across the actor boundary.
private final class StepFunction: @unchecked Sendable {
    let function: InferenceFunction
    let cachedDescriptor: NDArrayDescriptor
    let chunkDescriptor: NDArrayDescriptor
    let maskDescriptor: NDArrayDescriptor
    private var logits: NDArray
    private var chunkEmbeds: NDArray
    private var inputs: [String: NDArray]?

    init(model: AIModel, mode: String) throws {
        guard let function = try model.loadFunction(named: mode), let descriptor = model.functionDescriptor(for: mode)
        else { throw DiarizationError.missingFunction(mode) }
        guard case .ndArray(let cached) = descriptor.inputDescriptor(of: "cached_embeds"),
            case .ndArray(let chunk) = descriptor.inputDescriptor(of: "chunk_features"),
            case .ndArray(let mask) = descriptor.inputDescriptor(of: "key_mask"),
            case .ndArray(let logits) = descriptor.outputDescriptor(of: "logits"),
            case .ndArray(let embeds) = descriptor.outputDescriptor(of: "chunk_embeds")
        else { throw DiarizationError.missingFunction("\(mode) inputs/outputs") }
        self.function = function
        cachedDescriptor = cached
        chunkDescriptor = chunk
        maskDescriptor = mask
        self.logits = NDArray(descriptor: logits.resolvingDynamicDimensions(logits.shape))
        chunkEmbeds = NDArray(descriptor: embeds.resolvingDynamicDimensions(embeds.shape))
    }

    func run(cachedEmbeds: [Float], chunkFeatures: [Float], keyMask: [Float]) async throws {
        inputs = [
            "cached_embeds": makeArray(cachedDescriptor, cachedEmbeds),
            "chunk_features": makeArray(chunkDescriptor, chunkFeatures),
            "key_mask": makeArray(maskDescriptor, keyMask),
        ]
        try await rerun()
    }

    /// Runs the inputs of the last ``run(cachedEmbeds:chunkFeatures:keyMask:)`` again.
    func rerun() async throws {
        guard let inputs else { throw DiarizationError.noStepToRepeat }
        try await Self.run(function, inputs: inputs, logits: &logits, chunkEmbeds: &chunkEmbeds)
    }

    func readOutputs() -> (logits: [Float], chunkEmbeds: [Float]) {
        (readFloats(logits), readFloats(chunkEmbeds))
    }

    // The views must not outlive the buffers they borrow, so they bind `inout` parameters.
    private static func run(
        _ function: InferenceFunction, inputs: [String: NDArray], logits: inout NDArray, chunkEmbeds: inout NDArray
    ) async throws {
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&logits, for: "logits")
        outputs.insert(&chunkEmbeds, for: "chunk_embeds")
        _ = try await function.run(inputs: inputs, states: InferenceFunction.MutableViews(), outputViews: consume outputs)
    }
}

extension Duration {
    var milliseconds: Double {
        Double(components.seconds) * 1000 + Double(components.attoseconds) * 1e-15
    }
}

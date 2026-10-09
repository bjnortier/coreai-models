// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers

// MARK: - SpeechRecognitionModel

/// On-device speech recognition model.
///
/// Loads a CoreAISpeech bundle and transcribes audio. Supports both Whisper-style
/// encoder-decoder bundles and Parakeet TDT bundles; the architecture is
/// auto-detected from the bundle's metadata.json (or the legacy
/// encoder/decoder filename convention for Whisper).
public actor SpeechRecognitionModel {
    /// Audio shared between consecutive windows when transcribing long-form audio.
    ///
    /// Long enough to contain a word or two of real speech at a normal speaking
    /// rate, which is what seam matching needs to find its anchor. Longer costs
    /// proportionally more decode time for no extra reliability.
    public static let defaultOverlapSeconds: Double = 2.0

    package let bundle: SpeechRecognitionBundle
    package let decoder: any SpeechDecoder
    package let melConfig: MelConfig
    private let resources: DecoderResources
    /// Non-nil for Parakeet bundles. Streaming geometry and the exact
    /// `ceil(validMel / subsampling)` frame rule are both derived from it.
    package let tdtConfig: ParakeetTDTConfig?
    /// Streaming session state, non-nil between `startStream` and `finishStream`.
    package var streaming: StreamingSessionState?
    /// Window geometry recorded by a `--streaming` export. Authoritative when present.
    package let bundleStreamingConfig: StreamingConfig?

    /// Encoder function and its input descriptors, resolved once at init. These are
    /// shape-independent, so caching them keeps `loadFunction`/descriptor lookups off
    /// the per-transcription path; only `resolvingDynamicDimensions` is per-call.
    private let encoderFunction: InferenceFunction
    private let melInputDescriptor: NDArrayDescriptor
    /// Non-nil only for bundles exported with the optional `attention_mask` input.
    private let maskInputDescriptor: NDArrayDescriptor?

    /// Mel window/DFT/filterbank, built once for `melConfig`. Derived purely from the
    /// config, so rebuilding it on every transcription is wasted work.
    private let melBasis: MelSpectrogram.Basis

    /// - Parameter computeUnits: which compute unit each stage is specialized for.
    ///   The default is measured on Mac; see ``SpeechComputeUnits`` before relying
    ///   on it elsewhere.
    public init(
        resourcesAt url: URL, computeUnits: SpeechComputeUnits = .default
    ) async throws {
        self.bundle = try await SpeechRecognitionBundle(at: url, computeUnits: computeUnits)
        let encoder: AIModel
        switch bundle.kind {
        case .whisper(let assets):
            self.decoder = WhisperDecoder()
            self.tdtConfig = nil
            self.bundleStreamingConfig = nil
            self.melConfig = assets.melConfig
            self.resources = .whisper(decoder: assets.decoder, generationConfig: assets.generationConfig)
            encoder = assets.encoder
        case .parakeetTDT(let assets):
            self.decoder = try ParakeetTDTDecoder(decoderStep: assets.decoderStep, joint: assets.joint)
            self.tdtConfig = assets.config
            self.bundleStreamingConfig = assets.streamingConfig
            self.melConfig = assets.melConfig
            self.resources = .parakeetTDT(
                decoderStep: assets.decoderStep, joint: assets.joint, config: assets.config)
            encoder = assets.encoder
        }
        self.melBasis = MelSpectrogram.Basis(config: self.melConfig)

        guard let fn = try encoder.loadFunction(named: "main") else {
            throw SpeechError.missingModel("No 'main' function in encoder")
        }
        guard let encDesc = encoder.functionDescriptor(for: "main") else {
            throw SpeechError.missingModel("No 'main' descriptor in encoder")
        }
        guard case .ndArray(let melNDDesc) = encDesc.inputDescriptor(of: "input_features")
        else { throw SpeechError.missingModel("Unexpected encoder input descriptor") }
        self.encoderFunction = fn
        self.melInputDescriptor = melNDDesc
        if encDesc.inputNames.contains("attention_mask"),
            case .ndArray(let maskDesc) = encDesc.inputDescriptor(of: "attention_mask")
        {
            self.maskInputDescriptor = maskDesc
        } else {
            self.maskInputDescriptor = nil
        }

        try await warmUp()
    }

    /// Human-readable architecture label (for logging).
    public var architecture: String {
        switch bundle.kind {
        case .whisper: return "Whisper"
        case .parakeetTDT: return "Parakeet TDT"
        }
    }

    public var sampleRate: Double { melConfig.sampleRate }

    /// The mel features this model would hand its encoder for `pcm`, with the shape
    /// the encoder expects them in.
    ///
    /// The transcription path computes these internally; this exposes them so a
    /// verification harness can compare the Swift front-end against a reference
    /// implementation without re-deriving the bundle's `MelConfig`.
    public func melFeatures(pcm: [Float]) -> (values: [Float], shape: [Int]) {
        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig, basis: melBasis)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        return (mel, encoderInputShape(nFrames: nFrames))
    }

    /// Warm the encoder and decoder for a specific PCM sample count so MPSGraph
    /// compiles and caches graphs for that input shape before real audio arrives.
    ///
    /// The encoder runs at the full `sampleCount`, since its graph is compiled per input
    /// shape and that is the compilation worth paying for up front. The decoder graphs are
    /// shape-static, so the frame loop is capped at one frame rather than decoding the
    /// whole silence buffer — one pass compiles the same graphs.
    public func prewarm(sampleCount: Int) async throws {
        let (encOut, encShape, _) = try await runEncoder(
            pcm: [Float](repeating: 0, count: sampleCount))
        _ = try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: 1,
            resources: resources)
    }

    // MARK: - Transcription

    /// Transcribe an audio file, returning the full text and decode stats.
    ///
    /// Audio longer than a static encoder's window is decoded as several
    /// overlapping windows and stitched — see
    /// ``transcribe(pcm:overlapSeconds:language:task:resetAfterSilenceFrames:)``.
    /// - Parameter resetAfterSilenceFrames: See `ParakeetTDTDecoder.decode`. 0 keeps this path
    ///   reference-exact.
    public func transcribe(
        audioURL: URL,
        overlapSeconds: Double = SpeechRecognitionModel.defaultOverlapSeconds,
        language: SpeechLanguage = .detect,
        task: SpeechTask = .transcribe,
        resetAfterSilenceFrames: Int = 0,
        onPartial: (@Sendable (SpeechTranscriptionUpdate) -> Void)? = nil
    ) async throws -> (String, DecodeStats) {
        let pcm = try MelSpectrogram.loadAndResample(
            audioURL, targetSampleRate: melConfig.sampleRate)
        return try await transcribe(
            pcm: pcm, overlapSeconds: overlapSeconds, language: language, task: task,
            resetAfterSilenceFrames: resetAfterSilenceFrames, onPartial: onPartial)
    }

    /// Transcribe raw 16 kHz mono PCM samples, returning the full text and decode stats.
    ///
    /// A static encoder sees a fixed span of audio per pass — Whisper's is 30 s —
    /// so anything longer is split into consecutive windows that overlap by
    /// `overlapSeconds`, decoded independently, and joined with the duplicated
    /// words at each seam removed. `stats.windowCount` reports how many were used.
    ///
    /// Pass `0` to butt the windows against each other with no overlap and no
    /// de-duplication. Bundles with a dynamic encoder (Parakeet's, unless exported
    /// at a static length) take the whole waveform in one pass and ignore this.
    /// - Parameter language: which language to decode as. Defaults to detecting it
    ///   from the audio; pass ``SpeechLanguage/code(_:)`` to force one, or
    ///   ``SpeechLanguage/bundleDefault`` to keep whatever the export pinned.
    ///   Ignored by Parakeet, which decodes multilingual audio directly.
    /// - Parameter task: whether to write the audio down as spoken (the default)
    ///   or translate it into English. `language` still names the *spoken*
    ///   language when translating. Parakeet has no task slot and throws rather
    ///   than transcribing when asked to translate.
    /// - Parameter resetAfterSilenceFrames: See
    ///   ``transcribe(audioURL:overlapSeconds:language:task:resetAfterSilenceFrames:)``.
    ///   Applied per window: the predictor state does not cross a seam anyway,
    ///   since each window is decoded independently.
    public func transcribe(
        pcm: [Float],
        overlapSeconds: Double = SpeechRecognitionModel.defaultOverlapSeconds,
        language: SpeechLanguage = .detect,
        task: SpeechTask = .transcribe,
        resetAfterSilenceFrames: Int = 0,
        onPartial: (@Sendable (SpeechTranscriptionUpdate) -> Void)? = nil
    ) async throws -> (String, DecodeStats) {
        guard let windowSamples = staticWindowSamples, pcm.count > windowSamples else {
            let resolveStart = ContinuousClock.now
            let resolved = try await resolvedResources(
                forLeadingPCM: pcm, language: language, task: task)
            let languageDetectionMs = (ContinuousClock.now - resolveStart).inMilliseconds
            let reporter = tokenReporter(
                completedTexts: [], windowIndex: 0, windowCount: 1,
                startSample: 0, endSample: pcm.count, previousEndSample: 0, totalSamples: pcm.count, emit: onPartial)
            var (tokens, stats) = try await decodeAudio(
                pcm: pcm, resources: resolved,
                resetAfterSilenceFrames: resetAfterSilenceFrames, reporter: reporter)
            stats.timing.languageDetectionMs = languageDetectionMs
            let text = try detokenize(tokens)
            onPartial?(SpeechTranscriptionUpdate(text: text, source: .window, windowIndex: 0, windowCount: 1, progress: 1))
            return (text, stats)
        }
        let overlapSamples = Int(max(0, overlapSeconds) * melConfig.sampleRate)
        let starts = LongForm.windowStarts(
            pcmCount: pcm.count, windowSamples: windowSamples, overlapSamples: overlapSamples)

        // Detect once, on the first window, and reuse it for the rest: a recording
        // is almost always one language, and re-detecting per window would both
        // cost a step each time and let a noisy window flip the language
        // mid-transcript, producing a hybrid transcript from a monolingual file.
        let firstEnd = min(starts[0] + windowSamples, pcm.count)
        let resolveStart = ContinuousClock.now
        let resolved = try await resolvedResources(
            forLeadingPCM: Array(pcm[starts[0]..<firstEnd]), language: language, task: task)
        let languageDetectionMs = (ContinuousClock.now - resolveStart).inMilliseconds

        var texts: [String] = []
        var stats: [DecodeStats] = []
        for (index, start) in starts.enumerated() {
            // Each window is a full encoder pass; a cancelled task should not start another.
            try Task.checkCancellation()
            let end = min(start + windowSamples, pcm.count)
            let previousEnd = index == 0 ? 0 : min(starts[index - 1] + windowSamples, pcm.count)
            let reporter = tokenReporter(
                completedTexts: texts, windowIndex: index, windowCount: starts.count,
                startSample: start, endSample: end, previousEndSample: previousEnd,
                totalSamples: pcm.count, emit: onPartial)
            let (tokens, windowStats) = try await decodeAudio(
                pcm: Array(pcm[start..<end]), resources: resolved,
                resetAfterSilenceFrames: resetAfterSilenceFrames, reporter: reporter)
            texts.append(try detokenize(tokens))
            stats.append(windowStats)
            onPartial?(SpeechTranscriptionUpdate(
                text: LongForm.stitch(texts), source: .window, windowIndex: index, windowCount: starts.count,
                progress: Double(end) / Double(pcm.count)))
        }
        var merged = DecodeStats.merged(stats)
        merged.timing.languageDetectionMs = languageDetectionMs
        return (LongForm.stitch(texts), merged)
    }

    /// The language this request would decode as, as a code such as `"fr"`.
    ///
    /// `nil` for bundles that do not select a language (Parakeet).
    public func resolveLanguage(
        pcm: [Float], language: SpeechLanguage = .detect
    ) async throws -> String? {
        let resolved = try await resolvedResources(
            forLeadingPCM: pcm, language: language, task: .transcribe)
        guard case .whisper(_, let config) = resolved, let token = config.languageToken else {
            return nil
        }
        return bundle.languageTokens.codeByID[token]
    }

    /// The decoder resources for this request, with the language resolved.
    ///
    /// Detection runs the decoder once over `leadingPCM`, so it happens a single
    /// time per transcription rather than per window.
    private func resolvedResources(
        forLeadingPCM leadingPCM: [Float], language: SpeechLanguage, task: SpeechTask
    ) async throws -> DecoderResources {
        guard case .whisper(let decoderModel, let config) = resources else {
            // Parakeet decodes multilingual audio directly, and has no task slot to
            // put `<|translate|>` in. Thrown rather than ignored: a silently
            // untranslated transcript is fluent, plausible, and in the wrong language.
            guard task == .transcribe else {
                throw SpeechError.incompatibleResources(
                    "\(architecture) cannot translate: the task is named in Whisper's decoder "
                        + "prefix, and this architecture has none.")
            }
            return resources
        }
        var resolved = try await languageResolved(
            config: config, leadingPCM: leadingPCM, language: language)
        if task == .translateToEnglish {
            guard let token = bundle.taskTokens.token(for: task) else {
                throw SpeechError.incompatibleResources(
                    "This bundle ships no <|translate|> token, so it cannot translate.")
            }
            resolved = resolved.retargeted(toTaskToken: token)
        }
        return .whisper(decoder: decoderModel, generationConfig: resolved)
    }

    /// `config` with its language slot set to whatever `language` asks for.
    ///
    /// Split out from ``resolvedResources(forLeadingPCM:language:task:)`` so the two
    /// prefix slots are retargeted independently: a translation still names the
    /// *spoken* language in slot 1, so detection has to run for it as well.
    private func languageResolved(
        config: GenerationConfig, leadingPCM: [Float], language: SpeechLanguage
    ) async throws -> GenerationConfig {
        let table = bundle.languageTokens

        switch language {
        case .bundleDefault:
            return config

        case .code(let code):
            guard let token = table.token(for: code) else {
                throw SpeechError.incompatibleResources(
                    table.isEmpty
                        ? "This bundle ships no token table, so a language cannot be selected."
                        : "This bundle has no token for language '\(code)'.")
            }
            return config.retargeted(toLanguageToken: token)

        case .detect:
            // With no token table there is nothing to choose between; fall back to
            // the bundle's pin rather than failing the transcription.
            guard !table.isEmpty, let whisper = decoder as? WhisperDecoder else {
                return config
            }
            let (encOut, encShape, _) = try await runEncoder(pcm: leadingPCM)
            guard let detected = try await whisper.detectLanguageToken(
                encoderOutput: encOut,
                encoderOutputShape: encShape,
                resources: resources,
                candidates: Set(table.codeByID.keys))
            else { return config }
            return config.retargeted(toLanguageToken: detected)
        }
    }

    /// Samples the encoder consumes per pass, or `nil` when it accepts any length.
    private var staticWindowSamples: Int? {
        guard let nFrames = melConfig.nFrames else { return nil }
        return nFrames * melConfig.hopLength
    }

    // MARK: - Parity Testing

    /// Intermediates from one transcription, for stage-by-stage parity checking.
    ///
    /// `transcribe` returns only text, which is a weak signal: a front-end error large
    /// enough to shift every mel bin can still detokenize to the correct sentence. These
    /// are the values a reference implementation can actually be compared against.
    public struct StageCapture: Sendable {
        public let mel: [Float]
        public let melShape: [Int]
        public let encoderHiddenStates: [Float]
        public let encoderShape: [Int]
        /// Emitted tokens, before detokenization — blanks already filtered for TDT.
        public let tokens: [Int32]
        public let text: String
        /// Leading encoder frames treated as real audio (== `encoderShape[1]` when dynamic).
        public let validEncoderFrames: Int
        /// Decode-loop branch counts and first-step tensors for this transcription.
        public let stats: DecodeStats
    }

    /// Transcribe `pcm`, returning each stage's output alongside the text.
    ///
    /// Recomputes the mel separately from `runEncoder`'s internal copy so the encoder
    /// path stays untouched; that costs one extra front-end pass, which is irrelevant
    /// on a verification path.
    /// Single-window only: this captures one encoder pass, so PCM longer than a
    /// static window is truncated as before. Parity checking compares one window.
    public func transcribeCapturingStages(pcm: [Float]) async throws -> StageCapture {
        let (melValues, melShape) = melFeatures(pcm: pcm)
        let (encOut, encShape, validEnc) = try await runEncoder(pcm: pcm)
        // Copy the encoder output out *before* decoding. `decode` issues ~2 graph runs
        // per emitted token, and reading `encOut` afterwards would report whatever those
        // left in the buffer if the runtime recycles output storage.
        let encoderValues = flattenAsFloat(encOut)
        let (tokens, stats) = try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: validEnc,
            resources: resources)
        return StageCapture(
            mel: melValues, melShape: melShape,
            encoderHiddenStates: encoderValues, encoderShape: encShape,
            tokens: tokens, text: try detokenize(tokens), validEncoderFrames: validEnc,
            stats: stats)
    }

    // MARK: - Internals

    private func warmUp() async throws {
        switch bundle.kind {
        case .whisper:
            let nSamples = (melConfig.nFrames ?? 3_000) * melConfig.hopLength
            _ = try await runEncoder(pcm: [Float](repeating: 0, count: nSamples))
        case .parakeetTDT:
            // Static exports have nFrames set; warm up at exactly that size.
            // Dynamic exports skip init warmup — callers use prewarm(sampleCount:)
            // once the actual audio length is known, avoiding a wasted compilation.
            guard let nFrames = melConfig.nFrames else { return }
            _ = try await runEncoder(pcm: [Float](repeating: 0, count: nFrames * melConfig.hopLength))
        }
    }

    /// Run the encoder over PCM and return the encoder hidden states + concrete shape.
    ///
    /// Also returns the exact number of leading encoder frames that carry real audio.
    ///
    /// `fromPCM` pads up to the traced window itself and `padToFrameGrid` derives the
    /// valid-frame count from `pcm.count`, so what a caller passes sets how far the mask and
    /// the per-bin normalization statistics reach. Offline passes real samples only. The
    /// streaming path deliberately zero-fills its ramp-up windows first, so that a frame's
    /// representation does not shift as the window fills — see `runHopIfReady`.
    package func runEncoder(pcm: [Float]) async throws -> (NDArray, [Int], Int) {
        let (encOut, encShape, validEnc, _) = try await runEncoderTimed(pcm: pcm)
        return (encOut, encShape, validEnc)
    }

    /// ``runEncoder(pcm:)``, also returning how long preprocessing and the encoder run took.
    ///
    /// Only `preprocessMs` and `encodeMs` are set on the returned timing.
    private func runEncoderTimed(
        pcm: [Float]
    ) async throws -> (NDArray, [Int], Int, DecodeStats.Timing) {
        let start = ContinuousClock.now
        let mel = MelSpectrogram.fromPCM(pcm, config: melConfig, basis: melBasis)
        let nFrames = MelSpectrogram.frameCount(forPCMLength: pcm.count, config: melConfig)
        let inputShape = encoderInputShape(nFrames: nFrames)
        var melArray = NDArray(descriptor: melInputDescriptor.resolvingDynamicDimensions(inputShape))
        fillFloatNDArray(&melArray, with: mel)

        let validMelFrames = min(
            MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: melConfig), nFrames)

        // Attention mask (B, T_audio): true for real-audio frames, false for the
        // static window's zero-padding tail (and the dynamic path's trailing zero
        // frame). Lets the encoder exclude padding from self-attention and the conv
        // modules, matching HF. Guarded so bundles exported without the input still run.
        var inputs: [String: NDArray] = ["input_features": melArray]
        if let maskDesc = maskInputDescriptor {
            var maskArray = NDArray(descriptor: maskDesc.resolvingDynamicDimensions([1, nFrames]))
            fillNDArray(&maskArray, as: Bool.self, count: nFrames) { $0 < validMelFrames }
            inputs["attention_mask"] = maskArray
        }
        let preprocessDuration = ContinuousClock.now - start
        CLILogger.log("The preprocessing took \(preprocessDuration.inMilliseconds) ms", level: 1)
        let startEncode = ContinuousClock.now
        var outputs = try await encoderFunction.run(inputs: inputs)
        let encodeDuration = ContinuousClock.now - startEncode
        CLILogger.log("The encoding took \(encodeDuration.inMilliseconds) ms", level: 1)
        guard let encOut = outputs.remove("encoder_hidden_states")?.ndArray else {
            throw SpeechError.missingModel("Encoder did not produce 'encoder_hidden_states'")
        }
        // How many leading frames real audio backs. Both the offline decode cap and the
        // streaming path's frame slicing come from this one value.
        let validEnc = Self.validEncoderFrames(
            pcmCount: pcm.count, tEnc: encOut.shape[1], config: melConfig,
            subsamplingFactor: tdtConfig?.encoderSubsamplingFactor ?? 1)
        var timing = DecodeStats.Timing()
        timing.preprocessMs = preprocessDuration.inMilliseconds
        timing.encodeMs = encodeDuration.inMilliseconds
        return (encOut, encOut.shape, validEnc, timing)
    }

    private func encoderInputShape(nFrames: Int) -> [Int] {
        Self.encoderInputShape(nFrames: nFrames, config: melConfig)
    }

    /// The encoder's `input_features` shape for `nFrames` mel frames, in the config's layout.
    ///
    /// `static` and `MelConfig`-parameterized so it can be unit-tested: the actor's only
    /// initializer loads a bundle and warms the graphs, so an instance method here would be
    /// unreachable without model assets on disk.
    package static func encoderInputShape(nFrames: Int, config: MelConfig) -> [Int] {
        switch config.layout {
        case .channelMajor: return [1, config.nMelBins, nFrames]
        case .timeMajor: return [1, nFrames, config.nMelBins]
        }
    }

    private func tokenReporter(
        completedTexts: [String], windowIndex: Int, windowCount: Int,
        startSample: Int, endSample: Int, previousEndSample: Int, totalSamples: Int,
        emit: (@Sendable (SpeechTranscriptionUpdate) -> Void)?
    ) -> SpeechTokenReporter? {
        guard let emit, decoder is ParakeetTDTDecoder || decoder is WhisperDecoder else { return nil }
        // Whisper reports no frame progress mid-window, having no alignment between tokens and
        // audio, so its decoder updates hold at the previous window's end until this one completes.
        return SpeechTokenReporter(
            completedTexts: completedTexts, windowIndex: windowIndex, windowCount: windowCount,
            audioProgress: SpeechDecodeProgress(
                startSample: startSample, endSample: endSample, previousEndSample: previousEndSample,
                totalSamples: totalSamples,
                samplesPerFrame: melConfig.hopLength * (tdtConfig?.encoderSubsamplingFactor ?? 1)),
            detokenize: { tokens in try await self.detokenize(tokens) }, emit: emit)
    }

    /// Decodes exactly one encoder window. Long-form splitting happens in
    /// ``transcribe(pcm:overlapSeconds:language:resetAfterSilenceFrames:)``, which needs detokenized text
    /// to match the seams and so cannot stitch at this level.
    private func decodeAudio(
        pcm: [Float], resources: DecoderResources? = nil, resetAfterSilenceFrames: Int = 0,
        reporter: SpeechTokenReporter? = nil
    ) async throws -> ([Int32], DecodeStats) {
        let (encOut, encShape, validEnc, encoderTiming) = try await runEncoderTimed(pcm: pcm)
        var (tokens, stats) = try await decodeEncoded(
            encOut: encOut, encShape: encShape, validEnc: validEnc, resources: resources,
            resetAfterSilenceFrames: resetAfterSilenceFrames, reporter: reporter)
        stats.timing.preprocessMs = encoderTiming.preprocessMs
        stats.timing.encodeMs = encoderTiming.encodeMs
        return (tokens, stats)
    }

    /// The decoder half of ``decodeAudio(pcm:resources:resetAfterSilenceFrames:reporter:)``.
    private func decodeEncoded(
        encOut: NDArray, encShape: [Int], validEnc: Int, resources: DecoderResources?,
        resetAfterSilenceFrames: Int, reporter: SpeechTokenReporter?
    ) async throws -> ([Int32], DecodeStats) {
        let effectiveResources = resources ?? self.resources
        // Not on `SpeechDecoder`, so reached by downcast the way `startStream` does. Thrown
        // rather than ignored elsewhere: silently dropping a flag the caller passed is how a
        // missed effect reads as a clean run.
        if resetAfterSilenceFrames > 0 || (reporter != nil && decoder is ParakeetTDTDecoder) {
            guard let parakeet = decoder as? ParakeetTDTDecoder else {
                throw SpeechError.incompatibleResources(
                    "resetAfterSilenceFrames applies to a transducer's predictor state; "
                        + "\(architecture) has none")
            }
            let onToken: (@Sendable (Int32) async throws -> Void)?
            let onProgress: (@Sendable (Int) async throws -> Void)?
            if let reporter {
                onToken = { token in try await reporter.append(token) }
                onProgress = { frames in try await reporter.advance(frames: frames) }
            } else {
                onToken = nil
                onProgress = nil
            }
            return try await parakeet.decode(
                encoderOutput: encOut,
                encoderOutputShape: encShape,
                validEncoderFrames: validEnc,
                resources: effectiveResources,
                resetAfterSilenceFrames: resetAfterSilenceFrames,
                onToken: onToken, onProgress: onProgress)
        }
        if let reporter, let whisper = decoder as? WhisperDecoder {
            return try await whisper.decode(
                encoderOutput: encOut,
                encoderOutputShape: encShape,
                validEncoderFrames: validEnc,
                resources: effectiveResources,
                onToken: { token in try await reporter.append(token) })
        }
        return try await decoder.decode(
            encoderOutput: encOut,
            encoderOutputShape: encShape,
            validEncoderFrames: validEnc,
            resources: effectiveResources)
    }

    /// How many leading encoder frames carry real audio.
    ///
    /// Exact, not proportional. The FastConformer front end is a stack of stride-2 convs,
    /// each mapping `T` to `(T - 1) / 2 + 1`, so applying that per stage to the *valid* mel
    /// count yields the frame count backed by real audio — no ratio, no rounding. A
    /// proportional estimate lands one frame low for ~25% of audio lengths against the
    /// shipped geometry, which clips a trailing token.
    ///
    /// Sub-hop audio yields 0 here; `ParakeetTDTDecoder.decode` floors its own cap at one
    /// frame, so a tiny clip still decodes something.
    ///
    /// `static` and parameterized so it can be unit-tested — the actor cannot be constructed
    /// without model assets. `runEncoder` is the only caller, so offline and streaming share
    /// one definition rather than each deriving their own.
    package static func validEncoderFrames(
        pcmCount: Int, tEnc: Int, config: MelConfig, subsamplingFactor: Int
    ) -> Int {
        let frameCount = MelSpectrogram.frameCount(forPCMLength: pcmCount, config: config)
        let validMel = min(
            MelSpectrogram.validFrameCount(forPCMLength: pcmCount, config: config), frameCount)
        return min(
            tEnc, encoderFrameCount(melFrames: validMel, subsamplingFactor: subsamplingFactor))
    }

    package func detokenize(_ tokens: [Int32]) throws -> String {
        guard let tokenizer = bundle.tokenizer else { throw SpeechError.missingTokenizer }
        let ids: [Int]
        switch bundle.kind {
        case .whisper(let assets):
            ids = tokens.filter { $0 < assets.generationConfig.eotToken }.map { Int($0) }
        case .parakeetTDT:
            // Decoder already filters blanks; pass everything through.
            ids = tokens.map { Int($0) }
        }
        return tokenizer.decode(tokens: ids).trimmingCharacters(in: .whitespaces)
    }
}

import Foundation

/// A replacement snapshot of the whole transcript so far, never an append-only delta.
public struct SpeechTranscriptionUpdate: Sendable, Equatable {
    public enum Source: Sendable { case decoder, window }
    public let text: String
    public let source: Source
    public let windowIndex: Int
    public let windowCount: Int
    /// Fraction of source audio decoded, independent of wall-clock time.
    public var progress: Double? = nil
}

/// Maps decoded frames to unique source-audio coverage. Overlap cannot regress
/// progress; padding and duration jumps cannot advance beyond the window's end.
package struct SpeechDecodeProgress: Sendable {
    package let startSample: Int
    package let endSample: Int
    package let previousEndSample: Int
    package let totalSamples: Int
    package let samplesPerFrame: Int

    package func fraction(completedFrames: Int) -> Double {
        guard totalSamples > 0 else { return 0 }
        let decoded = min(endSample, startSample + max(0, completedFrames) * samplesPerFrame)
        return min(1, max(0, Double(max(previousEndSample, decoded)) / Double(totalSamples)))
    }
}

/// Keeps token delivery cheap: collect every token, but detokenize/display at most
/// twenty times a second. Each window has its own reporter and token history.
package actor SpeechTokenReporter {
    private var tokens: [Int32] = []
    private var lastEmission: ContinuousClock.Instant?
    private let audioProgress: SpeechDecodeProgress?
    private var completedFrames = 0
    private var text = ""
    private var decodedTokenCount = 0
    private let interval: Duration
    private let completedTexts: [String]
    private let windowIndex: Int
    private let windowCount: Int
    private let detokenize: @Sendable ([Int32]) async throws -> String
    private let emit: @Sendable (SpeechTranscriptionUpdate) -> Void

    package init(
        completedTexts: [String], windowIndex: Int, windowCount: Int,
        interval: Duration = .milliseconds(50),
        audioProgress: SpeechDecodeProgress? = nil,
        detokenize: @escaping @Sendable ([Int32]) async throws -> String,
        emit: @escaping @Sendable (SpeechTranscriptionUpdate) -> Void
    ) {
        self.completedTexts = completedTexts
        self.windowIndex = windowIndex
        self.windowCount = windowCount
        self.audioProgress = audioProgress
        self.interval = interval
        self.detokenize = detokenize
        self.emit = emit
    }

    package func append(_ token: Int32) async throws {
        try Task.checkCancellation()
        tokens.append(token)
        try await report()
    }

    package func advance(frames: Int) async throws {
        completedFrames = max(completedFrames, frames)
        try await report()
    }

    private func report() async throws {
        try Task.checkCancellation()
        let now = ContinuousClock.now
        if let lastEmission, lastEmission.duration(to: now) < interval { return }
        lastEmission = now
        if decodedTokenCount != tokens.count {
            text = try await detokenize(tokens)
            decodedTokenCount = tokens.count
        }
        emit(SpeechTranscriptionUpdate(
            text: LongForm.stitch(completedTexts + [text]), source: .decoder,
            windowIndex: windowIndex, windowCount: windowCount,
            progress: audioProgress?.fraction(completedFrames: completedFrames)))
    }
}

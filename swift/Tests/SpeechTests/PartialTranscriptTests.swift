import Foundation
import Testing
import os
@testable import CoreAISpeech

@Suite struct PartialTranscriptTests {
    @Test func tokensAreDetokenizedTogetherAndWindowsAreStitched() async throws {
        let updates = OSAllocatedUnfairLock<[SpeechTranscriptionUpdate]>(initialState: [])
        let reporter = SpeechTokenReporter(
            completedTexts: ["We saw the red"], windowIndex: 1, windowCount: 2, interval: .zero,
            detokenize: { tokens in
                tokens.map { [1: "the", 2: " red", 3: " car"][Int($0)]! }.joined()
            }, emit: { update in updates.withLock { $0.append(update) } })
        try await reporter.append(1)
        try await reporter.append(2)
        try await reporter.append(3)
        let emitted = updates.withLock { $0 }
        #expect(emitted.count == 3)
        #expect(emitted.last?.text == "We saw the red car")
        #expect(emitted.allSatisfy { $0.source == .decoder && $0.windowIndex == 1 && $0.windowCount == 2 })
    }

    @Test func throttlingStillEmitsTheFirstToken() async throws {
        let updates = OSAllocatedUnfairLock<[String]>(initialState: [])
        let reporter = SpeechTokenReporter(
            completedTexts: [], windowIndex: 0, windowCount: 1, interval: .seconds(3600),
            detokenize: { tokens in tokens.map(String.init).joined(separator: " ") },
            emit: { update in updates.withLock { $0.append(update.text) } })
        for token: Int32 in 1...10 { try await reporter.append(token) }
        #expect(updates.withLock { $0 } == ["1"])
    }
}

@Suite struct DecodeProgressTests {
    @Test func overlapPaddingAndDurationJumpsDoNotOvercountAudio() {
        let window = SpeechDecodeProgress(
            startSample: 28_000, endSample: 58_000, previousEndSample: 30_000,
            totalSamples: 65_000, samplesPerFrame: 80)
        #expect(window.fraction(completedFrames: 0) == 30.0 / 65)
        #expect(window.fraction(completedFrames: 10) == 30.0 / 65)
        #expect(window.fraction(completedFrames: 100) == 36.0 / 65)
        #expect(window.fraction(completedFrames: 1_000) == 58.0 / 65)
        let last = SpeechDecodeProgress(
            startSample: 56_000, endSample: 65_000, previousEndSample: 58_000,
            totalSamples: 65_000, samplesPerFrame: 80)
        #expect(last.fraction(completedFrames: 0) == window.fraction(completedFrames: 1_000))
        #expect(last.fraction(completedFrames: 200) == 1)
    }

    @Test func silenceAdvancesWithoutProducingTokens() async throws {
        let updates = OSAllocatedUnfairLock<[SpeechTranscriptionUpdate]>(initialState: [])
        let reporter = SpeechTokenReporter(
            completedTexts: [], windowIndex: 0, windowCount: 1, interval: .zero,
            audioProgress: SpeechDecodeProgress(
                startSample: 0, endSample: 1_000, previousEndSample: 0,
                totalSamples: 1_000, samplesPerFrame: 80),
            detokenize: { _ in Issue.record("Silence should not need detokenization"); return "" },
            emit: { update in updates.withLock { $0.append(update) } })
        try await reporter.advance(frames: 4)
        try await reporter.advance(frames: 8)
        #expect(updates.withLock { $0.map(\.progress) } == [0.32, 0.64])
        #expect(updates.withLock { $0.allSatisfy { $0.text.isEmpty } })
    }
}

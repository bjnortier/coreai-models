// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

/// Whisper names a *task* in the prefix slot after the language: transcribe the
/// audio as spoken, or translate it into English. Both come out of the same
/// decoder in one pass, and both read as fluent text — so the only evidence the
/// requested task actually ran is the prefix itself.
@Suite("Speech task selection")
struct SpeechTaskTests {
    /// Writes a token table shaped like a bundle's `added_tokens.json`.
    private func makeBundle(_ tokens: [String: Int]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "speech-task-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: tokens)
        try data.write(to: dir.appending(path: "added_tokens.json"))
        return dir
    }

    // MARK: GenerationConfig

    @Test("A config reports the task its prefix pins")
    func reportsPinnedTask() {
        #expect(GenerationConfig.whisper.taskToken == GenerationConfig.whisperTranscribeToken)
    }

    @Test("Retargeting the task leaves the language slot alone")
    func retargetKeepsLanguage() {
        let original = GenerationConfig.whisper
        let translating = original.retargeted(
            toTaskToken: GenerationConfig.whisperTranslateToken)

        #expect(translating.taskToken == GenerationConfig.whisperTranslateToken)
        // Whisper translates *from* the language in slot 1 into English, so that
        // slot must keep naming the spoken language rather than becoming <|en|>.
        #expect(translating.languageToken == original.languageToken)
        #expect(translating.forcedPrefix.first == original.forcedPrefix.first)
        #expect(translating.forcedPrefix.last == original.forcedPrefix.last)
        #expect(translating.forcedPrefix.count == original.forcedPrefix.count)
        #expect(translating.eotToken == original.eotToken)
    }

    @Test("Language and task retargeting compose in either order")
    func retargetingComposes() {
        let base = GenerationConfig.whisper
        let translate = GenerationConfig.whisperTranslateToken
        let french: Int32 = 50265

        let languageFirst = base.retargeted(toLanguageToken: french)
            .retargeted(toTaskToken: translate)
        let taskFirst = base.retargeted(toTaskToken: translate)
            .retargeted(toLanguageToken: french)

        #expect(languageFirst.forcedPrefix == taskFirst.forcedPrefix)
        #expect(languageFirst.forcedPrefix == [50258, french, translate, 50364])
    }

    @Test("A prefix with no task slot is returned unchanged")
    func retargetIgnoresShortPrefix() {
        let bare = GenerationConfig(
            forcedPrefix: [50258, 50259], eotToken: 50257, maxDecodeSteps: 50, tokenizerName: nil)
        #expect(bare.taskToken == nil)
        #expect(bare.retargeted(toTaskToken: 50359).forcedPrefix == [50258, 50259])
    }

    // MARK: Token table

    @Test("Task tokens are read from the bundle")
    func parsesTaskTokens() throws {
        let dir = try makeBundle([
            "<|startoftranscript|>": 50258,
            "<|en|>": 50259,
            "<|translate|>": 50359,
            "<|transcribe|>": 50360,
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = WhisperTaskTokens(bundleURL: dir)
        #expect(tokens.token(for: .transcribe) == 50360)
        #expect(tokens.token(for: .translateToEnglish) == 50359)
    }

    /// The numbering is generation-specific — large-v3 inserted `<|yue|>` ahead of
    /// the task tokens — which is why they are read rather than hardcoded.
    @Test("A bundle numbered differently is read, not assumed")
    func parsesShiftedNumbering() throws {
        let dir = try makeBundle(["<|translate|>": 50358, "<|transcribe|>": 50359])
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = WhisperTaskTokens(bundleURL: dir)
        #expect(tokens.token(for: .translateToEnglish) == 50358)
        #expect(tokens.token(for: .transcribe) == 50359)
    }

    @Test("A bundle with no token table reports no task tokens rather than guessing")
    func missingTableIsEmpty() {
        let tokens = WhisperTaskTokens(bundleURL: URL(filePath: "/nonexistent/speech-bundle"))
        #expect(tokens.token(for: .transcribe) == nil)
        #expect(tokens.token(for: .translateToEnglish) == nil)
    }
}

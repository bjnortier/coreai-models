// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

/// A Whisper export names its decode language in its prefix. Retargeting that
/// prefix is what lets one bundle transcribe more than the language it shipped
/// pinned to — without it, other languages come back as fluent translations into
/// the pinned language rather than as errors.
@Suite("Speech language selection")
struct SpeechLanguageTests {
    // MARK: GenerationConfig

    @Test("A config reports the language its prefix pins")
    func reportsPinnedLanguage() {
        #expect(GenerationConfig.whisper.languageToken == 50259)  // <|en|>
    }

    @Test("Retargeting replaces only the language slot")
    func retargetKeepsTaskTokens() {
        let original = GenerationConfig.whisper
        let french = original.retargeted(toLanguageToken: 50265)  // <|fr|>

        #expect(french.languageToken == 50265)
        // Start, task and timestamp tokens are the export's, and must survive.
        #expect(french.forcedPrefix.first == original.forcedPrefix.first)
        #expect(Array(french.forcedPrefix.dropFirst(2)) == Array(original.forcedPrefix.dropFirst(2)))
        #expect(french.forcedPrefix.count == original.forcedPrefix.count)
        #expect(french.eotToken == original.eotToken)
        #expect(french.maxDecodeSteps == original.maxDecodeSteps)
    }

    @Test("A prefix with no language slot is returned unchanged")
    func retargetIgnoresShortPrefix() {
        let bare = GenerationConfig(
            forcedPrefix: [50258], eotToken: 50257, maxDecodeSteps: 50, tokenizerName: nil)
        #expect(bare.languageToken == nil)
        #expect(bare.retargeted(toLanguageToken: 50265).forcedPrefix == [50258])
    }

    // MARK: Token table

    /// Writes a token table shaped like a bundle's `added_tokens.json`.
    private func makeBundle(_ tokens: [String: Int]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "speech-lang-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: tokens)
        try data.write(to: dir.appending(path: "added_tokens.json"))
        return dir
    }

    @Test("Language tags are read from the bundle, control tokens are not")
    func parsesLanguageTags() throws {
        let dir = try makeBundle([
            "<|startoftranscript|>": 50258,
            "<|en|>": 50259,
            "<|fr|>": 50265,
            "<|yue|>": 50358,
            "<|translate|>": 50359,
            "<|transcribe|>": 50360,
            "<|notimestamps|>": 50364,
        ])
        defer { try? FileManager.default.removeItem(at: dir) }

        let table = WhisperLanguageTokens(bundleURL: dir)
        #expect(table.idByCode == ["en": 50259, "fr": 50265, "yue": 50358])
        #expect(table.codeByID[50265] == "fr")
        // Control tokens are longer words and must not be mistaken for languages.
        #expect(table.idByCode["transcribe"] == nil)
        #expect(table.idByCode["notimestamps"] == nil)
        #expect(table.idByCode["startoftranscript"] == nil)
    }

    @Test("Locale-shaped identifiers resolve to their base language")
    func acceptsLocaleIdentifiers() throws {
        let dir = try makeBundle(["<|fr|>": 50265, "<|en|>": 50259])
        defer { try? FileManager.default.removeItem(at: dir) }

        let table = WhisperLanguageTokens(bundleURL: dir)
        #expect(table.token(for: "fr") == 50265)
        #expect(table.token(for: "fr-FR") == 50265)
        #expect(table.token(for: "fr_FR") == 50265)
        #expect(table.token(for: "FR") == 50265)
        #expect(table.token(for: "de") == nil)
    }

    @Test("A bundle with no token table reports empty rather than failing")
    func missingTableIsEmpty() {
        let table = WhisperLanguageTokens(
            bundleURL: URL(filePath: "/nonexistent/speech-bundle"))
        #expect(table.isEmpty)
        #expect(table.token(for: "fr") == nil)
    }
}

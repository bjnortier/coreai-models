import Foundation

/// Which language a Whisper decode should transcribe.
///
/// Whisper is told the language through its decoder prefix rather than inferring
/// it, so an export that pins `<|en|>` does not fail on French audio — it
/// transcribes it *as if* it were English, emitting a fluent English translation.
/// That reads as a plausible transcript and scores like a broken model, so the
/// language has to be an explicit part of the request.
///
/// Parakeet TDT decodes multilingual audio directly and ignores this.
public enum SpeechLanguage: Sendable, Equatable {
    /// Detect the language from the audio.
    ///
    /// Whisper predicts the language token in the first decoder step, so this
    /// costs one extra step per transcription.
    case detect

    /// Transcribe as the given language, e.g. `"fr"`. Case-insensitive.
    case code(String)

    /// Use whatever language the bundle's `generation_config.json` pins.
    case bundleDefault
}

/// What a Whisper decode should *do* with the audio.
///
/// Whisper's decoder prefix names a task in the slot after the language:
/// `<|startoftranscript|><|fr|><|transcribe|>` writes French,
/// `<|startoftranscript|><|fr|><|translate|>` writes English. The language slot
/// keeps naming the *spoken* language either way — Whisper's translation task is
/// X-to-English only, and there is no token for any other target.
///
/// Parakeet TDT has no task slot, and rejects ``translateToEnglish`` rather than
/// transcribing and calling it a translation.
public enum SpeechTask: Sendable, Equatable {
    /// Write the audio down in the language it was spoken in. The default.
    case transcribe

    /// Translate the audio into English as it is decoded.
    ///
    /// One pass, not transcribe-then-translate: the English comes out of the same
    /// decoder, so it costs no more than a transcription of the same audio.
    case translateToEnglish
}

/// The `<|xx|>` language tokens a Whisper bundle understands.
///
/// Read from the bundle rather than hardcoded: the set and ordering of language
/// tokens differ between Whisper generations (large-v3 added Cantonese), and the
/// bundle already ships the mapping.
struct WhisperLanguageTokens: Sendable {
    let idByCode: [String: Int32]
    let codeByID: [Int32: String]

    var isEmpty: Bool { idByCode.isEmpty }

    /// Parses the `<|xx|>` entries out of a bundle's token table.
    ///
    /// Language tags are two- or three-letter codes; the control tokens
    /// (`<|transcribe|>`, `<|notimestamps|>`, …) are longer words, which is what
    /// separates them without needing a list of either.
    init(bundleURL: URL) {
        var idByCode: [String: Int32] = [:]
        var codeByID: [Int32: String] = [:]

        for (token, id) in whisperSpecialTokens(bundleURL: bundleURL) {
            let code = String(token.dropFirst(2).dropLast(2))
            guard code.count >= 2, code.count <= 3, code.allSatisfy(\.isLetter) else { continue }
            idByCode[code.lowercased()] = id
            codeByID[id] = code.lowercased()
        }
        self.idByCode = idByCode
        self.codeByID = codeByID
    }

    /// The token for `code`, accepting either a bare code or a locale-ish
    /// identifier such as `"fr-FR"` or `"fr_FR"`.
    func token(for code: String) -> Int32? {
        let base = code.lowercased().split(whereSeparator: { $0 == "-" || $0 == "_" }).first.map(String.init)
        return base.flatMap { idByCode[$0] }
    }
}

/// The `<|transcribe|>` / `<|translate|>` task tokens a Whisper bundle understands.
///
/// Read from the bundle for the same reason the language tags are: the numbering
/// moved between Whisper generations (large-v3 inserted Cantonese ahead of them),
/// so a hardcoded id silently decodes as some other control token on a bundle it
/// was not measured against.
struct WhisperTaskTokens: Sendable {
    let transcribe: Int32?
    let translate: Int32?

    init(bundleURL: URL) {
        let table = whisperSpecialTokens(bundleURL: bundleURL)
        self.transcribe = table["<|transcribe|>"]
        self.translate = table["<|translate|>"]
    }

    init(transcribe: Int32?, translate: Int32?) {
        self.transcribe = transcribe
        self.translate = translate
    }

    /// The token that selects `task`, or `nil` when this bundle ships no table.
    func token(for task: SpeechTask) -> Int32? {
        switch task {
        case .transcribe: return transcribe
        case .translateToEnglish: return translate
        }
    }
}

/// A bundle's `<|…|>` special tokens, from `added_tokens.json` (falling back to
/// `vocab.json`).
///
/// Shared by the language and task tables so a bundle is parsed the same way for
/// both — and so the fallback order is stated once.
func whisperSpecialTokens(bundleURL: URL) -> [String: Int32] {
    for name in ["added_tokens.json", "vocab.json"] {
        let url = bundleURL.appending(path: name)
        guard let data = try? Data(contentsOf: url),
            let table = try? JSONSerialization.jsonObject(with: data) as? [String: Int]
        else { continue }

        var specials: [String: Int32] = [:]
        for (token, id) in table where token.hasPrefix("<|") && token.hasSuffix("|>") {
            specials[token] = Int32(id)
        }
        if !specials.isEmpty { return specials }
    }
    return [:]
}

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

        for name in ["added_tokens.json", "vocab.json"] {
            let url = bundleURL.appending(path: name)
            guard let data = try? Data(contentsOf: url),
                  let table = try? JSONSerialization.jsonObject(with: data) as? [String: Int]
            else { continue }

            for (token, id) in table {
                guard token.hasPrefix("<|"), token.hasSuffix("|>") else { continue }
                let code = String(token.dropFirst(2).dropLast(2))
                guard code.count >= 2, code.count <= 3, code.allSatisfy(\.isLetter) else { continue }
                idByCode[code.lowercased()] = Int32(id)
                codeByID[Int32(id)] = code.lowercased()
            }
            if !idByCode.isEmpty { break }
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

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

// MARK: - LongForm

/// Splitting long audio across a fixed-size encoder window, and putting the
/// per-window transcripts back together.
///
/// Whisper's encoder is architecturally fixed at 30 s, so audio longer than that
/// has to be decoded as several windows. Cutting on exact 30 s boundaries lands
/// mid-word often enough to matter, so windows overlap: each one starts slightly
/// before the previous ended, and the words that appear in both are dropped from
/// the second when the transcripts are joined.
///
/// This is a heuristic, and a weaker one than Whisper's own long-form algorithm,
/// which emits timestamp tokens and advances to the end of the last *complete*
/// segment. That would need timestamps in the forced prefix and a decode loop
/// that reads them. The overlap approach needs neither and fixes the common case.
///
/// Pure and `package`-visible so the boundary arithmetic and the seam matching can
/// be unit-tested; the actor that uses them cannot be constructed without a model.
package enum LongForm {
    /// How many words at a seam to consider when looking for the repeat.
    ///
    /// Two seconds of speech is rarely more than ~8 words; the extra headroom
    /// covers fast speech without letting an accidental match reach far back into
    /// unrelated text.
    package static let maxOverlapWords: Int = 16

    /// Sample offsets at which each window starts.
    ///
    /// Windows are `windowSamples` long and advance by `windowSamples - overlapSamples`.
    /// The final window is whatever audio remains, which the overlap guarantees is at
    /// least `overlapSamples` long — so no window is a sliver of near-silence, which
    /// Whisper tends to hallucinate on.
    package static func windowStarts(
        pcmCount: Int, windowSamples: Int, overlapSamples: Int
    ) -> [Int] {
        guard windowSamples > 0, pcmCount > windowSamples else { return [0] }
        let stride = max(1, windowSamples - max(0, overlapSamples))
        var starts: [Int] = []
        var start = 0
        while true {
            starts.append(start)
            if start + windowSamples >= pcmCount { break }
            start += stride
        }
        return starts
    }

    /// Joins per-window transcripts, dropping the words the overlap duplicated.
    ///
    /// For each seam, finds the longest run of words that ends the accumulated text
    /// and begins the next window, and drops that run from the next window. When no
    /// run matches — the two windows transcribed the shared audio differently, which
    /// happens because each decodes with its own context — the text is concatenated
    /// as-is rather than guessing, so the failure mode is a repeated word or two
    /// rather than dropped speech.
    package static func stitch(
        _ texts: [String], maxOverlapWords: Int = LongForm.maxOverlapWords
    ) -> String {
        var out: [String] = []
        for text in texts {
            let words = text.split(separator: " ").map(String.init)
            if words.isEmpty { continue }
            if out.isEmpty {
                out = words
                continue
            }
            let repeated = overlapLength(
                tail: out, head: words, maxOverlapWords: maxOverlapWords)
            out.append(contentsOf: words.dropFirst(repeated))
        }
        return out.joined(separator: " ")
    }

    /// Longest `k` (up to `maxOverlapWords`) where the last `k` words of `tail`
    /// match the first `k` of `head`, compared loosely. Zero when nothing matches.
    package static func overlapLength(
        tail: [String], head: [String], maxOverlapWords: Int
    ) -> Int {
        let limit = min(maxOverlapWords, min(tail.count, head.count))
        guard limit > 0 else { return 0 }
        // Longest first: a 5-word repeat also matches at k=1, and taking the
        // longest is what actually removes the duplicated span.
        for k in stride(from: limit, through: 1, by: -1) {
            if normalized(tail.suffix(k)) == normalized(head.prefix(k)) { return k }
        }
        return 0
    }

    /// Case- and punctuation-insensitive comparison key.
    ///
    /// The same word is routinely capitalized in one window and not the other (it
    /// starts a sentence in one, continues one in the other), and trailing commas
    /// and periods differ for the same reason. Comparing raw words misses most real
    /// repeats. Element count is preserved so the two sides stay aligned.
    private static func normalized(_ words: some Sequence<String>) -> [String] {
        words.map { word in
            String(word.lowercased().unicodeScalars.filter(CharacterSet.alphanumerics.contains))
        }
    }
}

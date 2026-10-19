// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Word timestamps for Whisper, from the cross-attention of its alignment heads.
///
/// Whisper's text tokens carry no time (this runtime decodes with `<|notimestamps|>`), so a
/// token's time is recovered from where the decoder was looking while it predicted it. A
/// `export_kv.py` decoder also returns `alignment_logits`: the pre-softmax cross-attention
/// scores of the heads the model's generation config names as aligned with speech. This warps
/// the generated steps onto the encoder frames the way openai-whisper's `find_alignment` does,
/// and HF `return_token_timestamps` after it, then groups tokens into words with
/// openai-whisper's punctuation rules. `verify_kv.py` runs the same pipeline in Python and
/// checks it against HuggingFace.
package enum WhisperAlignment {
    /// Encoder frames per second: 3000 mel frames per 30 s, halved by the conv stride.
    package static let framesPerSecond = 50.0
    static let medianFilterWidth = 7

    // openai-whisper's defaults for `prepend_punctuations` / `append_punctuations`.
    static let prependPunctuation = "\"'“¿([{-"
    static let appendPunctuation = "\"'.。,，!！?？:：”)]}、"
    static let sentenceEnd = ".。!！?？"
    /// Python's `string.punctuation`.
    static let asciiPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    /// Languages written without spaces between words, where openai-whisper makes every
    /// character a word.
    package static let unspacedLanguages: Set<String> = ["zh", "ja", "th", "lo", "my", "yue"]

    /// The alignment heads' scores for each generated step of one window.
    package struct Scores: Sendable {
        /// `[steps, heads, frames]`, row-major.
        package var values: [Float] = []
        package let heads: Int
        package let frames: Int

        package init(heads: Int, frames: Int, values: [Float] = []) {
            self.heads = heads
            self.frames = frames
            self.values = values
        }

        package var steps: Int { values.count / max(heads * frames, 1) }
    }

    // MARK: - Tokens to frames

    /// The encoder frame each step's token starts at, one per step.
    ///
    /// Softmax over the first `validFrames` frames (the ones real audio backs), normalize each
    /// head's frame across the steps, median-filter along time, average the heads, and warp
    /// the steps onto the frames. A step's token starts where the path first reaches it.
    package static func tokenFrames(_ scores: Scores, validFrames: Int) -> [Int] {
        let steps = scores.steps
        let heads = scores.heads
        let frames = min(validFrames, scores.frames)
        guard steps > 1, frames > 0 else { return [Int](repeating: 0, count: steps) }

        // weights[(head * steps + step) * frames + frame]: each step's softmax over the valid frames
        var weights = [Double](repeating: 0, count: heads * steps * frames)
        for step in 0..<steps {
            for head in 0..<heads {
                let source = (step * heads + head) * scores.frames
                let target = (head * steps + step) * frames
                var top = -Double.infinity
                for frame in 0..<frames { top = max(top, Double(scores.values[source + frame])) }
                var sum = 0.0
                for frame in 0..<frames {
                    let e = exp(Double(scores.values[source + frame]) - top)
                    weights[target + frame] = e
                    sum += e
                }
                for frame in 0..<frames { weights[target + frame] /= sum }
            }
        }

        // matrix[step][frame]: the heads' mean after normalizing and filtering
        var matrix = [[Double]](repeating: [Double](repeating: 0, count: frames), count: steps)
        var row = [Double](repeating: 0, count: frames)
        var filtered = [Double](repeating: 0, count: frames)
        for head in 0..<heads {
            let base = head * steps * frames
            for frame in 0..<frames {
                var mean = 0.0
                for step in 0..<steps { mean += weights[base + step * frames + frame] }
                mean /= Double(steps)
                var variance = 0.0
                for step in 0..<steps {
                    let d = weights[base + step * frames + frame] - mean
                    variance += d * d
                }
                let std = (variance / Double(steps)).squareRoot()
                for step in 0..<steps {
                    let i = base + step * frames + frame
                    weights[i] = (weights[i] - mean) / (std > 0 ? std : 1)
                }
            }
            for step in 0..<steps {
                for frame in 0..<frames { row[frame] = weights[base + step * frames + frame] }
                medianFilter(row, width: medianFilterWidth, into: &filtered)
                for frame in 0..<frames { matrix[step][frame] += filtered[frame] / Double(heads) }
            }
        }

        let (textIndices, timeIndices) = dynamicTimeWarping(cost: matrix.map { $0.map { -$0 } })
        var starts: [Int] = []
        starts.reserveCapacity(steps)
        for (k, text) in textIndices.enumerated() where k == 0 || text != textIndices[k - 1] {
            starts.append(timeIndices[k])
        }
        return starts
    }

    /// A running median of odd `width` with reflected edges, as HF's `_median_filter`. Rows no
    /// longer than half the width come back unchanged.
    static func medianFilter(_ row: [Double], width: Int) -> [Double] {
        var out = [Double](repeating: 0, count: row.count)
        medianFilter(row, width: width, into: &out)
        return out
    }

    /// ``medianFilter(_:width:)`` into a preallocated `out`, which this runs a few hundred
    /// thousand times per window.
    static func medianFilter(_ row: [Double], width: Int, into out: inout [Double]) {
        let pad = width / 2
        let n = row.count
        guard n > pad else {
            for i in 0..<n { out[i] = row[i] }
            return
        }
        var window = [Double](repeating: 0, count: width)
        for i in 0..<n {
            // torch's reflect padding: index -k reads k, index n-1+k reads n-1-k
            for k in 0..<width {
                let j = i - pad + k
                window[k] = row[j < 0 ? -j : j >= n ? 2 * (n - 1) - j : j]
            }
            // insertion sort: seven elements
            for a in 1..<width {
                let value = window[a]
                var b = a - 1
                while b >= 0 && window[b] > value {
                    window[b + 1] = window[b]
                    b -= 1
                }
                window[b + 1] = value
            }
            out[i] = window[pad]
        }
    }

    /// The cheapest monotonic path through `cost`, from its top-left to its bottom-right, as
    /// the row and column of each cell on it. HF's `_dynamic_time_warping`, down to its float32
    /// accumulator and its tie-breaking, so that both pick the same path.
    static func dynamicTimeWarping(cost: [[Double]]) -> (rows: [Int], columns: [Int]) {
        let n = cost.count
        let m = cost.first?.count ?? 0
        let width = m + 1
        var total = [Float](repeating: .infinity, count: (n + 1) * width)
        // 0 diagonal, 1 from the row above, 2 from the column to the left
        var trace = [Int8](repeating: -1, count: (n + 1) * width)
        total[0] = 0
        for j in 1..<width {
            for i in 1..<(n + 1) {
                let c0 = total[(i - 1) * width + j - 1]
                let c1 = total[(i - 1) * width + j]
                let c2 = total[i * width + j - 1]
                let (c, t): (Float, Int8) =
                    c0 < c1 && c0 < c2 ? (c0, 0) : c1 < c0 && c1 < c2 ? (c1, 1) : (c2, 2)
                total[i * width + j] = Float(cost[i - 1][j - 1] + Double(c))
                trace[i * width + j] = t
            }
        }
        for j in 0...m { trace[j] = 2 }
        for i in 0...n { trace[i * width] = 1 }

        var rows: [Int] = []
        var columns: [Int] = []
        var (i, j) = (n, m)
        while i > 0 || j > 0 {
            rows.append(i - 1)
            columns.append(j - 1)
            switch trace[i * width + j] {
            case 0: (i, j) = (i - 1, j - 1)
            case 1: i -= 1
            default: j -= 1
            }
        }
        return (rows.reversed(), columns.reversed())
    }

    // MARK: - Frames to words

    /// Groups one window's generated tokens into timed words, with times in seconds from the
    /// window's start.
    ///
    /// openai-whisper's `split_to_word_tokens`, its truncation of long words at sentence ends,
    /// `merge_punctuations` and its clamp on a long first word, in that order. A word ends
    /// where the next one starts, so punctuation merged onto a word takes no time of its own.
    ///
    /// - Parameters:
    ///   - tokens: the generated tokens, one per step, ending with `eot` unless the step limit
    ///     cut the window short.
    ///   - frames: ``tokenFrames(_:validFrames:)`` for the same steps.
    ///   - endFrame: where the last word ends when no `eot` step follows it.
    ///   - splitsOnSpaces: `false` for ``unspacedLanguages``, where each character is a word.
    ///   - decode: the tokenizer's decode, leading space kept.
    package static func words(
        tokens: [Int32], frames: [Int], eot: Int32, endFrame: Int, splitsOnSpaces: Bool = true,
        decode: ([Int]) -> String
    ) -> [TimedWord] {
        // text, first step, the step after the word's last token
        var groups: [(text: String, first: Int, after: Int)] = []
        var pending: [Int] = []
        var boundary = true
        for (index, token) in tokens.enumerated() {
            // eot, or a stray special token: never part of a word, always a boundary
            guard token < eot else {
                pending = []
                boundary = true
                continue
            }
            pending.append(Int(token))
            let piece = decode(pending)
            // an incomplete UTF-8 sequence: wait for the token carrying its next byte
            if piece.contains("\u{FFFD}") { continue }
            let first = index - pending.count + 1
            pending = []
            let stripped = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            // Python's `str in str` is a substring test, which openai-whisper relies on here
            let opens =
                boundary || !splitsOnSpaces || piece.hasPrefix(" ") || stripped.isEmpty
                || asciiPunctuation.contains(stripped)
            boundary = false
            if opens || groups.isEmpty {
                groups.append((piece, first, index + 1))
            } else {
                groups[groups.count - 1].text += piece
                groups[groups.count - 1].after = index + 1
            }
        }
        func seconds(_ step: Int) -> Double {
            Double(step < frames.count ? frames[step] : endFrame) / framesPerSecond
        }
        var words = groups.map { TimedWord(text: $0.text, start: seconds($0.first), end: seconds($0.after)) }

        let durations = words.map { $0.end - $0.start }.filter { $0 > 0 }.sorted()
        let median = durations.isEmpty ? 0 : min(0.7, Self.median(durations))
        let longest = median * 2
        if !durations.isEmpty {
            for i in words.indices.dropFirst() where words[i].end - words[i].start > longest {
                if sentenceEnd.contains(words[i].text) {
                    words[i].end = words[i].start + longest
                } else if sentenceEnd.contains(words[i - 1].text) {
                    words[i].start = words[i].end - longest
                }
            }
        }

        // prepended punctuation joins the word after it, scanning back from the end
        var (i, j) = (words.count - 2, words.count - 1)
        while i >= 0 {
            let stripped = words[i].text.trimmingCharacters(in: .whitespacesAndNewlines)
            if words[i].text.hasPrefix(" ") && (stripped.isEmpty || prependPunctuation.contains(stripped)) {
                words[j].text = words[i].text + words[j].text
                words[i].text = ""
            } else {
                j = i
            }
            i -= 1
        }
        // appended punctuation joins the word before it
        (i, j) = (0, 1)
        while j < words.count {
            let text = words[j].text
            if !words[i].text.hasSuffix(" ") && (text.isEmpty || appendPunctuation.contains(text)) {
                words[i].text += text
                words[j].text = ""
            } else {
                i = j
            }
            j += 1
        }
        words.removeAll { $0.text.isEmpty }

        // a first word stretched over the silence before it
        if let first = words.first, first.end > median * 4,
            first.end - first.start > longest
                || (words.count > 1 && words[1].end - first.start > longest * 2)
        {
            if words.count > 1 && words[1].end - words[1].start > longest {
                let boundary = max(words[1].end / 2, words[1].end - longest)
                words[0].end = boundary
                words[1].start = boundary
            }
            words[0].start = max(0, words[0].end - longest)
        }
        return words.map {
            TimedWord(text: $0.text.trimmingCharacters(in: .whitespacesAndNewlines), start: $0.start, end: $0.end)
        }
    }

    /// numpy's median of an ascending, non-empty array.
    private static func median(_ sorted: [Double]) -> Double {
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    }
}

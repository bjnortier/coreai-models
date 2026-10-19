// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAISpeech

/// Whisper word alignment against `verify_kv.py`'s Python reference, which uses HF's own
/// `_dynamic_time_warping` and `_median_filter`. Every expected value here was printed by that
/// reference for the same input; `verify_kv.py` in turn checks the pipeline end to end against
/// HuggingFace's cross-attention.
@Suite("Whisper word alignment")
struct WhisperAlignmentTests {
    @Test("Dynamic time warping takes HF's path")
    func dynamicTimeWarping() {
        let cost: [[Double]] = [[1, 2, 3, 0.5, 2], [2, 0.1, 2, 3, 1], [3, 2, 0.2, 0.3, 4]]
        let (rows, columns) = WhisperAlignment.dynamicTimeWarping(cost: cost)
        #expect(rows == [0, 1, 2, 2, 2])
        #expect(columns == [0, 1, 2, 3, 4])
    }

    @Test("The median filter reflects at the edges, as HF's does")
    func medianFilter() {
        let row: [Double] = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        #expect(WhisperAlignment.medianFilter(row, width: 7) == [1, 3, 3, 3, 4, 5, 5, 5, 6])
        #expect(WhisperAlignment.medianFilter([2, 1, 3], width: 7) == [2, 1, 3])
    }

    @Test("Each step's token starts where its attention peaks along the warp")
    func tokenFrames() {
        // attention sweeping 5.5 frames per step, plus a per-head ripple; 34 of 40 frames valid
        let (steps, heads, frames) = (6, 2, 40)
        var values: [Float] = []
        for step in 0..<steps {
            for head in 0..<heads {
                for frame in 0..<frames {
                    let peak = 4 * exp(-pow(Double(frame) - 5.5 * Double(step) - 2, 2) / 6)
                    values.append(Float(peak + 0.3 * sin(1.3 * Double(head) + 0.37 * Double(frame))))
                }
            }
        }
        let scores = WhisperAlignment.Scores(heads: heads, frames: frames, values: values)
        #expect(WhisperAlignment.tokenFrames(scores, validFrames: 34) == [0, 5, 10, 16, 21, 27])
    }

    /// Token pieces as Whisper's byte-level BPE decodes them, leading space and all.
    private let pieces: [Int: String] = [
        1: " And", 2: " so", 3: ",", 4: " my", 5: " fel", 6: "low", 7: " (", 8: "Amer", 9: "icans",
        10: ")", 11: ".",
    ]
    private let eot: Int32 = 99

    private func words(_ tokens: [Int32], frames: [Int]) -> [TimedWord] {
        WhisperAlignment.words(tokens: tokens, frames: frames, eot: eot, endFrame: 120) { ids in
            ids.map { pieces[$0]! }.joined()
        }
    }

    @Test("Subwords join, punctuation joins its neighbour, and a word ends where the next starts")
    func words() {
        let tokens: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, eot]
        let frames = [0, 30, 42, 50, 56, 60, 70, 75, 80, 95, 100, 104]
        let expected = [
            TimedWord(text: "And", start: 0, end: 0.6),
            TimedWord(text: "so,", start: 0.6, end: 0.84),
            TimedWord(text: "my", start: 1.0, end: 1.12),
            TimedWord(text: "fellow", start: 1.12, end: 1.4),
            TimedWord(text: "(Americans).", start: 1.4, end: 1.9),
        ]
        #expect(words(tokens, frames: frames) == expected)
    }

    @Test("A window the step limit cut short ends its last word at the end frame")
    func wordsWithoutEOT() {
        let words = words([1, 2, 3, 4, 5, 6], frames: [0, 30, 42, 50, 56, 60])
        #expect(words.map(\.text) == ["And", "so,", "my", "fellow"])
        #expect(words.last == TimedWord(text: "fellow", start: 1.12, end: 2.4))
    }

    @Test("A split UTF-8 character waits for its last byte, and languages without spaces split per character")
    func unicode() {
        // "日本" with 本 spread over two tokens, as byte-level BPE can emit it
        let tokens: [Int32] = [1, 2, 3, eot]
        let decode: ([Int]) -> String = { ids in
            let bytes: [Int: [UInt8]] = [1: Array("日".utf8), 2: [0xE6], 3: [0x9C, 0xAC]]
            return String(decoding: ids.flatMap { bytes[$0]! }, as: UTF8.self)
        }
        let words = WhisperAlignment.words(
            tokens: tokens, frames: [0, 10, 20, 30], eot: eot, endFrame: 40, splitsOnSpaces: false,
            decode: decode)
        #expect(words.map(\.text) == ["日", "本"])
        #expect(words.map(\.start) == [0, 0.2])
    }
}

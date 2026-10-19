// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// One speaker's turn: a run of activity, with short pauses bridged.
public struct SpeakerSegment: Sendable, Codable, Equatable {
    public var speaker: Int
    public var start: Double
    public var end: Double

    public init(speaker: Int, start: Double, end: Double) {
        self.speaker = speaker
        self.start = start
        self.end = end
    }
}

extension SpeakerActivity {
    /// How activity becomes speaker turns.
    public struct Segmentation: Sendable, Equatable {
        /// Activity at or above this counts as speaking.
        public var threshold: Float = 0.5
        /// Pauses shorter than this stay inside a turn.
        public var bridgeSeconds: Double = 0.3
        /// Turns shorter than this are dropped.
        public var minSegmentSeconds: Double = 0.3

        public init() {}
    }

    /// Each speaker's turns, bridging short pauses and dropping blips, in start order.
    public func segments(_ settings: Segmentation = Segmentation()) -> [SpeakerSegment] {
        var segments: [SpeakerSegment] = []
        for s in 0..<numSpeakers {
            var runs: [SpeakerSegment] = []
            var runStart: Int?
            for frame in 0...numFrames {
                let active = frame < numFrames && probability(frame: frame, speaker: s) >= settings.threshold
                if active, runStart == nil { runStart = frame }
                if !active, let start = runStart {
                    let run = SpeakerSegment(
                        speaker: s, start: Double(start) * frameSeconds, end: Double(frame) * frameSeconds)
                    if let last = runs.last, run.start - last.end < settings.bridgeSeconds {
                        runs[runs.count - 1].end = run.end
                    } else {
                        runs.append(run)
                    }
                    runStart = nil
                }
            }
            segments += runs.filter { $0.end - $0.start >= settings.minSegmentSeconds }
        }
        return segments.sorted { $0.start < $1.start }
    }
}

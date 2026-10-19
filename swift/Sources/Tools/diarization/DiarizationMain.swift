// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIDiarization
import CoreAISpeech
import Foundation

/// Who spoke when: Nemotron-3-Diarization speaker turns, without transcription.
@main
struct DiarizationRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diarization",
        abstract: "Speaker turns (speaker ID, start, end) from a Nemotron-3-Diarization bundle.")

    @Option(help: "Nemotron-3-Diarization bundle directory (from models/nemotron3-diarization/export.py).")
    var diarization: String

    @Option(help: "Audio file.")
    var audio: String

    @Option(help: "Diarization mode: offline, low_latency, very_low_latency or ultra_low_latency.")
    var mode = "low_latency"

    @Option(help: "Activity threshold for a speaker turn.")
    var threshold: Float = SpeakerActivity.Segmentation().threshold

    @Option(help: "Pauses shorter than this many seconds stay inside a turn.")
    var bridge: Double = SpeakerActivity.Segmentation().bridgeSeconds

    @Option(help: "Turns shorter than this many seconds are dropped.")
    var minSegment: Double = SpeakerActivity.Segmentation().minSegmentSeconds

    @Option(help: "Write the turns as RTTM here.")
    var rttm: String?

    @Option(help: "Write the turns, speaker activity and timing here as JSON.")
    var json: String?

    func run() async throws {
        let clock = ContinuousClock()
        var mark = clock.now
        let diarizer = try await Diarizer(bundleAt: URL(filePath: diarization), mode: mode)
        let loadSeconds = (clock.now - mark).seconds

        let pcm = try MelSpectrogram.loadAndResample(URL(filePath: audio), targetSampleRate: 16_000)
        let seconds = Double(pcm.count) / 16_000
        print("audio \(String(format: "%.1f", seconds)) s | load: diarizer \(String(format: "%.1f", loadSeconds)) s (\(await diarizer.assetName))")

        mark = clock.now
        let (activity, timing) = try await diarizer.diarize(pcm: pcm)
        let diarizeSeconds = (clock.now - mark).seconds

        var settings = SpeakerActivity.Segmentation()
        settings.threshold = threshold
        settings.bridgeSeconds = bridge
        settings.minSegmentSeconds = minSegment
        let segments = activity.segments(settings)

        print(String(
            format: "diarization %.2f s (%d steps: %.0f ms steps, %.0f ms host, %.0f ms features), %.0fx real time",
            diarizeSeconds, timing.steps, timing.stepsMs, timing.hostMs, timing.featuresMs, seconds / diarizeSeconds))
        print("speakers: \(Set(segments.map(\.speaker)).count), turns: \(segments.count)\n")
        for segment in segments {
            print("[\(Self.clock(segment.start)) - \(Self.clock(segment.end))] Speaker \(segment.speaker)")
        }

        if let rttm {
            let file = URL(filePath: audio).deletingPathExtension().lastPathComponent
            let lines = segments.map {
                String(format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>",
                       file, $0.start, $0.end - $0.start, $0.speaker)
            }
            try (lines.joined(separator: "\n") + "\n").write(to: URL(filePath: rttm), atomically: true, encoding: .utf8)
        }

        if let json {
            struct Output: Encodable {
                let segments: [SpeakerSegment]
                let activity: [Float]
                let numSpeakers: Int
                let frameSeconds: Double
                let timing: DiarizationTiming
            }
            let output = Output(
                segments: segments, activity: activity.probabilities, numSpeakers: activity.numSpeakers,
                frameSeconds: activity.frameSeconds, timing: timing)
            try JSONEncoder().encode(output).write(to: URL(filePath: json))
        }
    }

    private static func clock(_ seconds: Double) -> String {
        String(format: "%02d:%04.1f", Int(seconds) / 60, seconds.truncatingRemainder(dividingBy: 60))
    }
}

extension Duration {
    fileprivate var seconds: Double { Double(components.seconds) + Double(components.attoseconds) * 1e-18 }
}

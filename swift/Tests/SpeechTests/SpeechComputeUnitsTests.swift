// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAISpeech

@Suite("Speech compute-unit selection")
struct SpeechComputeUnitsTests {
    @Test("The default policy leaves the encoder automatic and pins the decode graphs to CPU")
    func defaultPolicy() {
        // Encoder: neither encoder lowers to the Neural Engine, and the runtime's
        // own fallback already lands them on the GPU, which measured fastest.
        // Decoder: per-token graphs see a new shape every step, which the
        // accelerator backends re-specialize on and the CPU backend does not.
        #expect(SpeechComputeUnits.default.encoder == .automatic)
        #expect(SpeechComputeUnits.default.decoder == .cpu)
    }

    @Test("The unspecialized policy is the pre-measurement baseline")
    func unspecializedPolicy() {
        // Kept so a new platform can be A/B'd against the old behaviour.
        #expect(SpeechComputeUnits.unspecialized.encoder == .automatic)
        #expect(SpeechComputeUnits.unspecialized.decoder == .automatic)
    }

    @Test("Only .automatic declines to pin a compute unit")
    func automaticMeansNoOptions() {
        #expect(SpeechComputeUnit.automatic.specializationOptions == nil)
        for unit in [SpeechComputeUnit.cpu, .gpu, .neuralEngine] {
            #expect(unit.specializationOptions != nil, "\(unit) should pin a compute unit")
        }
    }

    @Test("Raw values round-trip, so the CLI can parse them")
    func rawValuesRoundTrip() {
        // `--compute-decoder cpu` reaches this through ExpressibleByArgument, which
        // is derived from RawRepresentable.
        for unit in SpeechComputeUnit.allCases {
            #expect(SpeechComputeUnit(rawValue: unit.rawValue) == unit)
        }
        #expect(SpeechComputeUnit(rawValue: "cpu") == .cpu)
        #expect(SpeechComputeUnit(rawValue: "neuralEngine") == .neuralEngine)
        #expect(SpeechComputeUnit(rawValue: "ane") == nil)
    }

    @Test("A policy can be built per stage")
    func customPolicy() {
        let policy = SpeechComputeUnits(encoder: .gpu, decoder: .neuralEngine)
        #expect(policy.encoder == .gpu)
        #expect(policy.decoder == .neuralEngine)
        #expect(policy != .default)
    }
}

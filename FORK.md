# Fork divergences

This fork of [`apple/coreai-models`](https://github.com/apple/coreai-models) tracks
upstream `main` and adds work that is not upstream. This file indexes everything that
differs, so a rebase can be reasoned about one entry at a time.

Rebased onto upstream `8cec4f8` on 2026-09-01 (previously based on `c21ec9e`, #174 —
the earlier "forked at `f401272`" note in this file was already stale by nine commits).

Upstream's **Live Transcription with Parakeet v3** (`684ae8e`, #184) landed in that
range and is the only upstream change that touches this fork's work. See
[Relationship to upstream streaming](#relationship-to-upstream-streaming) below.

---

## Whisper KV-cache export

**Added:** `models/whisper/export_kv.py`, `models/whisper/verify_kv.py`,
`models/whisper/README_KV.md`

Upstream has a consumer with no producer. `CoreAISpeech`'s `WhisperDecoder` already
drives a `keyCache`/`valueCache` decoder step and `SpeechRecognitionBundle` already
resolves `encoder.aimodel` + `decoder.aimodel`, but nothing in the repo emits that
bundle — `models/whisper/export.py` still produces the monolithic graph that
recomputes the encoder for every generated token. This adds the missing export.

Decode goes from **1703 ms/step to 4.8 ms/step** on `whisper-large-v3-turbo`.

Two things to know before rebasing:

- The tensor named `encoder_hidden_states` carries **packed per-layer cross-attention
  K/V**, not hidden states. Without it the decoder would reproject 1500 encoder frames
  per token (19.66 GMAC vs 0.079 for the rest of the step). If upstream ever ships its
  own Whisper KV export, the two encoders will not be interchangeable.
- The export depends on runtime behaviour that upstream does not document: the KV
  state arrives uninitialized, and the forced prefix's last token is fed twice.

Full detail in [`models/whisper/README_KV.md`](models/whisper/README_KV.md).

---

## Swift changes

Two edits, both in support of the Whisper KV bundle. Kept as small as possible.

### Beta-SDK compilation fixes

Upstream `main` does not compile against the macOS 27 beta SDK in this checkout. The
fixes are spread over a few files and are **not specific to this fork's work** — they
should go upstream.

Two distinct defects:

**`let` where the SDK now needs `var`.** `mutableView(as:)` returns a view whose
`withUnsafeMutablePointer` is `mutating`, so every `let view = array.mutableView(...)`
fails to compile. Sites: `CoreAIShared/Runtime/NDArray+Helpers.swift:63`,
`CoreAIObjectDetector`, `CoreAIDiffusionPipeline`, `CoreAILanguageModels`,
`Tools/diffusion-runner`.

**Byte-level access to a bfloat16 array.** `rawView().withUnsafeBytes` and
`mutableRawView().withUnsafeMutableBytes` no longer exist. The obvious substitution
does not work either: *every* typed view — `NDArray.view(as:)`,
`NDArray.mutableView(as:)`, and `RawView`/`MutableRawView`'s own `view(as:)` — asserts
that the element type matches the array's scalar type, and `BFloat16` is not a public
type anywhere in the SDK. So a bfloat16 array cannot be viewed as itself from outside
the runtime; it compiles and then traps at run time. `RawSpan` / `MutableRawSpan` is
the one door that does not type-check, and is what these sites now use:
`CoreAIShared/Runtime/NDArray+Helpers.swift` (`flattenBFloat16NDArray`),
`CoreAIDiffusionPipeline` (`packFloatsIntoNDArray`), and
`Tests/CoreAISharedTests/NDArrayHelpersTests.swift`.

`CoreAIVisionLanguageModel.swift:32` has separate drift (`LanguageModelCapabilities`
gained a `capabilities:` argument label), fixed the same way.

**The package now builds and tests in full**, which the previous note said it could
not. Two test bundles still fail to *load* on this machine — `LanguageModelsTests` and
`GuidedGenerationTests` — because the installed `FoundationModels.framework` predates
the `capabilities:` signature the SDK compiles against. That is an OS/SDK mismatch,
not a fork problem, and it blocks a plain `swift test`; run the bundles directly to
work around it:

```sh
swift build --build-tests
xcrun xctest .build/out/Products/Debug/SpeechTests.xctest
```

408 tests pass across `SpeechTests` (163), `CoreAISharedTests` (53),
`DiffusionPipelineTests` (67), `ObjectDetectorTests` (22), `ImageSegmenterTests` (60)
and `CoreAILMCommonTests` (43).

### Long-form audio: overlapping windows

**Added:** `swift/Sources/CoreAISpeech/LongFormWindowing.swift`,
`swift/Tests/SpeechTests/LongFormWindowingTests.swift`
**Changed:** `SpeechRecognitionModel.transcribe`, `DecodeStats`, `SpeechRecognizerMain`

Upstream decodes exactly one encoder window and silently discards the rest of the
file: `MelSpectrogram.padToFrameGrid` truncates PCM to 480 000 samples and
`decodeAudio` runs the encoder once, with no chunk loop anywhere. A 62 s file
transcribed as its first 30 s, with nothing in the output to say so.

`transcribe` now splits audio longer than the encoder's window into consecutive
windows overlapping by `overlapSeconds` (default 2), decodes each, and joins the
transcripts with the duplicated words at each seam removed. `DecodeStats.windowCount`
reports the split and the CLI prints it. `--window-overlap` tunes it; `0` restores
butted windows with no de-duplication.

Re-measured after the `8cec4f8` rebase, on the same 62.3 s clip
(`whisper-large-v3-turbo-kv_float16`, release build, Apple silicon Mac):
**3 windows, 209 steps, 5.0 ms/step, 1559 ms end to end** (1036 ms of it decode, 66%).
Without the overlap the 30 s seam lands mid-sentence and six words are lost outright
("My biological mother was a young,"); with it they are recovered and no duplicate
4-gram appears anywhere in the output — still true on the rebased tree.

The pre-rebase figures were 200 steps at 6.9 ms/step and 2133 ms. Two things moved:
the step count is +9 because language detection is now on by default (one extra step,
and a slightly different transcript), and the per-step time is down because upstream's
decoder rewrite stopped flattening the logits to `[Float]` on every step. Do not read
the two as a controlled comparison.

This is the naive long-form strategy, one step up from the pre-fork
`WhisperASR.swift`, which used fixed non-overlapping windows and documented the
clipping. It is still weaker than Whisper's own algorithm, which emits timestamp
tokens and advances to the end of the last complete segment — that would need
timestamps in the forced prefix and a decode loop that reads them. Seam matching is
also a heuristic: when the two windows transcribe the shared audio differently no
run matches, and the text is concatenated rather than guessed at, so the failure
mode is a repeated word rather than dropped speech.

Windowing keys off `MelConfig.nFrames`, so a dynamic encoder (Parakeet's, unless
exported at a static length) still takes the whole waveform in one pass.

<a name="relationship-to-upstream-streaming"></a>
#### Relationship to upstream streaming

Upstream's `684ae8e` added a genuine streaming path — `startStream` / `append(pcm:)` /
`finishStream`, with a `[left | chunk | right]` window and the transducer state carried
across hops. **It does not replace this windowing, and the two do not overlap:**

- Streaming is **Parakeet-only** and needs a bundle exported with `--streaming`; it
  rejects a static or dynamic bundle. This windowing is what makes **Whisper's** fixed
  30 s encoder usable on long files, and Whisper has no streaming path at all.
- Streaming carries decoder state across chunk boundaries. This stitches *text* at the
  seam, because a Whisper window is decoded independently and there is no transducer
  state to carry.

So both stay. For a long *file* on Parakeet, this windowing is still the batch path;
upstream streaming is for live audio, and costs more encoder work per second of audio
(it re-encodes the whole window every hop).

### Compute-unit selection: decode graphs pinned to CPU

**Added:** `SpeechComputeUnit` / `SpeechComputeUnits` in `SpeechRecognitionBundle.swift`,
`--compute-encoder` / `--compute-decoder` on the CLI

Upstream loads every speech asset with bare `AIModel(contentsOf:)` — default
specialization, which probes the Neural Engine first. Measured, that is the wrong
default for the per-token graphs of *both* architectures, for the same reason: they
are tiny and launch-bound, and each step presents a shape the accelerator backends
have not specialized yet (Whisper's `position_ids` grows by one per token; Parakeet's
TDT loop runs its two graphs once or more per emitted token). The CPU backend does
not specialize per shape.

The encoder is left `.automatic`: it wants the GPU, and the runtime's own fallback
already lands it there (see the Neural Engine note below).

Re-measured after the `8cec4f8` rebase on an 11 s clip (`jfk.wav`), release build,
Apple silicon Mac. The original 5.5 s clip is not in this tree, so the absolute
figures are not comparable with the pre-rebase ones below them — **the conclusions
are unchanged, and that is the point of re-running them.**

**Whisper** (`whisper-large-v3-turbo-kv_float16`), per decode step, encoder default:

| Decoder placement | ms/step | end to end |
| --- | ---: | ---: |
| Default specialization | 1482.3 | 40 171 ms |
| **CPU** | **5.0** | **263 ms** |

A ~296x difference, and unambiguous. (Pre-rebase, on the 5.5 s clip: 1576.2 vs 4.9.)

**Parakeet TDT** (`float32_static`), per decode step, encoder default:

| Decode graphs | ms/step | decode total |
| --- | ---: | ---: |
| Default specialization | 0.7 | 30.4 ms |
| GPU | 0.7 | 32.2 ms |
| **CPU** | **0.4** | **17.7 ms** |

Still a ~1.7x faster decode loop on CPU. (Pre-rebase the ratio was ~2.4x; the absolute
totals dropped a lot because upstream's decoder rewrite reads the argmax straight out
of the `NDArray` instead of flattening the logits row to `[Float]` every step, and
slices the encoder output per frame instead of flattening it whole up front.)

**Encoder placement**, decoder pinned to CPU, same 11 s clip, end to end:

| encoder | parakeet f32 static | parakeet f16 static | whisper KV f16 |
| --- | ---: | ---: | ---: |
| automatic (the default) | **105.7 ms** | **39.4 ms** | **260.6 ms** |
| GPU | 106.7 ms | 40.1 ms | 270.5 ms |
| Neural Engine | 108.9 ms | 127.9 ms | 600.5 ms |
| CPU | 387.5 ms | — | — |

`.automatic` remains the right default for the encoder on Mac for all three. Note the
f16 Parakeet export is **2.7x faster end to end than f32** (39.4 vs 105.7 ms) — worth
knowing, since `CoreAIModel.parakeetTDT06BV3` still resolves to the f32 bundle.

The per-graph medians that used to sit here came from a separate `AIModel.load`
harness that is not in this tree, so they have **not** been re-measured and have been
dropped rather than left to rot.

There is precedent for pinning: `SpeechRecognizerMain.swift` already pins the
monolithic Whisper path to CPU, commented *"the monolithic f16/f32 Whisper export
decodes incorrectly on the default compute path"*.

For Whisper the alternative — a fixed-length `position_ids`, or setting
`expectFrequentReshapes` as `ModelStructure.dynamic` does — would be a larger change
to the runtime contract.

#### These numbers are Mac-only

Everything above was measured on Apple silicon **Mac**. iOS has a different balance:
the GPU is several times smaller while the Neural Engine is comparable silicon, so
the GPU-beats-ANE result for the encoder may well invert, and the CPU pin for the
decode graphs deserves re-checking against battery as well as latency.

The policy is therefore a parameter, not a constant:

```swift
// The measured Mac default.
let model = try await SpeechRecognitionModel(resourcesAt: url)

// Re-measuring on device: override per stage.
let model = try await SpeechRecognitionModel(
    resourcesAt: url,
    computeUnits: SpeechComputeUnits(encoder: .neuralEngine, decoder: .neuralEngine))

// The pre-measurement baseline, for A/B.
let model = try await SpeechRecognitionModel(
    resourcesAt: url, computeUnits: .unspecialized)
```

and on the CLI:

```sh
speech-recognizer --model <bundle> --audio-path a.wav \
  --compute-encoder gpu --compute-decoder cpu    # automatic | cpu | gpu | neuralEngine
```

#### Neural Engine: the lowering bug is gone, and the ANE still loses

This section previously recorded that both encoders failed to lower to the ANE with a
mis-inferred rank-promoting broadcast (`memref<1x1500x1500x1280xf16>` for Whisper,
`memref<1x376x376x1024xf16>` for Parakeet), that Parakeet's *aborted the process* in
Python, and that the Swift runtime only survived it by falling back. The
`SpeechComputeUnits` doc comment said neither lowers; `CLAUDE.md` in the Circe repo
recorded a third signature, *"ANE regionCall op not found"*.

**None of that reproduces on the rebased tree.** Re-measured with
`--compute-encoder neuralEngine` on the 11 s clip:

| bundle | ANE warnings emitted | end to end (ANE) | end to end (automatic) |
| --- | --- | ---: | ---: |
| `whisper-large-v3-turbo-kv_float16` | **none** | 600.5 ms | 260.6 ms |
| `parakeet-tdt-0.6b-v3_float16_static` | **none** | 127.9 ms | 39.4 ms |

Both f16 encoders now lower to the Neural Engine cleanly and run — no abort, no
fallback, no diagnostics on stderr at all. The ANE is simply **2.3x (Whisper) and
3.2x (Parakeet) slower than letting the runtime choose**, which lands them on the GPU.

The **f32** exports are a separate matter and not a bug: the ANE has no f32 path, so
`--compute-encoder neuralEngine` on `parakeet-tdt-0.6b-v3_float32_static` produces a
wall of

```
Incompatible element type for ANE: expected fp16, f8E4M3, si8, ui8, si16, or ui16
failed: ANE I/O op can only do F16 MemRef <-> F32 Tensor cast
```

and falls back, arriving at the same 108.9 ms as `.automatic`. Pinning the ANE for an
f32 bundle costs a failed probe and buys nothing.

So the conclusion that put the encoder on `.automatic` still holds — but for a
different reason than this file used to give. It is not that the ANE cannot be
reached; it is that reaching it is slower. Palettized weights remain the only thing
that would make it interesting (the pre-fork measurements put a 4-bit palettized
encoder at ~38 ms on ANE).

---

### Whisper language selection and detection

**Changed:** `swift/Sources/CoreAISpeech/SpeechRecognitionModel.swift`,
`swift/Sources/CoreAISpeech/SpeechRecognitionBundle.swift`,
`swift/Sources/CoreAISpeech/SpeechDecoder.swift`,
`swift/Sources/Tools/speech-recognizer/SpeechRecognizerMain.swift`
**Added:** `swift/Sources/CoreAISpeech/SpeechLanguage.swift`,
`swift/Tests/SpeechTests/SpeechLanguageTests.swift`

A Whisper export carries its decode language in the forced decoder prefix from
`generation_config.json` — `<|startoftranscript|><|en|><|transcribe|>` — and
upstream offers no way to change it. That is not a graceful limitation: fed
French audio, an `<|en|>` export does not fail, it transcribes the audio *as if*
it were English and emits a fluent English translation. Measured on a French
clip, the same bundle scored 96% WER against a French reference while whisper.cpp,
Parakeet and Apple all scored 11%. Nothing in the output says the language was
wrong.

`transcribe(pcm:overlapSeconds:language:)` now takes a `SpeechLanguage`:

- `.detect` (**the default**) — predicts the language from the audio.
- `.code("fr")` — decode as that language; accepts `"fr"`, `"fr-FR"` or `"fr_FR"`.
- `.bundleDefault` — whatever the export pinned, i.e. the previous behaviour.

Detection is one decoder step: Whisper's first prediction after
`<|startoftranscript|>` *is* the language token, so `WhisperDecoder.detectLanguageToken`
primes the cache with the start token and takes the argmax **restricted to the
language tokens**. The restriction is load-bearing — unrestricted, the top token
at that position is frequently ordinary text. Cost is ~110 ms on a 62 s clip
(1390 ms → 1500 ms end to end).

For long-form audio the language is detected **once, on the first window**, and
reused. A recording is almost always one language, and per-window detection would
both cost a step each time and let a noisy window flip the language mid-file.

Language tokens are read from the bundle's own `added_tokens.json` (falling back
to `vocab.json`) rather than hardcoded: the set and numbering differ between
Whisper generations, and language tags are separable from control tokens by being
two- or three-letter codes. A bundle with no token table falls back to its pin
rather than failing, which is how Parakeet — natively multilingual, and ignoring
this entirely — behaves.

`GenerationConfig` gained `languageToken` and `retargeted(toLanguageToken:)`,
which replaces only prefix slot 1 so the export's task and timestamp tokens
survive, and its memberwise initializer is now public.

`speech-recognizer` gained `--language` (`auto` | `bundle` | a code) and reports
which language it used and how it was chosen:

```
$ speech-recognizer --model exports/whisper-large-v3-turbo-kv_float16 --audio-path fr.wav
  language: fr (detected)
  Malheureusement, l'étude du flux de circulation est difficile car …

$ … --language bundle
  language: en (pinned by the bundle)
  Unfortunately, the study of circulation is difficult because …
```

**Rebase note:** the default changed from the bundle's pin to detection, so a
caller that relied on always decoding as the pinned language must now pass
`.bundleDefault` explicitly. `transcribeCapturingStages(pcm:)` is untouched and
still uses the pinned prefix, so the PyTorch parity traces are unaffected.

---

### The merged `transcribe` signature

The `8cec4f8` rebase had to fold this fork's two `transcribe` parameters together with
upstream's. Keeping them as separate overloads was not an option: every parameter past
the first has a default, so a bare `transcribe(pcm:)` would have been ambiguous.

```swift
public func transcribe(
    pcm: [Float],
    overlapSeconds: Double = SpeechRecognitionModel.defaultOverlapSeconds,  // this fork
    language: SpeechLanguage = .detect,                                     // this fork
    resetAfterSilenceFrames: Int = 0                                        // upstream
) async throws -> (String, DecodeStats)
```

`transcribe(audioURL:…)` takes the same three. `resetAfterSilenceFrames` is applied
**per window** on the long-form path, which is free of consequence: each window is
decoded independently, so no predictor state crosses a seam in the first place.

Upstream also removed `decodeAudio(from:)` — this fork had already folded it into
`transcribe(audioURL:)`, so the rebase kept the fork's shape.

Two silent breakages this rebase had to fix, neither of which git flagged as a
conflict — worth knowing about for the next one:

- `runEncoder` changed arity, from `(NDArray, [Int])` to `(NDArray, [Int], Int)`; the
  valid-frame count moved inside it. `resolvedResources(forLeadingPCM:language:)`
  destructured the old pair in a hunk that merged cleanly and then did not compile.
- `validEncoderFrames` went from an instance method to a `package static` taking
  `config:` and `subsamplingFactor:`, **and changed meaning**: from a proportional
  estimate rounded to nearest, to an exact per-stage stride-2 derivation. Whisper is
  unaffected (`WhisperDecoder` documents that it ignores the value and stops on EOT);
  Parakeet's last partial window now gets an exact cap instead of a rounded one.

---

### `speech-recognizer`: decode statistics vs end-to-end time

**Changed:** `swift/Sources/Tools/speech-recognizer/SpeechRecognizerMain.swift`

The bundle path printed the wall clock for the whole `transcribe(pcm:)` call as
`[… ms total]` on the end of the **Decode** line, next to the per-step statistics.
Read naturally that says decoding accounts for all of it, when on a multi-window
file the decoder steps are around two thirds and the rest is the mel spectrogram
and the encoder passes.

Split into two sections: `Decode` keeps the step statistics, and a new `Timing`
section reports the end-to-end figure alongside the summed step time and its
share of the total.

```
── Decode ─────
  steps: 200  latency: 6.9 ms/step  speed: 144.3 steps/s  min: 5.0 ms  max: 9.2 ms
  windows: 3 (audio longer than one encoder window)

── Timing ─────
  transcribe:   2132.6 ms   mel + encoder + decode, end to end
  decode:       1386.2 ms   sum of the 200 steps above (65% of total)
```

The legacy monolithic path was already unambiguous — it reports step statistics
and no total — and is unchanged.

---

## Known-benign warnings

### `Tokenizer model class ParakeetTokenizer is not registered`

Every Parakeet bundle load prints:

```
Warning: Tokenizer model class ParakeetTokenizer is not registered,
falling back to a standard BPE implementation.
```

It is cosmetic, and for this model the fallback is exactly right. Investigated
rather than assumed, so it does not get rediscovered:

- The exported `processor/tokenizer_config.json` declares
  `tokenizer_class: "ParakeetTokenizer"`. swift-transformers looks that name up in a
  hardcoded table (`Tokenizer.swift:158-177`) and falls back with
  `knownTokenizers[name] ?? BPETokenizer.self` (`:191`).
- **The fallback picks the same class a registered entry would.** Almost every entry
  in that table maps to `BPETokenizer.self` — Llama, Gemma, GPT2, Qwen2, Whisper,
  CodeGen, Cohere, Falcon, Roberta, `PreTrainedTokenizer`. Only Bert, T5 and
  XLM-Roberta differ. Parakeet's `tokenizer.json` carries `model.type: BPE`, so
  registering `"ParakeetTokenizer": BPETokenizer.self` would change nothing but the
  warning.
- **The behaviour that matters is not chosen by the class.** The pre-tokenizer and
  decoder come from `tokenizer.json` via
  `DecoderFactory.fromConfig(config: tokenizerData["decoder"], …)`
  (`Tokenizer.swift:531`). Parakeet's is `Metaspace` with replacement `▁`,
  which is what turns those markers back into spaces — observably working, since
  transcripts read `The quick brown fox` and not `▁The▁quick` or `Thequickbrown`.

**The one real risk** is `strict` mode: the same code path *throws*
`TokenizerError.unsupportedTokenizer` instead of warning (`Tokenizer.swift:193-197`).
`SpeechRecognitionBundle` reaches it through `AutoTokenizer.from(modelFolder:)`, which
is non-strict, so a Parakeet bundle loads. Any future caller taking the strict path
would fail outright rather than warn.

Fixing it properly means registering the class upstream in swift-transformers (a
pinned dependency at 1.2.0). Rewriting `tokenizer_class` in the exported processor
would silence it locally but make the bundle diverge from what HuggingFace emits, so
it is deliberately not done.

---

## Not yet ported from the pre-fork tree

Work that exists in the older `coreai-models.obsolete` checkout (forked at `0c1055f`,
71 commits behind) and has no upstream equivalent:

- **Parakeet encoder compression** — a 61-config quantization/palettization sweep plus
  4-bit and 8-bit palettized exports. Best measured: 43.2 dB at 2.58× (`palette_6b_gs4`),
  30.4 dB at 3.93× (`palette_4b_gs4`). Upstream has no compression work for Parakeet.
  Rebasing means retargeting onto upstream's 3-graph HuggingFace-sourced export.
- **Example iOS/macOS Xcode apps** — Whisper, Parakeet and SpeechAnalyzer transcriber
  apps. Upstream ships no `examples/`.
- **`sam-3d-body` export.**

Upstream's own Parakeet (`#136`) supersedes the pre-fork NeMo-sourced version:
it is HuggingFace-sourced, split into three graphs, and has a Swift runtime with tests.

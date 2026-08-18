# Fork divergences

This fork of [`apple/coreai-models`](https://github.com/apple/coreai-models) tracks
upstream `main` and adds work that is not upstream. This file indexes everything that
differs, so a rebase can be reasoned about one entry at a time.

Forked at `f401272`.

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

### `swift/Sources/CoreAIShared/Runtime/NDArray+Helpers.swift:63`

```diff
-    let view = array.mutableView(as: type)
+    var view = array.mutableView(as: type)
```

Upstream `main` does not compile against the macOS 27 beta SDK in this checkout —
`fillNDArray` calls a mutating member on a `let` binding. Nothing in the package
builds without this. **Not specific to this fork's work**; it should go upstream.

The same defect exists in `CoreAIObjectDetector`, `CoreAIDiffusionPipeline`,
`CoreAILanguageModels`, and `Tools/diffusion-runner`, and `CoreAILanguageModels/VLM/
CoreAIVisionLanguageModel.swift:32` has separate SDK drift (`missing argument label
'capabilities:'`). Those are left alone — they do not affect `speech-recognizer`, but
they do mean **`swift test` cannot build the full package** in this checkout.

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

Measured on a 62.3 s clip: 3 windows, 200 steps, 3.1 s wall clock. Without the
overlap the 30 s seam lands mid-sentence and six words are lost outright
("My biological mother was a young,"); with it they are recovered and no duplicate
4-gram appears anywhere in the output.

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

**Whisper**, 5.5 s clip, per decode step:

| Decoder placement | ms/step |
| --- | ---: |
| Default specialization | 1576.2 |
| **CPU** | **4.9** |

**Parakeet TDT**, same clip, per decode step (encoder unchanged at ~26 ms):

| Decode graphs | ms/step | decode total |
| --- | ---: | ---: |
| Default specialization | 0.7 | 52.8 ms |
| GPU | 0.8 | 55.3 ms |
| **CPU** | **0.3** | **37.8 ms** |

Per-graph medians behind that, from `AIModel.load` directly:

| graph | automatic | ANE | GPU | CPU |
| --- | ---: | ---: | ---: | ---: |
| parakeet encoder (30 s window) | *aborts* | 120.97 | **26.06** | 188.97 |
| parakeet decoder_step | 0.79 | 0.67 | 0.72 | **0.12** |
| parakeet joint | *fails to load* | 0.28 | 0.62 | **0.18** |

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

#### Neural Engine: both encoders hit the same lowering bug

Whisper's and Parakeet's encoders both fail to lower to the ANE with the same
signature — a rank-promoting broadcast the compiler mis-infers:

```
whisper:  inferred 'memref<1x1500x1500x1280xf16>' vs 'memref<1x1x1500x1280xf16>'
parakeet: inferred 'memref<1x376x376x1024xf16>'   vs 'memref<1x1x376x1024xf16>'
```

For Whisper this is **fixed in the export** (`_BatchedPositions`, see
`models/whisper/README_KV.md`). For Parakeet it is not, and there is little reason to:
even lowering cleanly, the ANE encoder is 121 ms against the GPU's 26 ms at fp16. It
would only become interesting with palettized weights — the pre-fork measurements put
a 4-bit palettized encoder at ~38 ms on ANE. Note that in Python the failure *aborts
the process*; the Swift runtime survives it by falling back, which is why the default
policy still gets a 26 ms encoder.

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

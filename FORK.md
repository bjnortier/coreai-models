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

### `swift/Sources/CoreAISpeech/SpeechRecognitionBundle.swift` — decoder pinned to CPU

`WhisperDecoder` sends `position_ids` whose length grows by one per token, so every
decode step presents a shape the accelerator backends have not specialized yet:
~300 ms of recompilation per step, and the specialization cache does not survive the
process, so it is paid on every run. The CPU backend does not specialize per shape and
runs this 4-layer single-token graph in ~5 ms.

| Decoder placement | ms/step |
| --- | ---: |
| Default specialization | 1669.3 |
| CPU | 4.8 |

There is precedent: `SpeechRecognizerMain.swift:219` already pins the monolithic
Whisper path to CPU, with the comment *"the monolithic f16/f32 Whisper export decodes
incorrectly on the default compute path"*.

The alternative — a fixed-length `position_ids`, or setting `expectFrequentReshapes`
as `ModelStructure.dynamic` does — would be a larger change to the runtime contract.

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

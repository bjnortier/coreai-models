# Nemotron 3 Diarization

Streaming speaker diarization from NVIDIA: "who spoke when", for up to eight speakers, ordered by first arrival.
A 99M-parameter, 31-layer RoPE Transformer encoder (width 512) over 80 ms frames, with a Sortformer head that
gives each speaker's activity every 10 ms. Streaming keeps an Arrival-Order Speaker Cache (AOSC) and a FIFO of
recent frames, and re-encodes `[speaker cache | FIFO | chunk | look-ahead]` from scratch at every step.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run export.py
```

Saves a bundle at `<repo-root>/exports/nemotron-3-diarization_float16/`:

| File | What it is |
| --- | --- |
| `diarization_step.aimodel` | One encoder step per latency mode, as functions sharing one copy of the weights (191 MB at float16) |
| `metadata.json` | Window geometry of every mode, its asset, the feature settings, and the speaker-cache policy |
| `silence_embeds.f32` | The learned embedding the cache policy writes into a compressed cache's silence slots (512 × float32) |

For iOS, export with `--split` (to `..._float16_split/`): one `step_<mode>.aimodel` per mode. See
[iOS](#ios).

| Flag | Description | Default |
| --- | --- | --- |
| `--model` | Model | `nvidia/Nemotron-3-Diarization` |
| `--dtype` | `float16`, `float32` | `float16` |
| `--modes` | Modes to export | all four |
| `--attention` | `sdpa` (fused) or `eager` (HF's matmul + float32 softmax) | `sdpa` |
| `--split` | One asset per mode instead of one function per mode | — |
| `--output-dir` | Output directory for the bundle | `<repo-root>/exports/` |
| `--overwrite` | Overwrite an existing bundle | — |

### Modes

Every mode is the same weights at a different static window. Latency is the audio a step waits for, its chunk
plus look-ahead, excluding compute. All frame counts are 80 ms encoder frames.

| Function | Latency | Chunk | Look-ahead | Cache window | Static window |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ultra_low_latency` | 0.32 s | 3 | 1 | 528 | 532 |
| `very_low_latency` | 0.64 s | 6 | 2 | 528 | 536 |
| `low_latency` | 1.04 s | 9 | 4 | 528 | 541 |
| `offline` | 30.4 s | 340 | 40 | 304 | 684 |

The cache window is the speaker cache (264) plus the FIFO: 264 in the streaming modes, 40 offline.

### What a step computes

```text
cached_embeds  [1, C, 512]      speaker cache + FIFO frames, LEFT-padded with zeros to C
chunk_features [1, K*8, 128]    log-mel frames of the chunk and its look-ahead, right-padded to K*8
key_mask       [1, C+K]         1 for a valid frame, 0 for padding
 -> logits       [1, (C+K)*8, 8]  speaker logits every 10 ms, over the whole window
    chunk_embeds [1, K, 512]      the chunk's encoder inputs, to push into the FIFO
```

The streaming loop and the speaker-cache policy stay on the host. torch.export cannot capture the loop. The
policy scores frames, takes a top-k and gathers them, which is a few hundred scalar operations on at most ~500
frames.

**Why left padding.** The HF step numbers its valid frames 0..n-1 for RoPE. Padding the cache on the left shifts
every valid frame by the same offset, and RoPE attention scores depend only on position differences, so one
static shape reproduces every step exactly. Padding between the cache and the chunk would not.

**Why SDPA.** HF's eager attention computes the scores, adds the mask and takes a float32 softmax as separate
ops. Exported that way, an iPhone 15 Pro allocated 600-900 MB of intermediates per step and the app was killed
within three steps. With `scaled_dot_product_attention`, the compiled graph has 1 softmax instead of 31 and 7 casts
instead of 68, and a step's memory stays flat (~10 MB).

**Why the padding is zeroed before the upsampler.** The head's kernel-3 Conv1d reads one frame either side. In
HF that neighbour is the conv's own zero padding, so the step zeroes padding frames rather than let the conv read
a padding frame's transformer output.

## Verify

```sh
uv run verify.py                                  # HF's 97.6 s, 7-speaker example, every mode
uv run verify.py --audio meeting.wav --modes low_latency
```

Runs a whole session twice per mode: once with the HF model in float32, and once with each encoder step replaced
by the exported function, driven by HF's own `Nemotron3DiarizationSpeakerCache`. Precision error compounds across
a session. The policy thresholds each step's probabilities and keeps the top-scoring frames, so a small difference
can keep a different frame and steer every later step. A single-step PSNR cannot show that.

Results on HF's `diarization_example.mp3` (97.6 s, 7 speakers), Apple M5 Pro, default specialization, SDPA export
(the single-asset and `--split` bundles give identical results):

| Mode | Steps | float16 logits PSNR | float16: frames where any speaker decision differs | Speakers found (HF / Core AI) | Step median |
| --- | ---: | ---: | ---: | --- | ---: |
| `low_latency` | 136 | 54.7 dB | 0.09% | 7 / 7 | 8.8 ms |
| `very_low_latency` | 203 | 44.9 dB | 0.91% | 7 / 7 | 8.7 ms |
| `ultra_low_latency` | 407 | 48.9 dB | 0.45% | 7 / 7 | 8.8 ms |
| `offline` | 4 | 45.5 dB | 0.03% | 6 / 6 | 10.6 ms |

A float32 export (eager attention) matches HF to 129.7 dB (`very_low_latency`) and 136.0 dB (`offline`) with no
differing decisions, so the host loop is exact and the float16 gap is precision alone. It shows up as occasional
divergent cache selections, which is why the PSNR for a whole session varies more between modes than a single
step's would.

`offline` leaves the last two encoder frames of the recording out of the comparison. When the processor masks
the final mel frame (all padding), HF still feeds that frame's hidden state to the upsampler conv; the step zeroes
it instead. It changes the last ~90 ms of a recording and nothing else.

`--fixture-dir DIR` also writes a few steps' padded inputs and float32 PyTorch outputs, plus a `manifest.json`
naming them, so a device can check its compiled step against PyTorch without Python (see [iOS](#ios)).

## iOS

Measured on an iPhone 15 Pro (A17 Pro, `h16p`), iOS 27.2 beta, with an on-device benchmark app. Each mode is a
`--split` asset compiled ahead of time with `xcrun coreai-build compile <asset> --platform iOS --architecture h16p`.
A wrong architecture fails to load with `incompatibleCompiledAssetArchitecture(device: ...)`, which names the right
one.

| Mode | Step median | p90 | Real-time budget | Logits PSNR vs float32 PyTorch | Decisions agreeing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ultra_low_latency` | 37.0 ms | 37.2 ms | 240 ms | 56.3-66.4 dB | 100% |
| `very_low_latency` | 36.0 ms | 36.5 ms | 480 ms | 65.9-69.9 dB | 100% |
| `low_latency` | 39.2 ms | 39.7 ms | 720 ms | 67.1-69.6 dB | 99.996-100% |
| `offline` | 52.8 ms | 53.0 ms | 27,200 ms | 67.3-72.8 dB | 100% |

Steady state (speaker cache and FIFO full), 50 timed runs after 3 warm-ups. PSNR and agreement are over a step's
valid frames, for the first step and the last full step of the 97.6 s example (the `--fixture-dir` fixtures). A 60 s
run of `ultra_low_latency` at real-time cadence (250 steps, one per 240 ms) held 47-48 ms per step, max 50-59 ms,
thermal state nominal throughout. That is about 20% of real time on one compute unit.

Asking for the CPU, GPU, Neural Engine or the default made no measurable difference: the four runs gave the same
latencies and identical PSNRs to the decimal, so the compiled asset appears to decide placement. Which unit ran
is not something an app can see.

**The source asset, specialized on the phone.** The source `step_<mode>.aimodel` also loads on device, so one
bundle can serve any iPhone, at about 2.5x the compiled step's latency. Same phone and OS, cold cache, one compute
unit preference per run (`Diarizer(bundleAt:mode:options:allowsCompiledAssets: false)`):

| Mode | Neural Engine | Automatic | CPU | Compiled `h16p` |
| --- | ---: | ---: | ---: | ---: |
| `ultra_low_latency` | 93.2 ms | 94.6 ms | 146.1 ms | 37.0 ms |
| `very_low_latency` | 94.0 ms | 97.4 ms | 145.6 ms | 36.0 ms |
| `low_latency` | 96.0 ms | 98.9 ms | 147.2 ms | 39.2 ms |
| `offline` | 125.5 ms | 128.2 ms | 184.2 ms | 52.8 ms |

Step medians, measured as above. Cold loads took 1.6-2.7 s (Neural Engine, automatic) and 0.8-1.0 s (CPU). The
worst logits PSNR per mode is 64.0, 69.1, 68.5 and 77.4 dB for both the Neural Engine and automatic, identical to
the decimal, so automatic picks the Neural Engine's placement; the CPU gives 46.3-61.3 dB. Decisions agree
99.995-100% throughout. Every load, whichever unit is asked for, logs that the GPU part of the specialized program
was rejected and carries on without it:

```
'mps_spi.sdpa' op operand count (5) does not match with the total size (0) specified in attribute 'operandSegmentSizes'
Failed to validate MLIR program
```

The on-device compiler emits an SDPA op that this phone's MPS parser does not accept. Seen only there so far: an
iPad Pro M2 on iPadOS 27.0.1 and an M5 Pro Mac on macOS 27.2 (26B5101f) specialize the same source without it, so
it is either the iOS 27.2 beta or the A17 Pro. The compiled asset, built by `coreai-build` on the Mac, does not hit
it, which may be where its 2.5x comes from.

What did not work, and why the recipe looks the way it does:

- **The source `.aimodel` on device, once.** An early attempt to specialize it on the phone aborted the app with
  `std::bad_alloc` in MPSGraph after 1 GB of mapped writes. Which export that was is not recorded; the SDPA export
  above specializes on device on every compute unit, with MPS rejecting its GPU program before it allocates.
- **One asset with four functions, compiled.** It compiles to 740 MB (the weights are duplicated per function)
  and fails to specialize on every compute unit. `--split` gives four 190 MB assets that load in 0.3-9 s.
- **Eager attention.** 600-900 MB of intermediates per step; the app was killed (see "Why SDPA").
- **The Neural Engine compiler on the eager graph.** `ANELargeModelCompilerService` crashed (a pointer-auth failure
  freeing its MLIR modules) and the app went with it. Not seen with the SDPA graph.
- **A stale specialization.** After an app was killed mid-specialization, later loads came back instantly from the
  cache and `loadFunction` returned nil (`nilError`). `Diarizer` drops the cache entry and specializes again, once.

Parakeet, run on the same phone, needed its own workaround: compiled the default way, its encoder and decoder
step crash the iOS 27.2 Neural Engine compiler (`ANELargeModelCompilerService`, a pointer-auth failure freeing its
MLIR modules) on every load, which surfaces as `failedToSpecialize` after ~10 s. Asking for the CPU or GPU at load
time does not avoid it, and `.cpuOnly` finds no compiled path. Compile it with `--preferred-compute gpu` and load
it with `SpeechComputeUnits.unspecialized`. It loads in 7.7 s with an empty cache, 2 s from it.

## Swift runtime

`CoreAIDiarization` (in this repo's Swift package) runs the exported bundle:

```swift
import CoreAIDiarization
import CoreAISpeech

let diarizer = try await Diarizer(bundleAt: diarizationBundleURL, mode: "low_latency")
let pcm = try MelSpectrogram.loadAndResample(audioURL, targetSampleRate: 16_000)
let (activity, timing) = try await diarizer.diarize(pcm: pcm)   // 8 speakers every 10 ms
for turn in activity.segments() {                               // thresholded, short pauses bridged
    print(turn.speaker, turn.start, turn.end)
}
```

- `Diarizer` is a line-by-line port of HF's `Nemotron3DiarizationSpeakerCache` plus `verify.py`'s step loop.
  It loads a mode's compiled `<asset>.<arch>.aimodelc` when one is present, else the source `.aimodel`;
  `allowsCompiledAssets: false` skips the compiled ones.
  `runStep` and `repeatStep` run a single padded step, for checking it against the `--fixture-dir` fixtures and
  timing it.
- `MelConfig.nemotronDiarization` is Parakeet's front end with the new `.none` normalization.
- `SpeechRecognitionModel.transcribeWords` is new in `CoreAISpeech`, for pairing speaker turns with words. The
  Parakeet decoder now records each token's frame and TDT duration (`DecodeStats.tokenTimings`), and windows of a
  static encoder are split at the middle of their overlap by time.

On the Mac it matches the Python driver on NVIDIA's 97.6 s clip: in `offline` mode 99.96% of 10 ms frames have
identical speaker decisions, in `low_latency` 99.48% (the Python driver computes features per chunk, the Swift
one once for the recording), with the same 7 speakers.

`diarization` prints `[start - end] Speaker n` per turn and can write RTTM (`--rttm`) or the turns plus
frame-level activity as JSON (`--json`):

```sh
swift run -c release diarization --diarization exports/nemotron-3-diarization_float16_split \
    --audio meeting.wav --mode low_latency --rttm meeting.rttm
```

Unit tests: `swift build --target DiarizationTests && xcrun xctest .build/out/Products/Debug/DiarizationTests.xctest`.

## Not covered yet

- **Compression.** Palettization or 8-bit weights are untested; check them with `verify.py`, not a single-step
  PSNR. Four 190 MB assets is 760 MB per architecture; an app shipping one or two modes needs only those.
- **Energy.** The on-device runs above report latency and thermal state only. Measure power with Instruments
  before relying on the sustained figures for battery life.
- **Accuracy on real benchmarks.** `verify.py` measures agreement with HF, not DER against ground truth.

[^1]: https://huggingface.co/nvidia/Nemotron-3-Diarization, OpenMDW License 1.1.

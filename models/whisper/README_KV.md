# Whisper KV-cache export

`export_kv.py` exports Whisper as a **run-once encoder plus a stateful, single-token
decoder step**, replacing the O(n²) monolithic graph that `export.py` produces.

Upstream ships a consumer for this bundle but no producer: `CoreAISpeech`'s
`WhisperDecoder` (`swift/Sources/CoreAISpeech/SpeechDecoder.swift`) already drives a
`keyCache`/`valueCache` decoder, and `SpeechRecognitionBundle` already resolves
`encoder.aimodel` + `decoder.aimodel` — but nothing in the repo emits them, and the
format is labelled "legacy". This script fills that gap.

Measured on `whisper-large-v3-turbo`, fp16, a 5.5 s clip (19 decode steps):

| Path | ms/step | Decode total |
| --- | ---: | ---: |
| Upstream monolithic `export.py` | 1703.2 | 32.4 s |
| This bundle, decoder on default specialization | 1669.3 | 31.7 s |
| **This bundle, decoder pinned to CPU** | **4.8** | **0.09 s** |

Encoder runs once at 89.8 ms; mel front-end 11.0 ms.

## Export

`export_kv.py` imports `coreai_models`, which is not on PyPI, so — unlike its
siblings — it is **not** a self-contained `uv run` script. Run it from the repo
environment:

```sh
uv run python models/whisper/export_kv.py --dtype float16 --overwrite
```

| Flag | Description | Default |
| --- | --- | --- |
| `--model` | `openai/whisper-large-v3-turbo` or `openai/whisper-large-v3` | turbo |
| `--output-dir` | Output directory for the bundle | `<repo-root>/exports/` |
| `--dtype` | `float16`, `float32` — **no bfloat16** (see Limits) | `float16` |
| `--overwrite` | Replace an existing bundle | — |

## Verify

```sh
uv run python models/whisper/verify_kv.py --wav sample.wav
```

Drives the Core AI graphs exactly the way the Swift loop drives them and checks:

1. **HuggingFace token parity** — greedy tokens must match a reference fed the
   identical sequence. Currently exact, with first-step logits at 31.6 dB PSNR
   (fp16 vs fp32 reference).
2. **Uninitialized-cache safety** — reruns with the KV state pre-filled with `NaN`.
   Identical tokens prove the graph never reads a slot it did not write. This is
   not hypothetical: the runtime really does hand the decoder uninitialized state.
3. **Timing** — per-step decode latency.

## Run

```sh
swift build --package-path swift -c release --product speech-recognizer
.build/out/Products/Release/speech-recognizer \
  --model exports/whisper-large-v3-turbo-kv_float16 \
  --audio-path sample.wav --verbose
```

Audio longer than 30 s is decoded as several overlapping windows; the decode line
reports how many. `--window-overlap <seconds>` tunes the overlap (default 2, `0`
disables both the overlap and the seam de-duplication).

## Bundle layout

```
whisper-large-v3-turbo-kv_float16/
├── metadata.json           metadata_version 0.2, kind speech_recognizer,
│                           config.architecture "whisper"
├── encoder.aimodel         filename is hardcoded in SpeechRecognitionBundle;
├── decoder.aimodel         the `assets` map is ignored on the Whisper path
├── generation_config.json  forced prefix, eos, max_new_tokens, tokenizer_name
└── tokenizer.json + tokenizer_config.json + vocab.json + merges.txt + …
```

`metadata.json` is required in practice: without it the CLI routes the directory to
its legacy single-asset path and fails. The tokenizer files must be shipped as a set
— swift-transformers needs `tokenizer.json` *and* `tokenizer_config.json`, and if the
local load throws it does **not** fall back to the HuggingFace cache.

`generation_config.json` is written explicitly rather than copied from HuggingFace:
HF's file carries `max_length`, not `max_new_tokens`, and the Swift parser reads only
the latter — a verbatim copy would silently cap every transcript at 50 tokens.

## Graph contracts

```
encoder.aimodel — entrypoint "main"
  in   input_features         [1, 128, 3000]    f16
  out  encoder_hidden_states   [1, 1500, 10240]  f16

decoder.aimodel — entrypoint "main"
  in     input_ids             [1, 1]            int32
         position_ids          [1, S] dynamic    int32   arange(0..pos)
         encoder_hidden_states [1, 1500, 10240]  f16
  state  keyCache              [4, 1, 20, 448, 64] f16
         valueCache            [4, 1, 20, 448, 64] f16
  out    logits                [1, 1, 51866]     f16
```

State names are `keyCache`/`valueCache` from `coreai_models._constants`, the same
convention the LLM exports use, because `WhisperDecoder` looks them up by literal
string. `position_ids` carries the populated cache length in its *shape*
(`offset = S - query_len`), matching the macOS LLM convention in
`models/macos/qwen3.py`.

## Deviation: `encoder_hidden_states` carries packed cross K/V

**The tensor named `encoder_hidden_states` does not contain hidden states.** It holds
the per-layer cross-attention K and V *already projected*, packed along the channel
axis:

```
[1, 1500, n_decoder_layers * 2 * d_model]     # 10240 for turbo
layer i:  K at channel offset i*2*d_model
          V at channel offset i*2*d_model + d_model
```

The Swift loop passes this tensor as a plain input on every step and has no
cross-attention cache, so a literal export would reproject all 1500 encoder frames
per token:

| | MACs per decode step |
| --- | ---: |
| Cross-attention K/V projection over 1500 frames | 19.66 G |
| Everything else in the step (self-attn + FFN, 4 layers) | 0.079 G |

That is a **250× ratio** — a faithful implementation would be slower than the
monolithic export it replaces. Precomputing in the encoder is what makes the bundle
worth having.

The Swift side needs no change: it treats the tensor opaquely, and the only shape it
reads is axis 1 (1500) for `validEncoderFrames`, which `WhisperDecoder` then ignores.
The `--parity-test` harness never loads a Whisper model, so nothing else observes it.

**Consequence:** this encoder is not interchangeable with one that emits true hidden
states, and a future upstream Whisper encoder will not be a drop-in for it. The
`config.cross_kv_packed: true` flag in `metadata.json` marks the bundle.

Rejected alternative: extra `crossKeyCache`/`crossValueCache` Core AI states. The
Swift binds only `keyCache` and `valueCache`, so a graph declaring more states would
fail at `run(states:)`.

## Two runtime behaviours the export is built around

**Uninitialized state.** The runtime allocates the KV cache with
`NDArray(descriptor:)` and never zero-fills it. Safety comes solely from
`KVCache.update_and_fetch` returning the populated prefix `[0, seq_len)` — every slot
it returns was written by an earlier step. Masking a full-length cache instead would
be unsafe: uninitialized storage can hold `NaN`, and no additive mask suppresses that
(`NaN + -inf = NaN`). `verify_kv.py` asserts this property directly.

**The forced prefix's last token is fed twice.** The Swift primes the prefix at
positions 0–3, then begins greedy decoding by re-feeding `tokens.last!` at position 4.
Cache slots 3 and 4 both hold `<|notimestamps|>`, and the last priming step's logits
are discarded. This is not fixable from the export side, so `verify_kv.py` reproduces
it rather than correcting it. It is visible in output: this bundle transcribes
"the quick brown fox…" where the monolithic export gives "The quick brown fox…".
Both match their own reference; the contexts genuinely differ.

## Neural Engine notes

Two ANE-specific issues surfaced, one fixed here and one worked around in Swift.

**Fixed in the export.** HuggingFace adds a rank-2 `[1500, 1280]` positional table
directly onto the rank-3 `[1, 1500, 1280]` convolution output
(`modeling_whisper.py:685`). That rank-promoting broadcast makes the ANE lowering
mis-infer the add as `[1, 1500, 1500, 1280]` and the encoder fails to compile.
`_BatchedPositions` emits the table at rank 3 so the add stays plainly elementwise.

**Worked around in Swift.** `position_ids` grows by one element per token, so every
step presents a shape the accelerator backends have not specialized yet — about
300 ms of recompilation each, and the specialization cache does not survive the
process, so it is paid on every run. The CPU backend does not specialize per shape
and runs this 4-layer single-token graph in ~5 ms, so the default
`SpeechComputeUnits` policy pins the decoder to CPU. Parakeet's per-token graphs
turned out to want the same treatment for the same reason.

Those defaults were measured on Mac and are overridable per stage —
`SpeechComputeUnits(encoder:decoder:)`, or `--compute-encoder` / `--compute-decoder`
on the CLI. Re-measure before trusting them on iOS, where the GPU is much smaller
relative to the Neural Engine. See [`FORK.md`](../../FORK.md).

## Limits

- **30 s per encoder pass.** `MelConfig.whisper` pins 3000 mel frames, so the encoder
  sees exactly 480 000 samples per run — that is Whisper's architecture, not a runtime
  choice. Longer audio is handled by the runtime, which splits it into overlapping
  windows and stitches the transcripts; see [`FORK.md`](../../FORK.md). Nothing about
  it is visible to the export.
- **`whisper-large-v3`** has 32 decoder layers, so the packed encoder output is
  `[1, 1500, 81920]` ≈ 245 MB in fp16, passed as an input on every step. It exports,
  but turbo is the intended target.
- **No bfloat16.** The runtime fills `input_features` and reads `logits` through
  helpers that accept only float16 and float32.
- Batched multi-window decoding is not exported; the Swift loads only `main`.

## Differences from `export.py`

| | `export.py` | `export_kv.py` |
| --- | --- | --- |
| Assets | one `.aimodel` | bundle: encoder + decoder + sidecars |
| Encoder | recomputed every token | run once |
| Decoder state | none | `keyCache` / `valueCache` |
| Decoder input | full `decoder_input_ids` | one token + `position_ids` |
| Runtime path | legacy monolithic | `SpeechRecognitionModel` bundle path |
| ms/step | 1703 | 4.8 |

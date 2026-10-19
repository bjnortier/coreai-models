# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Verify the exported Whisper KV-cache bundle against HuggingFace.

Drives the Core AI graphs exactly the way ``CoreAISpeech``'s ``WhisperDecoder``
drives them, so a contract mismatch shows up here rather than in the Swift CLI:

1. **Token parity** — greedy tokens must match a HuggingFace reference that is
   fed the identical sequence, including the runtime's duplicated prefix token
   (see ``_swift_fed_sequence``).
2. **Uninitialized-cache safety** — the runtime hands the decoder an
   *uninitialized* state buffer. Re-running with the cache pre-filled with NaN
   must produce identical tokens; anything else means the graph reads a slot it
   never wrote, which would be a live bug under the real runtime.
3. **Timing** — per-step decode latency.
4. **Word alignment** — the decoder's ``alignment_logits`` against HuggingFace's
   eager cross-attention on the same heads, and the word times dynamic time
   warping derives from each. The alignment is the one ``CoreAISpeech``'s
   ``WhisperAlignment`` implements: openai-whisper's ``find_alignment`` (softmax
   over the valid frames, per-frame normalization across tokens, a width-7
   median filter, the mean over heads, DTW), using HF's own median filter and DTW.

    uv run python models/whisper/verify_kv.py --bundle exports/<name> --wav <file.wav>
"""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import string

import numpy as np
import torch
import transformers
from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions
from transformers.models.whisper.generation_whisper import (
    _dynamic_time_warping,
    _median_filter,
)

# Encoder frames per second: 3000 mel frames per 30 s, halved by the conv stride.
FRAMES_PER_SECOND = 50
MEDIAN_FILTER_WIDTH = 7
# openai-whisper's defaults for `prepend_punctuations` / `append_punctuations`.
PREPEND_PUNCTUATION = "\"'“¿([{-"
APPEND_PUNCTUATION = "\"'.。,，!！?？:：”)]}、"
SENTENCE_END = ".。!！?？"

def _load_pcm(wav_path: str) -> np.ndarray:
    with wave.open(wav_path, "rb") as wf:
        if (
            wf.getnchannels() != 1
            or wf.getframerate() != 16000
            or wf.getsampwidth() != 2
        ):
            raise ValueError(
                f"{wav_path}: expected 16 kHz mono 16-bit, got {wf.getframerate()} Hz "
                f"{wf.getnchannels()}ch {8 * wf.getsampwidth()}-bit"
            )
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _specialization(name: str):
    if name == "default":
        return None
    kind = {"gpu": ComputeUnitKind.gpu(), "cpu": ComputeUnitKind.cpu()}[name]
    return SpecializationOptions.from_preferred_compute_unit_kind(kind)


async def _load(path: Path, compute_unit: str, auto_order: list[str] | None = None):
    """Load a graph, reporting which compute unit actually worked.

    ``auto`` mirrors what ``SpeechRecognitionBundle.loadWhisper`` does: the
    encoder takes bare default specialization (falling back, because it does not
    fit in Neural Engine program memory), while the decoder is pinned to CPU —
    its ``position_ids`` grows every step, and the accelerator backends
    re-specialize on each new shape at ~300 ms a time.
    """
    order = (
        (auto_order or ["default", "gpu", "cpu"])
        if compute_unit == "auto"
        else [compute_unit]
    )
    last: Exception | None = None
    for name in order:
        try:
            model = await AIModel.load(str(path), _specialization(name))
            return model.load_function("main"), name
        except Exception as exc:  # noqa: BLE001 - report and try the next unit
            last = exc
            if compute_unit != "auto":
                raise
    raise RuntimeError(f"{path.name}: no compute unit could load this graph") from last


def _swift_fed_sequence(prefix: list[int], generated: list[int]) -> list[int]:
    """The token sequence the runtime has actually pushed through the cache.

    ``SpeechDecoder.swift`` primes the forced prefix at positions 0..n-1 and
    then starts greedy decoding by re-feeding ``tokens.last!`` — so the last
    prefix token is written to two consecutive cache slots and the final
    priming step's logits are discarded. Reproduced here rather than corrected,
    because the reference has to match what the runtime really does.
    """
    return prefix + [prefix[-1]] + generated


async def decode_coreai(
    enc_fn,
    dec_fn,
    features: np.ndarray,
    prefix: list[int],
    max_new: int,
    eos: int,
    cache_fill: float,
    static: bool = False,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray | None]:
    """Greedy decode, also returning each generated step's ``alignment_logits``
    (``[steps, heads, 1500]``), or ``None`` for a decoder exported without them."""
    enc_out = (await enc_fn({"input_features": NDArray(features)}))[
        "encoder_hidden_states"
    ]

    desc = dec_fn.desc
    kc_desc, vc_desc = (
        desc.state_descriptor("keyCache"),
        desc.state_descriptor("valueCache"),
    )
    make = lambda d: NDArray(np.full(list(d.shape), cache_fill, dtype=np.float16))  # noqa: E731
    state = {"keyCache": make(kc_desc), "valueCache": make(vc_desc)}

    first_logits: np.ndarray | None = None

    alignment: list[np.ndarray] = []

    async def step(token: int, pos: int):
        positions = [pos] if static else np.arange(pos + 1)
        out = await dec_fn(
            {
                "input_ids": NDArray(np.array([[token]], dtype=np.int32)),
                "position_ids": NDArray(np.array(positions, dtype=np.int32)[None, :]),
                "encoder_hidden_states": enc_out,
            },
            state=state,
        )
        scores = out.get("alignment_logits") if hasattr(out, "get") else None
        return out["logits"].numpy(), None if scores is None else scores.numpy()

    for i, tok in enumerate(prefix):
        logits, _ = await step(tok, i)
        if i == len(prefix) - 1:
            first_logits = logits

    tokens = list(prefix)
    step_times: list[float] = []
    pos = len(prefix)
    while len(tokens) - len(prefix) < max_new:
        t0 = time.perf_counter()
        logits, scores = await step(tokens[-1], pos)
        step_times.append((time.perf_counter() - t0) * 1000)
        if scores is not None:
            alignment.append(scores.reshape(scores.shape[-2], scores.shape[-1]))
        nxt = int(logits.reshape(-1).argmax())
        tokens.append(nxt)
        pos += 1
        if nxt == eos:
            break
    return tokens, step_times, first_logits, np.stack(alignment) if alignment else None


def decode_hf(
    hf, features: torch.Tensor, prefix: list[int], max_new: int, eos: int
):
    """Greedy reference, fed exactly what the runtime feeds."""
    with torch.no_grad():
        enc = hf.model.encoder(features).last_hidden_state
        generated: list[int] = []
        first_logits = None
        while len(generated) < max_new:
            fed = _swift_fed_sequence(prefix, generated)
            dec = hf.model.decoder(
                input_ids=torch.tensor([fed], dtype=torch.long),
                encoder_hidden_states=enc,
            ).last_hidden_state
            logits = hf.proj_out(dec[:, -1])
            if first_logits is None:
                first_logits = logits.float().numpy()
            nxt = int(logits.argmax(-1))
            generated.append(nxt)
            if nxt == eos:
                break
    return prefix + generated, first_logits


def hf_alignment_weights(hf, features, prefix, tokens, heads, valid_frames):
    """HF's eager cross-attention for every generated step on the alignment heads,
    renormalized over the valid frames: ``[heads, steps, valid_frames]``.

    Renormalizing a softmax over a subset equals the softmax of the logits over that
    subset, which is what ``alignment_weights`` takes of the Core AI scores.
    """
    generated = tokens[len(prefix):]
    fed = _swift_fed_sequence(prefix, generated[:-1])
    with torch.no_grad():
        enc = hf.model.encoder(features).last_hidden_state
        out = hf.model.decoder(
            input_ids=torch.tensor([fed], dtype=torch.long),
            encoder_hidden_states=enc,
            output_attentions=True,
        )
    rows = slice(len(prefix), len(prefix) + len(generated))
    weights = torch.stack([out.cross_attentions[l][0, h, rows] for l, h in heads])
    weights = weights[..., :valid_frames].double()
    return (weights / weights.sum(-1, keepdim=True)).numpy()


def alignment_weights(scores: np.ndarray, valid_frames: int) -> np.ndarray:
    """``[steps, heads, 1500]`` scores -> ``[heads, steps, valid_frames]`` softmax."""
    x = scores[..., :valid_frames].astype(np.float64).transpose(1, 0, 2)
    x = np.exp(x - x.max(-1, keepdims=True))
    return x / x.sum(-1, keepdims=True)


def token_frames(weights: np.ndarray) -> np.ndarray:
    """The encoder frame each step's token starts at: openai-whisper's
    ``find_alignment`` on ``[heads, steps, frames]`` softmaxed weights."""
    w = torch.tensor(weights)
    std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
    w = _median_filter(((w - mean) / std)[None], MEDIAN_FILTER_WIDTH)[0]
    matrix = w.mean(0).numpy()
    text_indices, time_indices = _dynamic_time_warping(-matrix)
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    return time_indices[jumps]


def timed_words(
    tokenizer, generated: list[int], frames: np.ndarray, eot: int, end_frame: int,
    splits_on_spaces: bool = True,
):
    """Words with start/end seconds, as ``WhisperAlignment.words`` builds them: openai-whisper's
    ``split_to_word_tokens``, its long-word truncation, ``merge_punctuations`` and its clamp on a
    long first word, in that order."""
    groups: list[list] = []  # text, first step, step after the word
    pending: list[int] = []
    boundary = True
    for index, token in enumerate(generated):
        if token >= eot:  # eot, or a stray special token: never a word, always a boundary
            pending, boundary = [], True
            continue
        pending.append(token)
        piece = tokenizer.decode(pending)
        if "\ufffd" in piece:  # an incomplete UTF-8 sequence; wait for its next byte
            continue
        first, pending = index - len(pending) + 1, []
        opens = (
            boundary
            or not splits_on_spaces
            or piece.startswith(" ")
            or piece.strip() in string.punctuation
        )
        boundary = False
        if opens:
            groups.append([piece, first, index + 1])
        else:
            groups[-1][0] += piece
            groups[-1][2] = index + 1
    seconds = lambda step: (frames[step] if step < len(frames) else end_frame) / FRAMES_PER_SECOND  # noqa: E731
    words = [[text, seconds(a), seconds(b)] for text, a, b in groups]

    durations = [end - start for _, start, end in words if end > start]
    median = min(0.7, float(np.median(durations))) if durations else 0.0
    longest = median * 2
    if durations:
        for i in range(1, len(words)):
            if words[i][2] - words[i][1] > longest:
                if words[i][0] in SENTENCE_END:
                    words[i][2] = words[i][1] + longest
                elif words[i - 1][0] in SENTENCE_END:
                    words[i][1] = words[i][2] - longest

    i, j = len(words) - 2, len(words) - 1
    while i >= 0:
        if words[i][0].startswith(" ") and words[i][0].strip() in PREPEND_PUNCTUATION:
            words[j][0] = words[i][0] + words[j][0]
            words[i][0] = ""
        else:
            j = i
        i -= 1
    i, j = 0, 1
    while j < len(words):
        if not words[i][0].endswith(" ") and words[j][0] in APPEND_PUNCTUATION:
            words[i][0] += words[j][0]
            words[j][0] = ""
        else:
            i = j
        j += 1
    words = [word for word in words if word[0]]

    if words and words[0][2] > median * 4 and (
        words[0][2] - words[0][1] > longest
        or (len(words) > 1 and words[1][2] - words[0][1] > longest * 2)
    ):
        if len(words) > 1 and words[1][2] - words[1][1] > longest:
            boundary_time = max(words[1][2] / 2, words[1][2] - longest)
            words[0][2] = words[1][1] = boundary_time
        words[0][1] = max(0.0, words[0][2] - longest)
    return [(text.strip(), start, end) for text, start, end in words]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * np.log10(float(np.abs(a).max()) / np.sqrt(mse))


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an exported Whisper KV bundle."
    )
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--bundle", default=str(repo / "exports" / "whisper-large-v3-turbo-kv_float16")
    )
    parser.add_argument("--wav", required=True, help="16 kHz mono 16-bit WAV")
    parser.add_argument(
        "--compute-unit", choices=["auto", "default", "gpu", "cpu"], default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--skip-hf", action="store_true", help="skip the HuggingFace reference"
    )
    parser.add_argument(
        "--words-json", help="also write the Core AI word timestamps here, for the Swift to match"
    )
    args = parser.parse_args()

    bundle = Path(args.bundle)
    gen_cfg = json.loads((bundle / "generation_config.json").read_text())
    meta = json.loads((bundle / "metadata.json").read_text())
    prefix = [tok for _, tok in sorted(gen_cfg["forced_decoder_ids"])]
    max_new = args.max_new_tokens or gen_cfg["max_new_tokens"]
    eos = gen_cfg["eos_token_id"]
    # A `--static-positions` export takes the current position alone, not 0..pos.
    static = "-kv-static_" in meta["name"]
    model_name = gen_cfg["tokenizer_name"]

    processor = transformers.AutoProcessor.from_pretrained(model_name)
    pcm = _load_pcm(args.wav)
    features = processor.feature_extractor(
        pcm, sampling_rate=16000, return_tensors="np"
    )["input_features"].astype(np.float16)
    print(f"[INFO] {args.wav}: {len(pcm) / 16000:.1f}s -> features {features.shape}")

    enc_fn, enc_unit = await _load(bundle / "encoder.aimodel", args.compute_unit)
    dec_fn, dec_unit = await _load(
        bundle / "decoder.aimodel", args.compute_unit, auto_order=["cpu", "default"]
    )
    print(f"[INFO] encoder loaded on '{enc_unit}', decoder on '{dec_unit}'")
    if args.compute_unit == "auto" and enc_unit != "default":
        # The Swift loads the encoder with bare AIModel(contentsOf:); if that
        # needs a fallback here, the runtime is relying on one too.
        print(
            f"[WARN] encoder needed a '{enc_unit}' fallback — it did not load under\n"
            "       default specialization, which is what the Swift bundle loader uses."
        )

    t0 = time.perf_counter()
    tokens, step_times, coreai_logits, coreai_scores = await decode_coreai(
        enc_fn, dec_fn, features, prefix, max_new, eos, cache_fill=0.0, static=static
    )
    total = time.perf_counter() - t0
    tokenizer = processor.tokenizer
    text = tokenizer.decode(
        [t for t in tokens if t < eos], skip_special_tokens=True
    ).strip()
    print(f"\n=== Transcript ({len(step_times)} steps, {total:.2f}s total) ===")
    print(" ", repr(text))
    if step_times:
        print(
            f"  {np.mean(step_times):.2f} ms/step avg, "
            f"{np.min(step_times):.2f} min, {np.max(step_times):.2f} max"
        )

    ok = True

    print("\n=== Uninitialized-cache safety ===")
    nan_tokens, _, _, _ = await decode_coreai(
        enc_fn, dec_fn, features, prefix, max_new, eos, cache_fill=np.nan, static=static
    )
    nan_match = nan_tokens == tokens
    print(f"  NaN-filled cache produces identical tokens: {nan_match}")
    if not nan_match:
        print("  [FAIL] the graph reads cache slots it never wrote")
        ok = False

    if not args.skip_hf:
        print("\n=== HuggingFace parity ===")
        hf = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, dtype=torch.float32, use_safetensors=True
        )
        hf.eval()
        hf_tokens, hf_logits = decode_hf(
            hf, torch.tensor(features.astype(np.float32)), prefix, max_new, eos
        )
        match = hf_tokens == tokens
        print(f"  token sequences identical: {match}")
        if not match:
            n = min(len(tokens), len(hf_tokens))
            first = next((i for i in range(n) if tokens[i] != hf_tokens[i]), n)
            print(f"  first divergence at index {first}")
            print(
                "  coreai:",
                repr(tokenizer.decode([t for t in tokens if t < eos]).strip()),
            )
            print(
                "  hf    :",
                repr(tokenizer.decode([t for t in hf_tokens if t < eos]).strip()),
            )
            ok = False
        if coreai_logits is not None and hf_logits is not None:
            a, b = coreai_logits.astype(np.float32).ravel(), hf_logits.ravel()
            print(
                f"  prefix-step logits: max|diff| {np.abs(a - b).max():.4f}, "
                f"PSNR {_psnr(b, a):.1f} dB"
            )

    heads = meta["config"].get("alignment_heads")
    if coreai_scores is None or not heads:
        print("\n[WARN] decoder has no alignment_logits; re-export for word timestamps")
    else:
        print("\n=== Word alignment ===")
        # openai-whisper crops to `num_frames // 2` of the unpadded mel
        mel_frames = min(len(pcm) // processor.feature_extractor.hop_length, features.shape[-1])
        valid = max(1, mel_frames // 2)
        generated = tokens[len(prefix):]
        weights = alignment_weights(coreai_scores, valid)
        frames = token_frames(weights)
        words = timed_words(tokenizer, generated, frames, eos, valid)
        for text, start, end in words:
            print(f"  {start:6.2f} {end:6.2f}  {text}")
        if not args.skip_hf:
            hf_eager = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, dtype=torch.float32, use_safetensors=True, attn_implementation="eager"
            )
            hf_eager.eval()
            reference = hf_alignment_weights(
                hf_eager, torch.tensor(features.astype(np.float32)), prefix, tokens, heads, valid
            )
            ref_frames = token_frames(reference)
            ref_words = timed_words(tokenizer, generated, ref_frames, eos, valid)
            moved = np.abs(frames - ref_frames)
            print(
                f"  alignment weights PSNR {_psnr(reference, weights):.1f} dB | token start frames: "
                f"{int((moved == 0).sum())}/{len(moved)} identical, max shift {int(moved.max())} "
                f"({moved.max() / FRAMES_PER_SECOND * 1000:.0f} ms)"
            )
            word_shift = max(
                (max(abs(a[1] - b[1]), abs(a[2] - b[2])) for a, b in zip(words, ref_words)),
                default=0.0,
            )
            print(f"  words: {len(words)} Core AI, {len(ref_words)} HF | max word time shift {word_shift * 1000:.0f} ms")
            if [w[0] for w in words] != [w[0] for w in ref_words]:
                print("  [FAIL] word split differs")
                ok = False
        if args.words_json:
            Path(args.words_json).write_text(
                json.dumps([{"text": t, "start": a, "end": b} for t, a, b in words], indent=1) + "\n"
            )

    print(f"\n[{'OK' if ok else 'FAIL'}] {meta['name']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

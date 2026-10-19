# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "transformers>=5.18.0,<5.20",
#     "librosa",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""End-to-end check of an exported Nemotron-3-Diarization bundle against the HF model.

For each mode, runs a whole session twice:

* **reference** — the HF forward in float32, chunked exactly as the HF docs stream it;
* **Core AI** — the exported step, driven by HF's own `Nemotron3DiarizationSpeakerCache` on the host, so the
  only difference from the reference is the step graph (and its precision).

Precision differences compound across a session: the cache policy thresholds the step's probabilities and keeps
the top-scoring frames, so a small error can keep a different frame and steer every later step. A single-step
PSNR cannot show that; this does.

``--fixture-dir`` also writes a few steps' inputs and float32 reference outputs for an on-device parity check.

Usage::

    uv run verify.py                                    # HF's diarization example, every mode in the bundle
    uv run verify.py --audio meeting.wav --modes low_latency
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from coreai.runtime import AIModel, NDArray
from transformers.audio_utils import load_audio
from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationSpeakerCache,
)

from export import build_step, load_model

EXAMPLE_AUDIO = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3"
)


def psnr(reference: np.ndarray, actual: np.ndarray) -> float:
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    mse = np.mean((reference - actual) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(np.max(np.abs(reference)) ** 2 / mse)


def streaming_inputs(processor, audio: np.ndarray, sampling_rate: int):
    """The processor outputs of each chunk, as the HF docs stream them."""
    yield processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
    )
    mel_frame_idx = processor.num_mel_frames_per_step
    start_idx = processor.audio_chunk_start(mel_frame_idx)
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        yield processor(audio[start_idx:end_idx], sampling_rate=sampling_rate, is_streaming=True,
                        is_first_audio_chunk=False)
        mel_frame_idx += processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)
    yield processor(audio[start_idx:], sampling_rate=sampling_rate, is_streaming=True, is_first_audio_chunk=False,
                    is_last_audio_chunk=True)


def offline_chunks(inputs, geometry: dict, subsampling: int):
    """(chunk_features, embed_mask, num_lookahead_frames) of each step of the HF offline forward."""
    features = inputs["input_features"]
    embed_mask = inputs["attention_mask"][:, ::subsampling].bool()
    num_embeds = math.ceil(features.shape[1] / subsampling)
    chunk, lookahead = geometry["chunk_frames"], geometry["lookahead_frames"]
    for start in range(0, num_embeds, chunk):
        end = min(start + chunk, num_embeds)
        stop = min(end + lookahead, num_embeds)
        yield features[:, start * subsampling : stop * subsampling], embed_mask[:, start:stop], stop - end


def reference_session(model, processor, audio, sampling_rate, mode: str) -> torch.Tensor:
    with torch.inference_mode():
        if mode == "offline":
            return model(**processor(audio, sampling_rate=sampling_rate)).logits
        processor.set_streaming_mode(mode)
        cache, logits = None, []
        for inputs in streaming_inputs(processor, audio, sampling_rate):
            outputs = model(**inputs, speaker_cache=cache)
            logits.append(outputs.logits)
            cache = outputs.speaker_cache
        return torch.cat(logits, dim=1)


class StepRunner:
    """Runs one exported mode: pads a step to the static window, strips the padding from its outputs."""

    def __init__(self, function, geometry: dict, config, np_dtype):
        self.function, self.geometry, self.np_dtype = function, geometry, np_dtype
        self.subsampling = config.audio_config.subsampling_factor
        self.hidden_size = config.audio_config.hidden_size
        self.num_cache = geometry["cache_frames"]
        self.num_window = geometry["chunk_frames"] + geometry["lookahead_frames"]
        self.latencies_ms = []

    def pad(self, cached: torch.Tensor, features: torch.Tensor, embed_mask: torch.Tensor) -> dict[str, np.ndarray]:
        num_cached, num_embeds = cached.shape[1], embed_mask.shape[1]
        cached_embeds = np.zeros((1, self.num_cache, self.hidden_size), np.float32)
        cached_embeds[:, self.num_cache - num_cached :] = cached.numpy()
        chunk_features = np.zeros((1, self.num_window * self.subsampling, features.shape[2]), np.float32)
        chunk_features[:, : features.shape[1]] = features.numpy()
        key_mask = np.zeros((1, self.num_cache + self.num_window), np.float32)
        key_mask[:, self.num_cache - num_cached : self.num_cache] = 1.0
        key_mask[:, self.num_cache : self.num_cache + num_embeds] = embed_mask.float().numpy()
        return {"cached_embeds": cached_embeds, "chunk_features": chunk_features, "key_mask": key_mask}

    async def __call__(self, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        arrays = {name: NDArray(value.astype(self.np_dtype)) for name, value in inputs.items()}
        start = time.perf_counter()
        outputs = await self.function(arrays)
        self.latencies_ms.append((time.perf_counter() - start) * 1000)
        return outputs["logits"].numpy().astype(np.float32), outputs["chunk_embeds"].numpy().astype(np.float32)


async def coreai_session(runner: StepRunner, model, processor, audio, sampling_rate, mode: str, steps: list):
    """The session of `reference_session`, with each encoder step replaced by the Core AI graph.

    `steps` collects every step's padded inputs, for the fixture.
    """
    config, g = model.config, runner.geometry
    subsampling = runner.subsampling
    if mode == "offline":
        processed = processor(audio, sampling_rate=sampling_rate)
        chunks = offline_chunks(processed, g, subsampling)
        cache = Nemotron3DiarizationSpeakerCache(
            config.streaming_config,
            fifo_length=g["fifo_length"],
            speaker_cache_update_period=g["speaker_cache_update_period"],
        )
        total_frames = processed["input_features"].shape[1]
    else:
        processor.set_streaming_mode(mode)
        chunks = (
            (
                inputs["input_features"],
                inputs["attention_mask"][:, ::subsampling].bool(),
                inputs.get("num_lookahead_frames", 0),
            )
            for inputs in streaming_inputs(processor, audio, sampling_rate)
        )
        cache = Nemotron3DiarizationSpeakerCache(config.streaming_config)
        total_frames = None
    cache.lazy_initialization(torch.zeros(1, 1, runner.hidden_size))

    logits = []
    for features, embed_mask, num_lookahead in chunks:
        num_embeds = embed_mask.shape[1]
        num_chunk_frames = num_embeds - num_lookahead
        cached = cache.get_embeds(None)
        num_cached = cached.shape[1]

        inputs = runner.pad(cached, features, embed_mask)
        step_logits, chunk_embeds = await runner(inputs)
        steps.append({"inputs": inputs, "num_cached": num_cached, "num_embeds": num_embeds})

        # drop the left padding of the cache and the right padding of the chunk
        offset = runner.num_cache - num_cached
        step_logits = torch.from_numpy(step_logits[:, offset * subsampling : (runner.num_cache + num_embeds) * subsampling])
        chunk_input_embeds = torch.cat([cached, torch.from_numpy(chunk_embeds[:, :num_embeds])], dim=1)
        step_mask = torch.cat([embed_mask.new_ones(1, num_cached), embed_mask], dim=1)
        cache.update(chunk_input_embeds, step_logits, model.silence_embeds.detach(), num_chunk_frames, mask=step_mask)

        chunk_logits = step_logits[:, num_cached * subsampling : (num_cached + num_chunk_frames) * subsampling]
        logits.append(chunk_logits if mode == "offline" else chunk_logits[:, : features.shape[1]])
    logits = torch.cat(logits, dim=1)
    return logits[:, :total_frames] if total_frames is not None else logits


def write_fixture(fixture_dir: Path, mode: str, step_module, steps: list, picks: list[int]):
    """Inputs and float32 PyTorch outputs of a few steps, as raw little-endian float32, for an on-device parity check."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in picks:
        step = steps[index]
        tensors = {name: torch.from_numpy(value) for name, value in step["inputs"].items()}
        with torch.no_grad():
            logits, chunk_embeds = step_module(**tensors)
        files = {}
        for name, value in {**step["inputs"], "logits": logits.numpy(), "chunk_embeds": chunk_embeds.numpy()}.items():
            path = f"{mode}_step{index}_{name}.f32"
            value.astype("<f4").tofile(fixture_dir / path)
            files[name] = {"file": path, "shape": list(value.shape)}
        entries.append({"step": index, "num_cached": step["num_cached"], "num_embeds": step["num_embeds"],
                        "tensors": files})
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[mode] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parents[2] / "exports" /
                                                "nemotron-3-diarization_float16"))
    parser.add_argument("--audio", default=EXAMPLE_AUDIO, help="audio file or URL (default: HF diarization example)")
    parser.add_argument("--modes", nargs="+", help="modes to check (default: every mode in the bundle)")
    parser.add_argument("--fixture-dir", help="also write on-device fixtures here")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    metadata = json.loads((bundle / "metadata.json").read_text())
    model = load_model(metadata["model"])
    processor = transformers.AutoProcessor.from_pretrained(metadata["model"])
    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(args.audio, sampling_rate=sampling_rate)
    print(f"[INFO] {args.audio}: {len(audio) / sampling_rate:.1f} s")

    np_dtype = np.float16 if metadata["dtype"] == "float16" else np.float32
    aimodels = {}
    for mode in args.modes or list(metadata["modes"]):
        g = metadata["modes"][mode]
        if g["asset"] not in aimodels:
            aimodels[g["asset"]] = await AIModel.load(str(bundle / g["asset"]))
        runner = StepRunner(aimodels[g["asset"]].load_function(mode), g, model.config, np_dtype)

        reference = reference_session(model, processor, audio, sampling_rate, mode)
        steps = []
        actual = await coreai_session(runner, model, processor, audio, sampling_rate, mode, steps)
        if actual.shape != reference.shape:
            raise AssertionError(f"{mode}: Core AI logits {tuple(actual.shape)} != reference {tuple(reference.shape)}")

        excluded = ""
        if mode == "offline":
            # The offline processor masks the recording's final mel frame when its window is all padding. If that
            # frame opens an encoder frame, HF masks it from attention but still feeds its hidden state to the
            # upsampler conv; the step zeroes every masked frame instead. Only that encoder frame and the one before
            # it can differ, so leave the last two encoder frames out of the comparison.
            tail = 2 * model.config.audio_config.subsampling_factor
            reference, actual = reference[:, :-tail], actual[:, :-tail]
            excluded = f" (last {tail} frames excluded)"

        ref_active, act_active = reference.sigmoid() > 0.5, actual.sigmoid() > 0.5
        disagreement = (ref_active != act_active).any(-1).float().mean().item()
        speakers = lambda active: int(active[0].any(0).sum())
        latencies = sorted(runner.latencies_ms[1:]) or runner.latencies_ms
        print(
            f"[{mode}] {len(steps)} steps | logits PSNR{excluded} {psnr(reference.numpy(), actual.numpy()):.1f} dB | "
            f"frames with any speaker decision differing {disagreement * 100:.2f}% | speakers "
            f"{speakers(ref_active)} reference, {speakers(act_active)} Core AI | step median "
            f"{np.median(latencies):.1f} ms"
        )

        if args.fixture_dir:
            # the first step (cache empty, mostly padding) and the last full step (cache and FIFO at capacity)
            picks = sorted({0, max(len(steps) - 2, 0)})
            write_fixture(Path(args.fixture_dir), mode, build_step(model, g, torch.float32), steps, picks)
    if args.fixture_dir:
        print(f"[INFO] Fixtures written to {args.fixture_dir}.")


if __name__ == "__main__":
    asyncio.run(main())

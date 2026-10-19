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
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Export Nemotron-3-Diarization as static-shape streaming steps.

The HF forward (`Nemotron3DiarizationForAudioFrameClassification`) is a Python loop: every chunk re-encodes
`[speaker cache | FIFO | chunk | look-ahead]` from scratch, then `Nemotron3DiarizationSpeakerCache.update` moves
frames from the FIFO into the speaker cache and, when the cache overflows, compresses it with a score-and-top-k
policy. torch.export cannot capture that loop, and the cache policy is a few hundred scalar ops on at most ~500
frames, so it stays on the host. What is exported is one encoder step per latency mode:

    cached_embeds  [1, C, 512]    speaker cache + FIFO frames, LEFT-padded with zeros to C
    chunk_features [1, K*8, 128]  log-mel frames of the chunk and its look-ahead, right-padded to K*8
    key_mask       [1, C+K]       1 for a valid frame, 0 for padding
 -> logits         [1, (C+K)*8, 8]  speaker logits at 10 ms
    chunk_embeds   [1, K, 512]      the chunk's encoder inputs, for the host to push into the FIFO

Left padding is what makes one static shape exact. The HF step concatenates the valid frames and numbers them
0..n-1 for RoPE; padding the cache on the left shifts every valid frame by the same offset, and RoPE attention
scores depend only on position differences, so the valid frames see exactly what they see in HF. Padding
between the cache and the chunk would change those differences.

The padding frames are also zeroed before the upsampler: its kernel-3 Conv1d reads one frame either side, and the
HF step sees its own zero padding there, not the transformer output of a padding frame.

Usage::

    uv run export.py                                  # float16, all four modes in one asset
    uv run export.py --modes low_latency offline
    uv run export.py --split                          # one asset per mode, for iOS
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from huggingface_hub import hf_hub_download

DEFAULT_MODEL = "nvidia/Nemotron-3-Diarization"

# Additive attention bias for padding keys. exp(-1e4) underflows to 0 in float16 as in float32, and -1e4 stays
# well inside float16 range, unlike -inf, which turns a fully masked row into NaN.
MASK_BIAS = -1e4

ASSET_NAME = "diarization_step.aimodel"


class DiarizationStep(torch.nn.Module):
    """One streaming step of the HF model, at a static window of `num_cache_frames + num_chunk_frames` frames."""

    def __init__(
        self,
        model: "transformers.Nemotron3DiarizationForAudioFrameClassification",
        num_cache_frames: int,
        num_chunk_frames: int,
    ):
        super().__init__()
        audio_tower = model.model.audio_tower
        self.embedder = audio_tower.embedder
        self.input_layer_norm = audio_tower.input_layer_norm
        self.layers = audio_tower.layers
        self.layer_norm = audio_tower.layer_norm
        self.proj = model.model.proj
        self.upsampler = model.model.upsampler
        self.classifier = model.classifier

        num_frames = num_cache_frames + num_chunk_frames
        position_ids = torch.arange(num_frames)[None, :]
        cos, sin = audio_tower.rotary_emb(torch.zeros(1), position_ids)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        cached_embeds: torch.Tensor,
        chunk_features: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_embeds = self.embedder(chunk_features)
        hidden_states = torch.cat([cached_embeds, chunk_embeds], dim=1)
        dtype = hidden_states.dtype

        bias = ((1.0 - key_mask) * MASK_BIAS)[:, None, None, :].to(dtype)
        position_embeddings = (self.cos.to(dtype), self.sin.to(dtype))

        hidden_states = self.input_layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=bias, position_embeddings=position_embeddings)
        hidden_states = self.proj(self.layer_norm(hidden_states))
        hidden_states = hidden_states * key_mask[..., None].to(dtype)
        logits = self.classifier(self.upsampler(hidden_states))
        return logits, chunk_embeds


def load_model(
    model_name: str, attn_implementation: str = "eager"
) -> "transformers.Nemotron3DiarizationForAudioFrameClassification":
    """The HF model in float32. Either attention takes our additive bias as its `attention_mask` unchanged.

    `eager` is HF's reference: separate matmuls and a float32 softmax. `sdpa` is one
    `scaled_dot_product_attention`, which Core AI keeps fused. Export with `sdpa`: on an iPhone 15 Pro the eager
    graph allocated ~600-900 MB of intermediates per step (31 layers of float32 score matrices), enough to have
    the app killed.
    """
    model = transformers.AutoModelForAudioFrameClassification.from_pretrained(
        model_name, attn_implementation=attn_implementation, dtype=torch.float32
    )
    return model.eval()


def mode_geometry(model_name: str, config) -> dict[str, dict]:
    """Static window of every mode, in encoder frames, plus the cache sizes the host runs it with.

    Streaming modes come from `processor_config.json`, the processor being their single source of truth in HF.
    `offline` is what the HF forward does when given a whole recording: 30.4 s chunks with a 40-frame FIFO.
    """
    with open(hf_hub_download(model_name, "processor_config.json")) as f:
        streaming_modes = json.load(f)["streaming_modes"]
    streaming = config.streaming_config

    geometry = {}
    for name, (chunk, lookahead) in streaming_modes.items():
        geometry[name] = {
            "chunk_frames": chunk,
            "lookahead_frames": lookahead,
            "fifo_length": streaming.fifo_length,
            "speaker_cache_update_period": streaming.speaker_cache_update_period,
        }
    geometry["offline"] = {
        "chunk_frames": config.chunk_length,
        "lookahead_frames": config.chunk_right_context,
        "fifo_length": config.fifo_length,
        "speaker_cache_update_period": config.speaker_cache_update_period,
    }
    for g in geometry.values():
        # before a step the cache holds at most `speaker_cache_length` frames and the FIFO at most `fifo_length`
        g["cache_frames"] = streaming.speaker_cache_length + g["fifo_length"]
        g["window_frames"] = g["cache_frames"] + g["chunk_frames"] + g["lookahead_frames"]
    return geometry


def step_inputs(geometry: dict, config, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    audio = config.audio_config
    num_chunk_frames = geometry["chunk_frames"] + geometry["lookahead_frames"]
    return {
        "cached_embeds": torch.zeros(1, geometry["cache_frames"], audio.hidden_size, dtype=dtype),
        "chunk_features": torch.zeros(
            1, num_chunk_frames * audio.subsampling_factor, audio.num_mel_bins, dtype=dtype
        ),
        "key_mask": torch.ones(1, geometry["window_frames"], dtype=dtype),
    }


def build_step(model, geometry: dict, dtype: torch.dtype) -> DiarizationStep:
    step = DiarizationStep(model, geometry["cache_frames"], geometry["chunk_frames"] + geometry["lookahead_frames"])
    return step.to(dtype).eval()


def _metadata(model_name: str, modes: list[str]) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "NVIDIA"
    metadata.license = "OpenMDW-1.1"
    metadata.model_description = (
        f"Nemotron-3-Diarization streaming step, one function per mode ({', '.join(modes)}): a 31-layer RoPE Transformer encoder over "
        f"[speaker cache | FIFO | chunk | look-ahead] with a Sortformer head giving 8 arrival-ordered speaker "
        f"activities at 10 ms. Source: https://huggingface.co/{model_name}"
    )
    metadata.creation_date = int(time.time())
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["low_latency", "very_low_latency", "ultra_low_latency", "offline"],
        help="latency modes to export, one function each",
    )
    parser.add_argument(
        "--attention",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="attention kernel: fused SDPA (default) or HF's eager matmul + float32 softmax",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="one asset per mode instead of one asset with a function per mode (needed for iOS, see README)",
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "exports"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    suffix = "_split" if args.split else ""
    bundle_dir = Path(args.output_dir) / f"{Path(args.model).name.lower()}_{args.dtype}{suffix}"
    if bundle_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{bundle_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    model = load_model(args.model, args.attention)
    config = model.config
    geometry = mode_geometry(args.model, config)
    unknown = sorted(set(args.modes) - set(geometry))
    if unknown:
        raise ValueError(f"unknown modes {unknown}, expected some of {sorted(geometry)}")

    # Default: one asset, one function per mode, sharing a single copy of the weights (191 MB at float16).
    # --split: one asset per mode. Compiled for iOS, the shared asset is 740 MB (compilation duplicates the weights
    # per function) and fails to specialize on an iPhone 15 Pro; four single-function assets compile to 190 MB each
    # and load.
    groups = [[mode] for mode in args.modes] if args.split else [args.modes]
    assets = {}
    for group in groups:
        converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
        for mode in group:
            g = geometry[mode]
            print(f"[INFO] Exporting {mode}: window {g['window_frames']} frames ({g['cache_frames']} cache + "
                  f"{g['chunk_frames']} chunk + {g['lookahead_frames']} look-ahead)...")
            step = build_step(model, g, dtype)
            with torch.no_grad():
                exported = torch.export.export(step, args=(), kwargs=step_inputs(g, config, dtype))
            converter.add_exported_program(
                exported.run_decompositions(get_decomp_table()),
                input_names=["cached_embeds", "chunk_features", "key_mask"],
                output_names=["logits", "chunk_embeds"],
                entrypoint_name=mode,
            )
        program = converter.to_coreai()
        program.optimize()
        asset = f"step_{group[0]}.aimodel" if args.split else ASSET_NAME
        program.save_asset(bundle_dir / asset, _metadata(args.model, group))
        print(f"[INFO] Saved {bundle_dir / asset}.")
        assets.update({mode: asset for mode in group})

    # the learned embedding the host writes into the reserved silence slots of a compressed speaker cache
    silence = model.silence_embeds.detach().to(torch.float32).numpy()
    silence.astype("<f4").tofile(bundle_dir / "silence_embeds.f32")

    streaming = config.streaming_config
    metadata = {
        "model": args.model,
        "dtype": args.dtype,
        "inputs": ["cached_embeds", "chunk_features", "key_mask"],
        "outputs": ["logits", "chunk_embeds"],
        "features": {
            "sample_rate": 16000,
            "num_mel_bins": config.audio_config.num_mel_bins,
            "hop_length": 160,
            "win_length": 400,
            "n_fft": 512,
            "preemphasis": 0.97,
            "normalize": None,
        },
        "subsampling_factor": config.audio_config.subsampling_factor,
        "hidden_size": config.audio_config.hidden_size,
        "num_speakers": streaming.num_speakers,
        "silence_embeds": "silence_embeds.f32",
        "speaker_cache": {
            "speaker_cache_length": streaming.speaker_cache_length,
            "speaker_cache_silence_frames_per_speaker": streaming.speaker_cache_silence_frames_per_speaker,
            "prediction_score_threshold": streaming.prediction_score_threshold,
            "latest_frames_score_boost": streaming.latest_frames_score_boost,
            "min_positive_scores_rate": streaming.min_positive_scores_rate,
            "strong_boost_rate": streaming.strong_boost_rate,
            "weak_boost_rate": streaming.weak_boost_rate,
        },
        # each mode is the function named after it, in the asset named here
        "modes": {mode: {**geometry[mode], "asset": assets[mode]} for mode in args.modes},
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[INFO] Bundle written to {bundle_dir}.")


if __name__ == "__main__":
    main()

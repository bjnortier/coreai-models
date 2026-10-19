# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export Whisper as a KV-cache bundle: a run-once encoder and a stateful,
single-token decoder step.

Unlike ``export.py`` (the plain HuggingFace forward, which recomputes the whole
encoder for every generated token), this writes the encoder/decoder bundle that
``CoreAISpeech``'s ``WhisperDecoder`` already knows how to drive:

* ``encoder.aimodel`` — ``input_features [1,n_mels,3000]`` -> ``encoder_hidden_states``
  (``n_mels`` is 128 for large-v3/turbo, 80 for medium)
* ``decoder.aimodel`` — one decoder token per call, with the self-attention KV
  cache carried as Core AI **state** (``keyCache`` / ``valueCache``) and updated
  in place between calls. Besides ``logits`` it returns ``alignment_logits``,
  the cross-attention scores of the model's alignment heads, which is what word
  timestamps are aligned from.

``encoder_hidden_states`` does not hold hidden states. The Swift decode loop
passes it as a plain input on every step and has no cross-attention cache, so a
literal export would reproject all 1500 encoder frames per token: ~19.7 GMACs
against ~0.079 GMACs for the whole rest of the step. Instead the encoder emits
the per-layer cross-attention K/V *already projected*, packed along the channel
axis as ``[1, 1500, n_layers * 2 * d_model]``; layer ``i`` reads K at channel
offset ``i*2*d_model`` and V at ``i*2*d_model + d_model``. The Swift side treats
the tensor opaquely, so this needs no runtime change — but it does mean this
encoder is not interchangeable with one that emits true hidden states. See
``README_KV.md``.

The decoder never reads an unwritten cache slot: the runtime hands it an
*uninitialized* cache, and ``KVCache.update_and_fetch`` returns only the
populated prefix ``[0, seq_len)`` where ``seq_len = position_ids.shape[-1]``.
Masking a full-length cache instead would be unsafe — uninitialized storage can
hold NaN, which no additive mask can suppress.

This script imports ``coreai_models``, which is not on PyPI, so unlike its
siblings it is not a self-contained ``uv run`` script. Run it from the repo
environment::

    uv run python models/whisper/export_kv.py --dtype float16 --overwrite
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import transformers
from coreai.runtime import AIModelAssetMetadata

from coreai_models._constants import KEY_CACHE_NAME, VALUE_CACHE_NAME
from coreai_models.export.macos import export_to_coreai
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives._ops import mutable_cache_update_and_fetch

# Whisper's forced decode prefix. Resolved to ids through the model's own
# tokenizer: large-v3 inserted <|yue|> ahead of the control tokens, so
# <|transcribe|>/<|notimestamps|> are 50360/50364 there but 50359/50363 on
# medium and earlier — a hardcoded list silently decodes as the wrong task.
FORCED_DECODER_TOKENS = [
    "<|startoftranscript|>",
    "<|en|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]

# Generated tokens per 30 s window. HuggingFace's own generation_config carries
# `max_length`, not `max_new_tokens`, and the Swift parser only reads the
# latter — leaving it out would silently cap decoding at 50 tokens.
MAX_NEW_TOKENS = 224

# Trace length for the dynamic `position_ids` axis. Not 1: a size-1 tracing hint
# gets specialized, and the runtime's first priming call passes length 1.
POSITION_TRACE_LEN = 8


class _BatchedPositions(torch.nn.Module):
    """Encoder positional table with an explicit leading batch axis.

    HuggingFace adds a rank-2 ``[1500, d_model]`` table straight onto the rank-3
    ``[1, 1500, d_model]`` convolution output (``modeling_whisper.py:685``).
    That rank-promoting broadcast makes the Neural Engine lowering mis-infer the
    add as ``[1, 1500, 1500, d_model]``, and the encoder then fails to load on
    ANE at all ("Program load failed — no memory"). Since the runtime loads
    assets with default specialization — which probes ANE first — that failure
    is fatal, not a fallback. Emitting rank 3 keeps the add plainly elementwise.
    """

    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.num_embeddings = embedding.num_embeddings

    def forward(self, positions):
        return self.embedding(positions).unsqueeze(0)


class WhisperEncoderCrossKV(torch.nn.Module):
    """Encoder forward that emits packed per-layer cross-attention K/V.

    Output is ``[1, 1500, n_decoder_layers * 2 * d_model]`` — for each decoder
    layer in order, that layer's cross-attention K projection followed by its V
    projection. Doing this once here is what keeps the decode step cheap.
    """

    def __init__(self, hf: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = hf.model.encoder
        self.encoder.embed_positions = _BatchedPositions(self.encoder.embed_positions)
        self.decoder_layers = hf.model.decoder.layers

    def forward(self, input_features):
        enc = self.encoder(input_features)[0]  # [1, 1500, d_model]
        parts = []
        for layer in self.decoder_layers:
            parts.append(layer.encoder_attn.k_proj(enc))
            parts.append(layer.encoder_attn.v_proj(enc))
        return torch.cat(parts, dim=-1)


class WhisperDecoderStep(torch.nn.Module):
    """One greedy decoder token, with self-attention K/V carried as state.

    The forward signature order is load-bearing: reference inputs bind
    positionally to the traced signature, while ``input_names`` and
    ``state_names`` each carry only the relative order of their own kind.

    Inputs:
        input_ids             [1, 1] int32 — previous token
        position_ids          [1, S] int32 — arange(0..pos); its *length* is the
                              populated cache length, matching the macOS LLM
                              convention (offset = S - query_len)
        encoder_hidden_states [1, 1500, n_layers*2*d_model] — packed cross K/V
    State:
        k_cache / v_cache     [n_layers, 1, n_heads, 448, head_dim], exported as
                              ``keyCache`` / ``valueCache``
    Outputs:
        logits           [1, 1, vocab]
        alignment_logits [1, n_alignment_heads, 1500] — pre-softmax cross-attention
                         scores of ``generation_config.alignment_heads``, in that
                         order. Word timestamps are aligned from these (dynamic time
                         warping, as HF ``return_token_timestamps`` and openai-whisper
                         ``word_timestamps`` do); they are left unnormalized so the
                         host can take the softmax over the valid frames only.
    """

    def __init__(self, hf: torch.nn.Module) -> None:
        super().__init__()
        dec = hf.model.decoder
        cfg = hf.config
        self.layers = dec.layers
        self.embed_tokens = dec.embed_tokens
        self.embed_positions = dec.embed_positions
        self.layer_norm = dec.layer_norm
        self.proj_out = hf.proj_out
        self.d_model = cfg.d_model
        self.n_heads = cfg.decoder_attention_heads
        self.head_dim = cfg.d_model // cfg.decoder_attention_heads
        # No causal mask: query_len is 1 and the fetched K/V is exactly the
        # populated prefix, which already includes the slot just written.
        # `scale=None` resolves to head_dim**-0.5, matching WhisperAttention.
        self.self_sdpa = SDPA()
        self.cross_sdpa = SDPA()
        # The fused SDPA returns no attention weights, so the alignment heads' scores are
        # recomputed beside it: a [1, 64] x [64, 1500] product per head, negligible next
        # to the step. Grouped by layer, in the generation config's (layer, head) order.
        heads = getattr(hf.generation_config, "alignment_heads", None)
        if not heads:
            raise ValueError(f"{cfg.name_or_path} has no alignment_heads in its generation config")
        self.alignment_heads = [tuple(pair) for pair in heads]

    def _alignment_logits(self, i, q, ck):
        """Layer ``i``'s alignment heads' cross-attention scores, one ``[1, 1, query_len, src]`` each."""
        return [
            torch.matmul(q.narrow(1, head, 1), ck.narrow(1, head, 1).transpose(-1, -2))
            * self.head_dim**-0.5
            for layer, head in self.alignment_heads
            if layer == i
        ]

    def _heads(self, x, length):
        """[1, length, d_model] -> [1, n_heads, length, head_dim]."""
        return x.reshape(1, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, input_ids, position_ids, encoder_hidden_states, k_cache, v_cache):
        d = self.d_model
        query_len = input_ids.shape[1]
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)

        cache = KVCache(k_cache, v_cache)
        alignment = []

        h = self.embed_tokens(input_ids)
        pos = position_ids.narrow(-1, offset, query_len).reshape(-1)
        h = h + self.embed_positions.weight.index_select(0, pos).reshape(
            1, query_len, d
        )

        for i, layer in enumerate(self.layers):
            # --- self-attention, cached ---
            residual = h
            x = layer.self_attn_layer_norm(h)
            sa = layer.self_attn
            q = self._heads(sa.q_proj(x), query_len)
            k = self._heads(sa.k_proj(x), query_len)
            v = self._heads(sa.v_proj(x), query_len)
            k, v = cache.update_and_fetch(
                i, offset, k, v, seq_len=seq_len, query_len=query_len
            )
            a = self.self_sdpa(q, k, v).transpose(1, 2).reshape(1, query_len, d)
            h = residual + sa.out_proj(a)

            # --- cross-attention, K/V sliced out of the packed encoder tensor ---
            residual = h
            x = layer.encoder_attn_layer_norm(h)
            ca = layer.encoder_attn
            q = self._heads(ca.q_proj(x), query_len)
            base = i * 2 * d
            src = encoder_hidden_states.shape[1]
            ck = self._heads(encoder_hidden_states.narrow(2, base, d), src)
            cv = self._heads(encoder_hidden_states.narrow(2, base + d, d), src)
            a = self.cross_sdpa(q, ck, cv).transpose(1, 2).reshape(1, query_len, d)
            h = residual + ca.out_proj(a)
            alignment += self._alignment_logits(i, q, ck)

            # --- feed-forward ---
            residual = h
            x = layer.final_layer_norm(h)
            h = residual + layer.fc2(layer.activation_fn(layer.fc1(x)))

        src = encoder_hidden_states.shape[1]
        alignment = torch.cat(alignment, dim=1).reshape(1, len(self.alignment_heads), src)
        return self.proj_out(self.layer_norm(h)), alignment


class WhisperDecoderStepStatic(WhisperDecoderStep):
    """``WhisperDecoderStep`` with every input shape fixed, for the GPU.

    The dynamic step attends over the populated prefix, whose length is
    ``position_ids.shape[-1]``, so every token is a new shape and the GPU
    re-specializes the graph at each step. Here ``position_ids`` is ``[1, 1]``
    holding the current position, the new K/V is written at that position by
    value, and self-attention always reads all ``max_target_positions`` slots
    under a mask. Slots past the position are zeroed with ``where`` before the
    matmul, since the runtime's cache is uninitialized and NaN survives an
    additive mask.

    Inputs:
        input_ids             [1, 1] int32 — previous token
        position_ids          [1, 1] int32 — that token's position
        encoder_hidden_states [1, 1500, n_layers*2*d_model] — packed cross K/V
    """

    def forward(self, input_ids, position_ids, encoder_hidden_states, k_cache, v_cache):
        d = self.d_model
        n_heads = k_cache.shape[2]
        max_tgt = k_cache.shape[3]
        head_dim = k_cache.shape[4]
        pos = position_ids.reshape(1)

        h = self.embed_tokens(input_ids)
        h = h + self.embed_positions.weight.index_select(0, pos).reshape(1, 1, d)

        slots = torch.arange(max_tgt, dtype=torch.int32)
        valid = (slots <= pos).reshape(1, 1, max_tgt, 1)
        zero = torch.zeros((), dtype=h.dtype)
        mask = torch.where(
            valid.reshape(1, 1, 1, max_tgt), zero, torch.finfo(h.dtype).min
        ).to(h.dtype)
        zero_i = torch.zeros(1, dtype=torch.int32)
        alignment = []

        def update(cache, i, x):
            layer = torch.tensor([i], dtype=torch.int32)
            begin = torch.cat([layer, zero_i, zero_i, pos, zero_i])
            end = torch.cat(
                [
                    layer + 1,
                    torch.tensor([1, n_heads], dtype=torch.int32),
                    pos + 1,
                    torch.tensor([head_dim], dtype=torch.int32),
                ]
            )
            full = mutable_cache_update_and_fetch(
                x=cache,
                update=x,
                begin=begin,
                end=end,
                layer_idx=i,
                seq_dim=-2,
                seq_len=None,
            )
            return torch.where(valid, full, zero)

        for i, layer in enumerate(self.layers):
            # --- self-attention over every cache slot, masked past `pos` ---
            residual = h
            x = layer.self_attn_layer_norm(h)
            sa = layer.self_attn
            q = self._heads(sa.q_proj(x), 1)
            k = update(k_cache, i, self._heads(sa.k_proj(x), 1))
            v = update(v_cache, i, self._heads(sa.v_proj(x), 1))
            a = self.self_sdpa(q, k, v, attn_mask=mask)
            h = residual + sa.out_proj(a.transpose(1, 2).reshape(1, 1, d))

            # --- cross-attention, K/V sliced out of the packed encoder tensor ---
            residual = h
            x = layer.encoder_attn_layer_norm(h)
            ca = layer.encoder_attn
            q = self._heads(ca.q_proj(x), 1)
            base = i * 2 * d
            src = encoder_hidden_states.shape[1]
            ck = self._heads(encoder_hidden_states.narrow(2, base, d), src)
            cv = self._heads(encoder_hidden_states.narrow(2, base + d, d), src)
            a = self.cross_sdpa(q, ck, cv).transpose(1, 2).reshape(1, 1, d)
            h = residual + ca.out_proj(a)
            alignment += self._alignment_logits(i, q, ck)

            # --- feed-forward ---
            residual = h
            x = layer.final_layer_norm(h)
            h = residual + layer.fc2(layer.activation_fn(layer.fc1(x)))

        src = encoder_hidden_states.shape[1]
        alignment = torch.cat(alignment, dim=1).reshape(1, len(self.alignment_heads), src)
        return self.proj_out(self.layer_norm(h)), alignment


def _metadata(model_name: str, component: str) -> AIModelAssetMetadata:
    # export/metadata.py has no Whisper entry, so build this inline rather than
    # take its empty-author warning path.
    metadata = AIModelAssetMetadata()
    metadata.author = "A. Radford et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = (
        f"Whisper ASR ({component}, KV-cache bundle). "
        f"Source: https://huggingface.co/{model_name}"
    )
    metadata.creation_date = int(time.time())
    return metadata


def _prepare_bundle_dir(bundle_dir: Path, overwrite: bool) -> None:
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{bundle_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)


def _write_bundle_metadata(
    bundle_dir: Path, variant: str, cfg, dtype: torch.dtype, alignment_heads
) -> None:
    metadata = {
        "metadata_version": "0.2",
        "kind": "speech_recognizer",
        "name": variant,
        # The Whisper loader resolves encoder.aimodel/decoder.aimodel by
        # hardcoded filename and ignores this map; it is here for humans.
        "assets": {"encoder": "encoder.aimodel", "decoder": "decoder.aimodel"},
        "config": {
            "architecture": "whisper",
            "num_mel_bins": cfg.num_mel_bins,
            "d_model": cfg.d_model,
            "decoder_layers": cfg.decoder_layers,
            "decoder_attention_heads": cfg.decoder_attention_heads,
            "max_target_positions": cfg.max_target_positions,
            "vocab_size": cfg.vocab_size,
            "dtype": str(dtype).split(".")[-1],
            # Flags the deviation documented in README_KV.md: the tensor named
            # `encoder_hidden_states` carries packed per-layer cross K/V.
            "cross_kv_packed": True,
            # The (layer, head) pairs behind the decoder's `alignment_logits`, in order.
            "alignment_heads": alignment_heads,
        },
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _forced_decoder_ids(tokenizer) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(FORCED_DECODER_TOKENS)
    for token, token_id in zip(FORCED_DECODER_TOKENS, ids):
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"tokenizer has no {token} token")
    return ids


def _write_generation_config(
    bundle_dir: Path, model_name: str, cfg, forced_decoder_ids: list[int]
) -> None:
    config = {
        "forced_decoder_ids": [[i, tok] for i, tok in enumerate(forced_decoder_ids)],
        "eos_token_id": cfg.eos_token_id,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tokenizer_name": model_name,
    }
    (bundle_dir / "generation_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


def _write_added_tokens(bundle_dir: Path) -> None:
    """Write the special-token table CoreAISpeech reads language and task ids from.

    transformers 5 saves only tokenizer.json and tokenizer_config.json, so the
    table is taken from tokenizer.json's `added_tokens` rather than relying on
    `save_pretrained` to emit added_tokens.json.
    """
    tokenizer = json.loads((bundle_dir / "tokenizer.json").read_text())
    table = {token["content"]: token["id"] for token in tokenizer["added_tokens"]}
    for required in ("<|translate|>", "<|transcribe|>", "<|en|>"):
        if required not in table:
            raise ValueError(f"tokenizer.json has no {required} token")
    (bundle_dir / "added_tokens.json").write_text(json.dumps(table, indent=2) + "\n")


def create_whisper_kv(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
    static_positions: bool = False,
) -> Path:
    print("[INFO] Sourcing model...")
    hf = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=dtype, use_safetensors=True
    )
    hf.eval()
    cfg = hf.config
    n_layers = cfg.decoder_layers
    n_heads = cfg.decoder_attention_heads
    head_dim = cfg.d_model // n_heads
    max_tgt = cfg.max_target_positions
    packed = n_layers * 2 * cfg.d_model

    kind = "kv-static" if static_positions else "kv"
    variant = f"{Path(model_name).name}-{kind}_{str(dtype).split('.')[-1]}"
    bundle_dir = Path(output_dir) / variant
    _prepare_bundle_dir(bundle_dir, overwrite)

    processor = transformers.AutoProcessor.from_pretrained(model_name)
    forced_decoder_ids = _forced_decoder_ids(processor.tokenizer)

    # --- Encoder ---
    print(
        f"[INFO] Exporting encoder (input_features -> packed cross K/V, {packed} ch)..."
    )
    encoder = WhisperEncoderCrossKV(hf)
    encoder.eval()
    enc_program = export_to_coreai(
        encoder,
        reference_inputs={
            "input_features": torch.zeros(
                1, cfg.num_mel_bins, 2 * cfg.max_source_positions
            ).to(dtype)
        },
        dynamic_shapes={"input_features": None},
        input_names=("input_features",),
        output_names=("encoder_hidden_states",),
    )
    enc_program.optimize()
    enc_program.save_asset(
        bundle_dir / "encoder.aimodel", _metadata(model_name, "encoder")
    )
    print("[INFO] Encoder saved.")

    # --- Decoder step ---
    print("[INFO] Exporting decoder step (stateful self-attention KV cache)...")
    decoder = (WhisperDecoderStepStatic if static_positions else WhisperDecoderStep)(hf)
    decoder.eval()
    reference_inputs = {
        "input_ids": torch.tensor([[forced_decoder_ids[0]]], dtype=torch.int32),
        "position_ids": (
            torch.tensor([[POSITION_TRACE_LEN - 1]], dtype=torch.int32)
            if static_positions
            else torch.arange(POSITION_TRACE_LEN, dtype=torch.int32).unsqueeze(0)
        ),
        "encoder_hidden_states": torch.zeros(
            1, cfg.max_source_positions, packed, dtype=dtype
        ),
        # KVCache.create_cache_tensors reads LLM config attribute names that
        # WhisperConfig does not have, so build the tensors directly.
        "k_cache": torch.zeros(n_layers, 1, n_heads, max_tgt, head_dim, dtype=dtype),
        "v_cache": torch.zeros(n_layers, 1, n_heads, max_tgt, head_dim, dtype=dtype),
    }
    dynamic_shapes = {
        "input_ids": None,
        # min=1: the runtime's first priming call passes a single position.
        "position_ids": (
            None
            if static_positions
            else {1: torch.export.Dim("seq_pos", min=1, max=max_tgt)}
        ),
        "encoder_hidden_states": None,
        # Traced at the full context, so the cache has nowhere to grow.
        "k_cache": None,
        "v_cache": None,
    }
    dec_program = export_to_coreai(
        decoder,
        reference_inputs=reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids", "encoder_hidden_states"),
        output_names=("logits", "alignment_logits"),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    dec_program.optimize()
    dec_program.save_asset(
        bundle_dir / "decoder.aimodel", _metadata(model_name, "decoder step")
    )
    print("[INFO] Decoder saved.")

    # --- Sidecars ---
    _write_bundle_metadata(
        bundle_dir, variant, cfg, dtype, [list(pair) for pair in decoder.alignment_heads]
    )
    _write_generation_config(bundle_dir, model_name, cfg, forced_decoder_ids)
    # swift-transformers needs tokenizer.json *and* tokenizer_config.json in the
    # bundle root; if the local load fails it does not fall back to the HF cache.
    processor.tokenizer.save_pretrained(str(bundle_dir))
    _write_added_tokens(bundle_dir)
    print(f"[INFO] Wrote KV-cache bundle to {bundle_dir}")
    return bundle_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export a KV-cache Whisper bundle for the CoreAISpeech runtime."
    )
    parser.add_argument(
        "--model",
        choices=[
            "openai/whisper-large-v3-turbo",
            "openai/whisper-large-v3",
            "openai/whisper-medium",
        ],
        default="openai/whisper-large-v3-turbo",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--dtype",
        # No bfloat16: the Swift fills input_features and reads logits through
        # helpers that accept only float16/float32.
        choices=["float16", "float32"],
        default="float16",
        help="Torch dtype to use for the model.",
    )
    parser.add_argument(
        "--static-positions",
        action="store_true",
        help="Export a decoder whose inputs all have fixed shapes, so the GPU "
        "specializes it once (bundle suffix -kv-static).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle at the output path.",
    )
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    output_dir = args.output_dir or str(Path(__file__).resolve().parents[2] / "exports")
    create_whisper_kv(
        output_dir, args.model, dtype, args.overwrite, args.static_positions
    )


if __name__ == "__main__":
    main()

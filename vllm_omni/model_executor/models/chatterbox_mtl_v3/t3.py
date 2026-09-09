# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 T3: text + conditioning -> speech codec ids.

The backbone is vLLM's paged-attention ``LlamaModel``; everything around it --
the text/speech embeddings, the two learned position tables, the conditioning
encoder and the speech head -- is Chatterbox's own and is loaded from the V3
checkpoint verbatim.

Three properties of the reference are load-bearing and are asserted here rather
than left implicit (plan sections 3.1, 3.3):

* the backbone's own ``embed_tokens`` is an unused 8-entry placeholder; a codec
  id must never be routed through it;
* the prefill ends with **two** BOS embeddings, both at learned speech
  position 0;
* the first generated token is fed back at learned speech position **1**, and
  the index comes from the request's own generated count -- never from a
  batch-wide counter or the global transformer position.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K

logger = init_logger(__name__)

# Upstream `tfmr.*` -> vLLM packed-projection destinations.
_STACKED_PARAMS: tuple[tuple[str, str, int | str], ...] = (
    (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
    (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
    (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
    (".mlp.gate_up_proj", ".mlp.gate_proj", 0),
    (".mlp.gate_up_proj", ".mlp.up_proj", 1),
)

# Parameters present in the checkpoint that inference legitimately does not use.
# This is an explicit allow-list, not a "tolerate whatever is missing" policy:
# `text_head` only matters for training-time text loss, and the backbone's
# placeholder embedding table is never read because we always supply
# `inputs_embeds`.
INFERENCE_UNUSED_WEIGHTS = frozenset(
    {
        "text_head.weight",
        "tfmr.embed_tokens.weight",
    }
)


class WeightLoadError(RuntimeError):
    """A checkpoint that must not be served.

    Raised for a missing required tensor, an unexpected tensor, or a shape that
    identifies the wrong checkpoint (e.g. the 704-entry English text embedding
    instead of the 2454-entry multilingual one).
    """


class T3LearnedPositions(nn.Module):
    """Learned absolute position table (``LearnedPositionEmbeddings`` upstream).

    Kept as a module with the same parameter name (``emb.weight``) so the
    checkpoint's ``text_pos_emb.emb.weight`` / ``speech_pos_emb.emb.weight``
    map across without renaming.
    """

    def __init__(self, seq_len: int, model_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


class ChatterboxT3(nn.Module):
    """T3 with a vLLM paged-attention backbone.

    ``vllm_config`` must already carry a Llama ``hf_config`` describing the
    520M backbone (see :meth:`backbone_vllm_config`).
    """

    def __init__(self, *, vllm_config: VllmConfig | None = None, prefix: str = "") -> None:
        """``vllm_config=None`` builds the custom modules WITHOUT the paged
        backbone. That mode exists for the embedding-parity gates and for tools
        that only need conditioning; it cannot generate."""
        super().__init__()
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.t3.modules.cond_enc import (
            T3CondEnc,
        )
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.t3.modules.t3_config import (
            T3Config,
        )

        self.hp = T3Config.multilingual()
        if self.hp.text_tokens_dict_size != K.TEXT_VOCAB_SIZE:
            raise WeightLoadError(
                "T3Config.multilingual() must describe the 2454-token multilingual "
                f"vocabulary, got {self.hp.text_tokens_dict_size}"
            )
        dim = K.HIDDEN_SIZE

        self.tfmr = (
            LlamaModel(vllm_config=vllm_config, prefix=f"{prefix}tfmr")
            if vllm_config is not None
            else None
        )
        self.cond_enc = T3CondEnc(self.hp)
        self.text_emb = nn.Embedding(K.TEXT_VOCAB_SIZE, dim)
        self.speech_emb = nn.Embedding(K.SPEECH_VOCAB_SIZE, dim)
        self.text_pos_emb = T3LearnedPositions(K.TEXT_POS_TABLE_SIZE, dim)
        self.speech_pos_emb = T3LearnedPositions(K.SPEECH_POS_TABLE_SIZE, dim)
        self.speech_head = nn.Linear(dim, K.SPEECH_VOCAB_SIZE, bias=False)

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def prepare_conditioning(
        self,
        speaker_emb: torch.Tensor,
        cond_prompt_speech_tokens: torch.Tensor,
        exaggeration: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 34-position conditioning prefix.

        Mirrors ``T3.prepare_conditioning`` + ``T3CondEnc.forward``, but takes
        the three inputs explicitly instead of a mutable ``T3Cond`` whose
        derived embeddings are cached in place. That caching is exactly what
        makes the reference wrapper unsafe to share between concurrent
        requests.
        """
        from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.t3.modules.cond_enc import (
            T3Cond,
        )

        tokens = cond_prompt_speech_tokens.to(device=self.speech_emb.weight.device, dtype=torch.long)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        # Reference: prompt codec embeddings receive learned SPEECH positions
        # 0..len-1 before the Perceiver.
        prompt_emb = self.speech_emb(tokens)
        prompt_emb = prompt_emb + self.speech_pos_emb.emb(
            torch.arange(tokens.shape[1], device=tokens.device)
        )

        speaker_emb = speaker_emb.to(device=prompt_emb.device, dtype=prompt_emb.dtype)
        exaggeration = exaggeration.to(device=prompt_emb.device, dtype=prompt_emb.dtype)
        cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=tokens,
            cond_prompt_speech_emb=prompt_emb,
            emotion_adv=exaggeration,
        )
        out = self.cond_enc(cond)
        if out.shape[1] != K.COND_PREFIX_LEN:
            raise WeightLoadError(
                f"conditioning prefix is {out.shape[1]} positions, expected {K.COND_PREFIX_LEN}"
            )
        return out

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def text_content_embedding(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Text CONTENT only. Learned text positions are added separately.

        Splitting content from position is what lets the unconditional CFG row
        zero the content while keeping identical positions -- the reference's
        ``text_emb[1].zero_()``, which happens *before* positions are added.
        """
        return self.text_emb(text_ids)

    def speech_content_embedding(self, speech_ids: torch.Tensor) -> torch.Tensor:
        return self.speech_emb(speech_ids)

    def text_position_embedding(self, local_positions: torch.Tensor) -> torch.Tensor:
        return self.text_pos_emb.emb(local_positions)

    def speech_position_embedding(self, local_positions: torch.Tensor) -> torch.Tensor:
        return self.speech_pos_emb.emb(local_positions)

    def bos_embedding(self) -> torch.Tensor:
        """The prefill BOS embedding: speech BOS at learned speech position 0."""
        return (
            self.speech_emb.weight[K.START_SPEECH_TOKEN]
            + self.speech_pos_emb.emb.weight[0]
        )

    # ------------------------------------------------------------------
    # Backbone / head
    # ------------------------------------------------------------------
    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.tfmr is None:
            raise RuntimeError("ChatterboxT3 was built without a backbone and cannot generate")
        return self.tfmr(
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )

    def compute_speech_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.speech_head(hidden_states)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strict load of the V3 T3 checkpoint.

        Every required tensor must be consumed; anything unexpected fails. The
        only tolerated absentees are :data:`INFERENCE_UNUSED_WEIGHTS`.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        unexpected: list[str] = []

        for name, weight in weights:
            if name in INFERENCE_UNUSED_WEIGHTS:
                continue

            if name.startswith("tfmr."):
                if self.tfmr is None:
                    continue  # embeddings-only mode: the backbone is absent by design
                self._load_backbone_weight(name, weight, params, loaded, unexpected)
                continue

            target = params.get(name)
            if target is None:
                unexpected.append(name)
                continue
            self._validate_shape(name, target, weight)
            loader = getattr(target, "weight_loader", default_weight_loader)
            loader(target, weight)
            loaded.add(name)

        required = {
            n for n in params
            if n not in INFERENCE_UNUSED_WEIGHTS and not n.startswith("tfmr.embed_tokens")
        }
        missing = sorted(required - loaded)
        if missing or unexpected:
            raise WeightLoadError(
                f"Chatterbox V3 T3 checkpoint mismatch: "
                f"{len(missing)} required tensors never loaded {missing[:8]}, "
                f"{len(unexpected)} unexpected tensors {unexpected[:8]}"
            )
        logger.info(
            "Chatterbox MTL V3 T3: loaded %d parameters, %d documented inference-unused "
            "tensors skipped (%s)",
            len(loaded),
            len(INFERENCE_UNUSED_WEIGHTS),
            ", ".join(sorted(INFERENCE_UNUSED_WEIGHTS)),
        )
        return loaded

    def _load_backbone_weight(
        self,
        name: str,
        weight: torch.Tensor,
        params: dict[str, nn.Parameter],
        loaded: set[str],
        unexpected: list[str],
    ) -> None:
        """Route one ``tfmr.*`` tensor into the vLLM backbone.

        Packed projections go through the installed parameter's own
        ``weight_loader`` with the right shard id -- never a manual concat or
        slice, which would silently produce a wrong-but-loadable model under
        tensor parallelism.
        """
        for packed, unpacked, shard_id in _STACKED_PARAMS:
            if unpacked not in name:
                continue
            target_name = name.replace(unpacked, packed)
            target = params.get(target_name)
            if target is None:
                unexpected.append(name)
                return
            target.weight_loader(target, weight, shard_id)
            loaded.add(target_name)
            return

        target = params.get(name)
        if target is None:
            unexpected.append(name)
            return
        self._validate_shape(name, target, weight)
        loader = getattr(target, "weight_loader", default_weight_loader)
        loader(target, weight)
        loaded.add(name)

    @staticmethod
    def _validate_shape(name: str, target: nn.Parameter, weight: torch.Tensor) -> None:
        """Reject a wrong-checkpoint tensor instead of resizing it.

        The English checkpoint's 704-entry ``text_emb`` is loadable-looking and
        catastrophic: it would tokenize Arabic into ids the table cannot even
        index.
        """
        if tuple(target.shape) != tuple(weight.shape):
            raise WeightLoadError(
                f"{name}: checkpoint shape {tuple(weight.shape)} != model shape "
                f"{tuple(target.shape)}. This is a different checkpoint, not a "
                f"resizable one."
            )


def backbone_vllm_config(parent: VllmConfig) -> VllmConfig:
    """A ``VllmConfig`` describing the 520M Llama backbone.

    The backbone config is kept SEPARATE from the speech-output config: its
    ``vocab_size`` is the upstream 8-entry placeholder (never used, because we
    always supply ``inputs_embeds``), while the engine samples in the 8194-wide
    speech space. Conflating them is the "codec id through the placeholder
    embedding" bug the plan calls out.
    """
    from transformers import LlamaConfig

    hf = LlamaConfig(
        vocab_size=K.BACKBONE_PLACEHOLDER_VOCAB_SIZE,
        max_position_embeddings=K.BACKBONE_MAX_POSITION_EMBEDDINGS,
        hidden_size=K.HIDDEN_SIZE,
        intermediate_size=K.INTERMEDIATE_SIZE,
        num_hidden_layers=K.NUM_LAYERS,
        num_attention_heads=K.NUM_HEADS,
        num_key_value_heads=K.NUM_KV_HEADS,
        head_dim=K.HEAD_DIM,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        rms_norm_eps=K.RMS_NORM_EPS,
        rope_theta=K.ROPE_THETA,
        # The full scaling dict matters: dropping it (keeping only theta)
        # changes every RoPE frequency and therefore every logit.
        rope_scaling=dict(K.ROPE_SCALING),
        tie_word_embeddings=False,
        initializer_range=0.02,
        attention_dropout=0.0,
    )
    return parent.with_hf_config(hf, architectures=["LlamaModel"])


def build_prefill_embeddings(
    t3: ChatterboxT3,
    *,
    cond_prefix: torch.Tensor,
    text_ids: torch.Tensor,
    role: str = "cond",
) -> torch.Tensor:
    """The canonical Chatterbox V3 prefill, as one ``[34 + T + 2, 1024]`` matrix.

    This is the *definition* the flat-batch serving path must agree with; a test
    asserts the two produce identical rows. Layout::

        conditioning(34)
        text_embedding(T) + text_position(0 .. T-1)     [content zeroed if uncond]
        speech_embedding(BOS) + speech_position(0)
        speech_embedding(BOS) + speech_position(0)      <- duplicate, reference behaviour

    ``role="uncond"`` zeroes the text CONTENT and keeps the text POSITIONS. It
    is not an empty-text request: the row must stay the same length with the
    same position indices, or the guidance pair desynchronises.
    """
    if role not in ("cond", "uncond"):
        raise ValueError(f"role must be 'cond' or 'uncond', got {role!r}")

    device = cond_prefix.device
    text_ids = text_ids.to(device=device, dtype=torch.long).reshape(-1)
    n_text = int(text_ids.shape[0])

    text = t3.text_content_embedding(text_ids)
    if role == "uncond":
        text = torch.zeros_like(text)
    text = text + t3.text_position_embedding(torch.arange(n_text, device=device))

    bos = t3.bos_embedding().unsqueeze(0).expand(K.NUM_PREFILL_BOS, -1)
    return torch.cat((cond_prefix.reshape(-1, cond_prefix.shape[-1]), text, bos), dim=0)


def decode_step_embedding(
    t3: ChatterboxT3,
    speech_ids: torch.Tensor,
    generated_index: torch.Tensor,
) -> torch.Tensor:
    """Embedding for a fed-back speech token.

    ``generated_index`` is the request's own 0-based index of the token being
    fed, so the learned speech position is ``generated_index + 1``. Deriving it
    from a batch-wide counter or from the global transformer position is the
    failure mode the plan calls out: both are wrong the moment requests have
    different prompt lengths or the scheduler compacts slots.
    """
    content = t3.speech_content_embedding(speech_ids.to(torch.long))
    return content + t3.speech_position_embedding(generated_index.to(torch.long) + 1)

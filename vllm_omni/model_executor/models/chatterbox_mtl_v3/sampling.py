# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guided sampling for Chatterbox Multilingual V3.

One non-argmax-invariant logits processor does the three things the engine
cannot do generically, in the reference's own order:

1. **CFG blend.** The reference computes ``g = l_c + w*(l_c - l_u)``; Omni's
   pair infrastructure computes ``g = l_u + s*(l_c - l_u)``. They are the same
   function with ``s = 1 + w``, so ``cfg_weight=0.5`` is ``cfg_scale=1.5``.
2. **Speech-only repetition penalty.** The reference's history is
   ``[speech BOS] + generated speech ids`` -- *not* the prompt, which under this
   model's layout is conditioning placeholders and text ids. vLLM's builtin
   penalty would include the prompt, so it is required to be 1.0 and this
   processor does the penalty itself.
3. **Legal-domain masking** (``hardened`` policy only), applied *after* the
   blend so ``-inf - -inf`` can never produce NaN.

Everything downstream -- temperature, min-p, top-p, the draw -- is the engine's,
and vLLM 0.28 runs it in exactly the reference's order:

    allowed-token mask -> THIS (non-argmax-invariant) -> builtin penalties(off)
        -> temperature -> min-p (argmax-invariant) -> top-p/top-k -> sample

Strict pairing: a row that declares a CFG role but whose partner is absent is
**never** allowed to continue unguided. It is forced to EOS and its pair id is
recorded in :func:`failed_cfg_pairs`, so the serving layer raises instead of
returning audio that was generated without guidance.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K

logger = init_logger(__name__)

COND = "cond"
UNCOND = "uncond"

#: Request-id suffix of the unconditional companion.
CFG_UNCOND_SUFFIX = "__cbx_cfg_uncond"

#: Sampling policies. ``reference`` reproduces the official loop exactly.
#: ``hardened`` additionally forbids any output outside ``0..6560`` plus EOS.
#: They are different distributions and are qualified separately.
POLICY_REFERENCE = "reference"
POLICY_HARDENED = "hardened"

_failed_pairs: set[str] = set()
_failed_lock = threading.Lock()


def record_failed_cfg_pair(pair_id: str) -> None:
    with _failed_lock:
        _failed_pairs.add(pair_id)


def failed_cfg_pairs() -> frozenset[str]:
    """Pair ids whose guidance broke. Their audio must not be served."""
    with _failed_lock:
        return frozenset(_failed_pairs)


def take_cfg_failure(pair_id: str) -> bool:
    """Consume (and clear) the failure flag for one pair."""
    with _failed_lock:
        if pair_id in _failed_pairs:
            _failed_pairs.discard(pair_id)
            return True
        return False


def clear_cfg_failures() -> None:
    with _failed_lock:
        _failed_pairs.clear()


def cfg_scale_from_weight(cfg_weight: float) -> float:
    """Chatterbox ``cfg_weight`` -> Omni ``cfg_scale``. See the module docstring."""
    return 1.0 + float(cfg_weight)


def legal_speech_token_mask(device: torch.device | str = "cpu") -> torch.Tensor:
    """``True`` for ids the acoustic stage can actually consume, plus EOS.

    Speech BOS (6561) is deliberately excluded: it is a prefill-only embedding,
    never a legal generated output.
    """
    mask = torch.zeros(K.SPEECH_VOCAB_SIZE, dtype=torch.bool, device=device)
    mask[: K.CODEC_VOCAB_SIZE] = True
    mask[K.STOP_SPEECH_TOKEN] = True
    return mask


class ChatterboxCFGError(RuntimeError):
    """Guidance could not be applied. Never downgraded to unguided generation."""


class _RowState:
    __slots__ = ("role", "pair_id", "cfg_scale", "penalty", "policy", "output_tokens", "seen", "seen_len")

    def __init__(
        self,
        role: str,
        pair_id: str,
        cfg_scale: float,
        penalty: float,
        policy: str,
        output_tokens: list[int],
    ) -> None:
        self.role = role
        self.pair_id = pair_id
        self.cfg_scale = cfg_scale
        self.penalty = penalty
        self.policy = policy
        self.output_tokens = output_tokens
        # CPU "has this id been generated" mask, grown incrementally so
        # the per-step cost is O(new tokens) rather than O(history).
        self.seen: np.ndarray | None = None
        self.seen_len = 0


class ChatterboxCFGLogitsProcessor(LogitsProcessor):
    """CFG blend + speech-only repetition penalty + optional legal-domain mask."""

    _sample_patched = False

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra = params.extra_args or {}
        role = extra.get("cfg_role")
        if role is None:
            return
        if role not in (COND, UNCOND):
            raise ValueError(f"cfg_role must be 'cond' or 'uncond', got {role!r}")
        if not extra.get("cfg_pair_id"):
            raise ValueError("a Chatterbox CFG request must carry cfg_pair_id")
        scale = extra.get("cfg_scale")
        if scale is None or not isinstance(scale, (int, float)) or scale < 1.0:
            raise ValueError(f"cfg_scale must be a number >= 1.0, got {scale!r}")
        policy = extra.get("chatterbox_policy", POLICY_REFERENCE)
        if policy not in (POLICY_REFERENCE, POLICY_HARDENED):
            raise ValueError(
                f"chatterbox_policy must be {POLICY_REFERENCE!r} or {POLICY_HARDENED!r}, got {policy!r}"
            )
        penalty = extra.get("chatterbox_repetition_penalty", K.DEFAULT_REPETITION_PENALTY)
        if not isinstance(penalty, (int, float)) or penalty <= 0:
            raise ValueError(f"chatterbox_repetition_penalty must be > 0, got {penalty!r}")
        # The builtin penalty would apply to the prompt as well; requiring 1.0
        # here makes double-application impossible rather than merely unlikely.
        if abs(float(getattr(params, "repetition_penalty", 1.0)) - 1.0) > 1e-9:
            raise ValueError(
                "Chatterbox applies its own speech-only repetition penalty; the engine's "
                "repetition_penalty must be exactly 1.0"
            )

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self._rows: dict[int, _RowState] = {}
        self._pairs: list[tuple[int, int, float]] = []
        self._dirty = True
        self._legal_mask = legal_speech_token_mask(device)
        self._illegal_mask = ~self._legal_mask
        self._ensure_sample_patched()

    # -- engine plumbing ----------------------------------------------------
    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def _ensure_sample_patched(cls) -> None:
        """Copy the cond row's sampled token into the uncond row.

        Identical logits do not give identical independent draws, so the two
        rows must be forced to the same token or the pair's KV histories
        diverge and every later step is guided against the wrong context.
        """
        if cls._sample_patched:
            return
        cls._sample_patched = True

        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        orig_sample = GPUARModelRunner._sample

        def _sample_with_chatterbox_cfg_sync(self, logits, spec_decode_metadata):
            out = orig_sample(self, logits, spec_decode_metadata)
            for proc in self.input_batch.logitsprocs.all:
                if isinstance(proc, ChatterboxCFGLogitsProcessor) and proc._pairs:
                    sampled = out.sampled_token_ids
                    num_rows = sampled.shape[0] if hasattr(sampled, "shape") else len(sampled)
                    pairs = [(c, u) for c, u, _ in proc._pairs if c < num_rows and u < num_rows]
                    if pairs:
                        indices = torch.tensor(pairs, dtype=torch.long, device=sampled.device)
                        sampled[indices[:, 1]] = sampled[indices[:, 0]]
                    break
            return out

        _sample_with_chatterbox_cfg_sync._chatterbox_cfg_sync = True  # type: ignore[attr-defined]
        GPUARModelRunner._sample = _sample_with_chatterbox_cfg_sync
        logger.info("ChatterboxCFGLogitsProcessor: patched GPUARModelRunner._sample for pair token sync")

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._rows.pop(idx, None)

        for idx, params, _prompt_token_ids, output_token_ids in batch_update.added:
            extra = (params.extra_args if params else None) or {}
            role = extra.get("cfg_role")
            if role in (COND, UNCOND):
                self._rows[idx] = _RowState(
                    role=role,
                    pair_id=str(extra.get("cfg_pair_id")),
                    cfg_scale=float(extra.get("cfg_scale", K.DEFAULT_CFG_SCALE)),
                    penalty=float(
                        extra.get("chatterbox_repetition_penalty", K.DEFAULT_REPETITION_PENALTY)
                    ),
                    policy=str(extra.get("chatterbox_policy", POLICY_REFERENCE)),
                    output_tokens=output_token_ids,
                )
            else:
                # A non-Chatterbox request taking over this slot must not
                # inherit the previous occupant's history.
                self._rows.pop(idx, None)

        for src, dst, direction in batch_update.moved:
            src_state = self._rows.pop(src, None)
            dst_state = self._rows.pop(dst, None)
            if src_state is not None:
                self._rows[dst] = src_state
            if direction == MoveDirectionality.SWAP and dst_state is not None:
                self._rows[src] = dst_state

        self._dirty = True

    def _rebuild_pairs(self) -> None:
        by_pair: dict[str, dict[str, tuple[int, float]]] = {}
        for idx, st in self._rows.items():
            by_pair.setdefault(st.pair_id, {})[st.role] = (idx, st.cfg_scale)
        self._pairs = [
            (roles[COND][0], roles[UNCOND][0], roles[COND][1])
            for roles in by_pair.values()
            if COND in roles and UNCOND in roles
        ]
        self._paired_rows = {i for pair in self._pairs for i in pair[:2]}
        self._dirty = False

    # -- the actual math ----------------------------------------------------
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._rows:
            return logits
        if self._dirty:
            self._rebuild_pairs()

        num_rows = logits.shape[0]
        handled: set[int] = set()

        pairs = [(c, u, scale) for c, u, scale in self._pairs if c < num_rows and u < num_rows]
        if pairs:
            # A fixed-shape batch replaces per-caller kernel launches, finite
            # checks, device-side nonzero and masked_scatter operations.
            cond_indices = torch.tensor([p[0] for p in pairs], device=logits.device)
            uncond_indices = torch.tensor([p[1] for p in pairs], device=logits.device)
            states = [self._rows[p[0]] for p in pairs]
            cond = logits[cond_indices].float()
            uncond = logits[uncond_indices].float()
            finite = torch.isfinite(cond).all(dim=1) & torch.isfinite(uncond).all(dim=1)
            if not bool(finite.all()):
                failed = (~finite).nonzero().flatten().cpu().tolist()
                raise ChatterboxCFGError(
                    "non-finite raw speech logits for CFG pair "
                    + ", ".join(states[i].pair_id for i in failed)
                )
            scales = torch.tensor([p[2] for p in pairs], device=logits.device, dtype=torch.float32)
            guided = uncond + scales[:, None] * (cond - uncond)
            hardened = torch.tensor(
                [st.policy == POLICY_HARDENED for st in states], device=logits.device, dtype=torch.bool
            )
            guided.masked_fill_(hardened[:, None] & self._illegal_mask[None, :], float("-inf"))

            # Histories originate on the CPU. Track membership there and send
            # one dense boolean matrix, avoiding a synchronizing nonzero per
            # row per token. State moves with its request, including rewinds.
            seen = np.empty((len(states), K.SPEECH_VOCAB_SIZE), dtype=np.bool_)
            for i, st in enumerate(states):
                history = st.output_tokens
                n = len(history)
                if st.seen is None or n < st.seen_len:
                    st.seen = np.zeros(K.SPEECH_VOCAB_SIZE, dtype=np.bool_)
                    st.seen[K.START_SPEECH_TOKEN] = True
                    st.seen_len = 0
                for token in history[st.seen_len:n]:
                    if 0 <= token < K.SPEECH_VOCAB_SIZE:
                        st.seen[token] = True
                st.seen_len = n
                seen[i] = st.seen
            mask = torch.from_numpy(seen).to(logits.device)
            penalties = torch.tensor([st.penalty for st in states], device=logits.device, dtype=torch.float32)
            penalized = torch.where(guided < 0, guided * penalties[:, None], guided / penalties[:, None])
            guided = torch.where(mask, penalized, guided)
            logits[cond_indices] = guided
            logits[uncond_indices] = guided
            handled = {i for c, u, _ in pairs for i in (c, u)}

        # Strict policy: any CFG row that was sampled this step without a
        # complete pair is a guidance failure, not an unguided request.
        for idx, st in self._rows.items():
            if idx >= num_rows or idx in handled:
                continue
            record_failed_cfg_pair(st.pair_id)
            logger.error(
                "Chatterbox CFG pair %s lost its %s companion; forcing EOS and failing the request",
                st.pair_id,
                UNCOND if st.role == COND else COND,
            )
            row = logits[idx]
            row.fill_(float("-inf"))
            row[K.STOP_SPEECH_TOKEN] = 0.0

        return logits

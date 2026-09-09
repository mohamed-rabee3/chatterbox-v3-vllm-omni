# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 stage input processors (T3 -> S3Gen).

Three responsibilities:

* ``expand_cfg_prompts`` -- mint the unconditional CFG companion. It carries the
  SAME prompt (same token ids, same reference conditioning, same length); only
  its role differs, and the model zeroes the text CONTENT for that row while
  keeping the text POSITIONS. Building a shorter "empty text" prompt instead
  would desynchronise the pair immediately.
* ``codec_full_payload`` / ``codec_token_only`` -- hand the completed clause's
  codec ids plus the acoustic reference conditioning to stage 1.
* ``codec_async_chunk`` -- stream completed codec blocks to stage 1 while the
  clause is still being generated, through a monotonic cursor.

The cursor is the point of ``codec_async_chunk``: chunk boundaries are decided
by how many ids have been *accepted*, never by ``len(tokens) % chunk == 0``,
because a repeated callback or the EOS callback revisits the same boundary and
would emit the same audio twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, EmbeddingsStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.sampling import (
    CFG_UNCOND_SUFFIX,
    COND,
    UNCOND,
    take_cfg_failure,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3.streaming import (
    CodecCursor,
    StreamStateError,
    TerminalReason,
)
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

logger = init_logger(__name__)

_STATE_KEY = "_chatterbox_async_state"

#: Cursors keyed by request id. The connector hands `codec_async_chunk` a fresh
#: request object on some callbacks, so state attached with `setattr` is lost
#: and the cursor restarts -- which makes it re-send the whole history every
#: time. Keying by id is what keeps "how much has already been sent" true.
_CURSORS: dict[str, CodecCursor] = {}


def _release_cursor(request_id: str) -> None:
    _CURSORS.pop(str(request_id), None)


# ----------------------------------------------------------------------------
# CFG prompt expansion
# ----------------------------------------------------------------------------
def expand_cfg_prompts(prompt: dict[str, Any] | str, sampling_params: Any) -> list[ExpandedPrompt]:
    """Emit the unconditional companion for a guided Chatterbox request.

    The companion is a byte-for-byte copy of the conditional prompt with
    ``cfg_role="uncond"``. Everything that determines sequence length and
    position indices -- text ids, the 34 conditioning positions, the two BOS --
    is identical, which is exactly the property the guidance blend needs.
    """
    extra = getattr(sampling_params, "extra_args", None) or {}
    if extra.get("cfg_role") != COND:
        return []
    try:
        cfg_scale = float(extra.get("cfg_scale", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid cfg_scale {extra.get('cfg_scale')!r}") from exc
    if cfg_scale <= 1.0:
        # cfg_weight == 0: the blend depends only on the conditional row, so a
        # companion would be pure waste. This is a separately qualified quality
        # profile, not the default.
        return []

    pair_id = extra.get("cfg_pair_id")
    if not pair_id:
        raise ValueError("a guided Chatterbox request must carry cfg_pair_id in extra_args")

    if isinstance(prompt, dict):
        companion: dict[str, Any] | str = dict(prompt)
        info = dict(companion.get("additional_information") or {})
        info["cfg_role"] = UNCOND
        companion["additional_information"] = info
    else:
        companion = prompt

    overrides = {k: v for k, v in extra.items()}
    overrides["cfg_role"] = UNCOND
    return [
        ExpandedPrompt(
            prompt=companion,
            role=UNCOND,
            request_id_suffix=CFG_UNCOND_SUFFIX,
            sampling_params_override={"extra_args": overrides},
        )
    ]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    out: list[int] = []
    for item in value:
        if isinstance(item, (list, tuple, torch.Tensor)):
            out.extend(_as_int_list(item))
        else:
            out.append(int(item))
    return out


def _strip_prefix(ids: list[int], prefix: list[int]) -> list[int]:
    if prefix and len(ids) >= len(prefix) and ids[: len(prefix)] == prefix:
        return ids[len(prefix) :]
    return ids


def _reference_embed(multi_modal_data: Any) -> dict[str, Any]:
    """Pull the acoustic reference conditioning out of stage 0's payload.

    Accepts any ``Mapping``: stage output arrives as ``MultimodalPayload``,
    which implements ``Mapping`` but is not a ``dict``, so an
    ``isinstance(..., dict)`` check silently sees nothing.
    """
    if not isinstance(multi_modal_data, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in ("speech_token", "speech_token_len", "speech_feat", "embedding"):
        value = multi_modal_data.get(f"embed.{key}")
        if value is None:
            nested = multi_modal_data.get("embed")
            if isinstance(nested, dict):
                value = nested.get(key)
        if value is not None:
            out[key] = value
    return out


def _valid_codes(ids: list[int]) -> list[int]:
    """Keep only real acoustic codec ids.

    Drops BOS/EOS and refuses anything outside ``0..6560`` -- including
    negatives, which a bare ``< 6561`` filter would let through.
    """
    return [i for i in ids if 0 <= i < K.CODEC_VOCAB_SIZE]


# ----------------------------------------------------------------------------
# sync (complete clause) path
# ----------------------------------------------------------------------------
def codec_token_only(
    source_outputs: list,
    prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = True,
):
    """Build stage-1 inputs from completed stage-0 generations."""
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        # A request whose guidance broke was terminated by the logits
        # processor; its codec ids were generated UNGUIDED and must not be
        # synthesised. This is the first point in the pipeline that knows both
        # the request id and that the generation finished.
        if take_cfg_failure(str(source_output.request_id)):
            raise RuntimeError(
                f"classifier-free guidance broke for {source_output.request_id} "
                f"(its companion row was lost); refusing to return unguided audio"
            )

        output = source_output.outputs[0]
        prompt_ids = _as_int_list(source_output.prompt_token_ids)
        raw = _strip_prefix(_as_int_list(output.cumulative_token_ids), prompt_ids)
        codes = _valid_codes(raw)

        multi_modal_data = output.multimodal_output
        embed = _reference_embed(multi_modal_data)
        if not embed:
            logger.error(
                "chatterbox_mtl_v3.codec_token_only: no acoustic reference conditioning for %s "
                "(multimodal_output type=%s keys=%s)",
                source_output.request_id,
                type(multi_modal_data).__name__,
                list(multi_modal_data.keys()) if isinstance(multi_modal_data, Mapping) else None,
            )
            raise RuntimeError(
                f"missing acoustic reference conditioning for {source_output.request_id}; "
                "refusing to synthesise with a substituted voice"
            )

        info: dict[str, Any] = dict(multi_modal_data)
        info["embed"] = embed
        info.setdefault("meta", {})["req_id"] = [str(source_output.request_id)]
        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codes,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs


def codec_full_payload(transfer_manager, pooling_output, request):
    """Producer-side payload: ship the acoustic reference conditioning."""
    del transfer_manager
    rid = getattr(request, "external_req_id", None) or getattr(request, "request_id", "?")
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "chatterbox_mtl_v3.codec_full_payload: pooling_output is %s for req=%s",
            type(pooling_output).__name__, rid,
        )
        return None

    embed_out: dict[str, Any] = {}
    for key in ("speech_token", "speech_token_len", "speech_feat", "embedding"):
        value = pooling_output.get(f"embed.{key}")
        if value is None:
            nested = pooling_output.get("embed")
            if isinstance(nested, dict):
                value = nested.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            embed_out[key] = value
    if not embed_out:
        logger.warning(
            "chatterbox_mtl_v3.codec_full_payload: no reference conditioning in pooling_output "
            "(keys=%s) for req=%s",
            list(pooling_output.keys()), rid,
        )
        return None

    return {
        "meta": {"finished": torch.tensor(True, dtype=torch.bool), "req_id": [str(rid)]},
        "embed": embed_out,
    }


# ----------------------------------------------------------------------------
# streaming (async chunk) path
# ----------------------------------------------------------------------------
def _connector_cfg(transfer_manager: Any) -> dict[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    if isinstance(raw, dict):
        extra = raw.get("extra", raw)
        return extra if isinstance(extra, dict) else {}
    return {}


def codec_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Stream completed codec blocks from stage 0 to stage 1.

    State lives on the request under :data:`_STATE_KEY` and is a
    :class:`CodecCursor`, so:

    * ids are consumed cumulatively and monotonically -- a repeated callback
      returns ``None`` instead of re-emitting the previous block;
    * EOS terminates and never reaches the acoustic stage;
    * a rewritten history raises instead of silently rewriting audio;
    * exactly one terminal payload is ever produced.

    ``holdback`` codes are retained until the flush so the final
    ``max(1, N-1)*960`` crop can still be applied.
    """
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", "?")
    cfg = _connector_cfg(transfer_manager)

    # Incremental acoustic decoding: stage 1 renders audio for codes 0..k while
    # the rest of the clause is still being generated. It is qualified by
    # `port/tools/eval_streaming_via_decode.py`, which drives this exact path
    # and measures what matters -- ASR CER identical to the one-shot decode,
    # speaker similarity within 0.019, no seam (the streamed waveform's largest
    # sample jump sits in the same range as the one-shot decode's), and total
    # length identical to the sample.
    #
    # What it is NOT is bit-equal to the one-shot decode: the token encoder is
    # bidirectional, so a prefix re-decoded with more context renders slightly
    # differently (5-36% of peak in mel). That is a different-but-equally-valid
    # rendering, which is why the gate measures intelligibility and identity
    # rather than equality.
    first_block = int(cfg.get("codec_chunk_frames", K.ACOUSTIC_STREAM_FIRST_BLOCK))
    growth = float(cfg.get("codec_chunk_growth", K.ACOUSTIC_STREAM_BLOCK_GROWTH))
    max_block = int(cfg.get("codec_max_chunk_frames", K.ACOUSTIC_STREAM_MAX_BLOCK))
    lookahead = int(cfg.get("codec_pre_lookahead_frames", K.ACOUSTIC_PRE_LOOKAHEAD_LEN))
    if first_block <= 0 or lookahead < 0 or growth < 1.0:
        raise ValueError(
            f"invalid chunk config: first_block={first_block} growth={growth} "
            f"lookahead={lookahead}"
        )

    cursor = _CURSORS.get(str(request_id))
    if cursor is None:
        cursor = CodecCursor(request_id=str(request_id))
        _CURSORS[str(request_id)] = cursor
    setattr(request, _STATE_KEY, cursor)

    finished = bool(is_finished or getattr(request, "is_finished", lambda: False)())

    # The unconditional twin must never publish codes. A guided request runs as
    # a PAIR of stage-0 sequences; the conditional row carries the guided result
    # and the companion exists only so the logits processor has something to
    # blend against. Its raw output is unguided -- decoding it produces audio
    # that is not speech. In the completed-clause path the companion is dropped
    # implicitly because only the conditional output is consumed; the chunked
    # path is called for BOTH rows, so it has to say so explicitly.
    if str(request_id).endswith(CFG_UNCOND_SUFFIX):
        if not finished:
            return None
        # Still deliver the terminal marker so the pair can complete rather
        # than leaving the connector waiting on a stream that never ends.
        if not cursor.send_terminal():
            return None
        seq = cursor.chunk_sequence
        _release_cursor(request_id)
        return OmniPayloadStruct(
            meta=MetaStruct(
                req_id=[str(request_id)],
                chunk_seq=seq,
                stream_finished=torch.tensor(True),
                finished=torch.tensor(True),
            )
        )

    output_ids = _as_int_list(getattr(request, "output_token_ids", None) or [])
    try:
        cursor.observe(output_ids)
    except StreamStateError:
        logger.exception("chatterbox_mtl_v3: stream state violated for %s", request_id)
        raise
    if finished and not cursor.is_terminal:
        cursor.mark_terminal(TerminalReason.LENGTH_LIMIT)

    # Send only the codes produced since the last call. The acoustic stage
    # keeps each request's cumulative sequence itself (it has to: the flow
    # decoder needs the whole prefix), so putting cumulative lists on the wire
    # as well would make the two accumulations compound -- a ~100-code
    # utterance arrived as 2550 codes before this was split.
    # End-of-stream is taken from the CURSOR (it sees the AR's EOS), not from
    # the connector's finished flag: stage 1 receives `stream_finished=False`
    # on every event because a terminal payload carrying no codes has nothing
    # to put on the token path and is dropped in transit. Holding one code back
    # guarantees the final payload is non-empty, so the flag always has a
    # carrier and the last chunk of every utterance actually gets decoded.
    flushing = bool(finished or cursor.is_terminal)
    chunk = cursor.take_chunk(
        block=1, holdback=0 if flushing else 1, force_flush=flushing
    )
    is_final = flushing
    terminal = False
    if flushing:
        # The stream must be closed exactly once even when the flush has no new
        # codes left to send (everything was already handed over by the last
        # incremental chunk). Without this the client would wait forever for an
        # end that never arrives.
        terminal = cursor.send_terminal()
        if chunk is None and terminal:
            chunk, is_final = [], True
    if chunk is None:
        return None

    # 1-D on purpose: a 1-D codes tensor is rerouted onto the TOKEN path and
    # arrives as stage 1's `input_ids`, which is where `_forward_s3gen` reads
    # the codec ids from. A 2-D (N, 1) tensor stays in the payload view instead,
    # and stage 1 then decodes a run of zeros -- right length, no speech.
    if terminal:
        _release_cursor(request_id)
    codes = torch.tensor(chunk, dtype=torch.long).reshape(-1)
    embed = _reference_embed(multimodal_output or {})
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        embed=EmbeddingsStruct(**embed) if embed else None,
        meta=MetaStruct(
            req_id=[str(request_id)],
            chunk_seq=cursor.chunk_sequence,
            stream_finished=torch.tensor(bool(terminal)),
            finished=torch.tensor(bool(terminal)),
            right_holdback_size=lookahead,
        ),
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming state for Chatterbox Multilingual V3.

Four coordinate systems are kept strictly apart (plan section 7.5); overloading
one ``offset`` for all of them is how duplicated or missing audio happens:

===========================  ====================================================
cumulative generated ids     everything the AR stage has emitted, EOS included
valid codec positions        ids in ``0..6560`` only -- BOS/EOS/invalid dropped
mel frames                   2 per codec token
absolute PCM samples         960 per codec token, at 24 kHz
===========================  ====================================================

Two invariants this module exists to enforce:

* a producer consumes **cumulative** output tokens through a monotonic cursor.
  Triggering on ``len(tokens) % chunk == 0`` is wrong: a repeated callback, or
  the EOS callback, revisits the same boundary and emits a chunk twice.
* every absolute PCM sample is emitted exactly once, with no gap and no
  retraction. The committer can prove that; it cannot prove the decoder's
  samples are linguistically stable -- that is the streaming *quality* gate,
  which is a separate question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K


class TerminalReason(str, Enum):
    """Why a stream ended. Deliberately not a boolean."""

    EOS = "eos"
    LENGTH_LIMIT = "length_limit"
    ABORTED = "aborted"
    FAILED = "failed"


class StreamStateError(RuntimeError):
    """A stream invariant was violated. Never repaired silently."""


def final_sample_limit(n_valid_codes: int) -> int:
    """The wrapper's final crop: ``max(1, N-1) * 960`` samples.

    The last codec token is emitted just before EOS with degraded attention and
    decodes to roughly 40 ms of noise, so the reference drops it. Streaming must
    reserve enough tail to be able to apply this at the end.
    """
    if n_valid_codes <= 0:
        return 0
    return max(1, n_valid_codes - 1) * K.SAMPLES_PER_CODEC_TOKEN


def is_valid_codec_id(token: int) -> bool:
    """``0 <= token < 6561``.

    The upstream ``drop_invalid_tokens`` only slices around BOS/EOS and a
    ``< 6561`` filter does not reject negatives, so validation is explicit here.
    """
    return 0 <= token < K.CODEC_VOCAB_SIZE


@dataclass
class CodecCursor:
    """Monotonic consumer of one request's cumulative AR output.

    ``request_epoch`` distinguishes restarted internal work from stale connector
    events: an event carrying an older epoch is discarded rather than replayed.
    """

    request_id: str
    request_epoch: int = 0
    #: How many cumulative ids have been accepted. The accepted prefix is immutable.
    seen_generated_ids: int = 0
    #: Codec ids accepted so far (BOS/EOS/invalid excluded).
    valid_codes: list[int] = field(default_factory=list)
    #: First valid codec position not yet handed to the acoustic stage.
    next_codec_to_commit: int = 0
    #: Monotonic connector sequence for this request/epoch.
    chunk_sequence: int = 0
    terminal_reason: TerminalReason | None = None
    terminal_sent: bool = False
    #: Ids seen that were neither a legal codec id nor a known control token.
    rejected: int = 0

    @property
    def valid_codec_count(self) -> int:
        return len(self.valid_codes)

    @property
    def is_terminal(self) -> bool:
        return self.terminal_reason is not None

    def observe(self, cumulative_ids: list[int]) -> int:
        """Accept a cumulative output list; return how many NEW ids were consumed.

        Idempotent for a repeated identical callback (returns 0). A cumulative
        list that contradicts the accepted prefix, or that shrinks, is a bug in
        the producer and is refused -- accepting it would rewrite audio that may
        already have been played.
        """
        n = len(cumulative_ids)
        if n < self.seen_generated_ids:
            raise StreamStateError(
                f"{self.request_id}: cumulative output shrank from "
                f"{self.seen_generated_ids} to {n} ids"
            )
        if n == self.seen_generated_ids:
            return 0

        prefix = cumulative_ids[: self.seen_generated_ids]
        if prefix != self._accepted_prefix[: self.seen_generated_ids]:
            raise StreamStateError(
                f"{self.request_id}: cumulative output rewrote its accepted prefix"
            )

        new_ids = cumulative_ids[self.seen_generated_ids : n]
        for token in new_ids:
            token = int(token)
            self._accepted_prefix.append(token)
            if token == K.STOP_SPEECH_TOKEN:
                self.mark_terminal(TerminalReason.EOS)
                continue
            if token == K.START_SPEECH_TOKEN:
                # BOS is prefill-only; the reference drops it from the output.
                continue
            if not is_valid_codec_id(token):
                self.rejected += 1
                continue
            if self.is_terminal:
                # Anything after EOS is not part of this utterance.
                continue
            self.valid_codes.append(token)
        consumed = n - self.seen_generated_ids
        self.seen_generated_ids = n
        return consumed

    _accepted_prefix: list[int] = field(default_factory=list, repr=False)

    def mark_terminal(self, reason: TerminalReason) -> None:
        """First terminal reason wins; later ones do not overwrite it."""
        if self.terminal_reason is None:
            self.terminal_reason = reason

    def take_chunk(self, *, block: int, holdback: int, force_flush: bool = False) -> list[int] | None:
        """Return the next codec block to decode, or ``None`` if not ready.

        ``holdback`` codes are retained so a later flush can still apply the
        final ``max(1, N-1)*960`` crop and any acoustic-stability policy. When
        the stream is terminal (or ``force_flush``), everything left is returned.
        """
        if block < 1:
            raise ValueError("block must be >= 1")
        if holdback < 0:
            raise ValueError("holdback must be >= 0")

        available = self.valid_codec_count - self.next_codec_to_commit
        flushing = force_flush or self.is_terminal
        if flushing:
            if available <= 0:
                return None
            take = available
        else:
            if available - holdback < block:
                return None
            take = available - holdback

        start = self.next_codec_to_commit
        end = start + take
        self.next_codec_to_commit = end
        self.chunk_sequence += 1
        return self.valid_codes[start:end]

    #: Codec count at which the next incremental decode fires.
    next_decode_at: int = 0
    #: Current block size; grows so cost stays near-linear (see take_prefix).
    current_block: int = 0
    #: Codec count handed to the acoustic stage by the last take_prefix.
    last_prefix_len: int = 0

    def take_prefix(
        self,
        *,
        first_block: int,
        growth: float,
        max_block: int,
        lookahead: int,
        force_flush: bool = False,
    ) -> tuple[list[int], bool] | None:
        """Return the CUMULATIVE codes to decode next, and whether it is final.

        Incremental decoding of this checkpoint has to re-decode the whole
        prefix, not just the new block: the flow decoder is conditioned on the
        token encoder's output, and that encoder is bidirectional, so a block
        decoded on its own has no left context and renders as a fragment. The
        acoustic stage therefore receives ``codes[0:k]`` each time and emits
        only the samples past what it already sent.

        The block GROWS after each chunk. Time-to-first-audio depends only on
        ``first_block``, while a fixed block would make the cumulative re-decode
        cost O(n^2/B) -- measured at x5.0 of one-shot at 225 codes, against
        x2.2 when the block doubles.
        """
        if first_block < 1:
            raise ValueError("first_block must be >= 1")
        if growth < 1.0:
            raise ValueError("growth must be >= 1.0")

        available = self.valid_codec_count
        flushing = force_flush or self.is_terminal
        if not self.current_block:
            self.current_block = first_block
            self.next_decode_at = first_block

        if flushing:
            if available <= self.last_prefix_len and self.last_prefix_len > 0:
                return None
            if available <= 0:
                return None
            self.last_prefix_len = available
            self.chunk_sequence += 1
            return list(self.valid_codes[:available]), True

        # A chunk shorter than the encoder's lookahead can emit nothing.
        if available < self.next_decode_at or available <= lookahead:
            return None
        self.last_prefix_len = available
        self.current_block = min(int(self.current_block * growth), max_block)
        self.next_decode_at = available + self.current_block
        self.chunk_sequence += 1
        return list(self.valid_codes[:available]), False

    def next_sequence(self) -> int:
        self.chunk_sequence += 1
        return self.chunk_sequence

    def send_terminal(self) -> bool:
        """Claim the single terminal event. Returns ``False`` if already sent."""
        if self.terminal_sent:
            return False
        self.terminal_sent = True
        return True


@dataclass
class PCMCommitter:
    """Emits each absolute PCM sample exactly once, in order.

    Rejects a gap (samples skipped) and a retraction (an attempt to re-emit or
    un-emit already-committed audio). It cannot say whether those samples are
    the *right* audio -- that is Gate D.
    """

    request_id: str
    next_sample_to_emit: int = 0
    total_committed: int = 0
    sample_limit: int | None = None

    def set_final_limit(self, n_valid_codes: int) -> None:
        """Apply the reference's final crop once the valid code count is known."""
        limit = final_sample_limit(n_valid_codes)
        if limit < self.next_sample_to_emit:
            raise StreamStateError(
                f"{self.request_id}: final crop to {limit} samples would retract "
                f"{self.next_sample_to_emit - limit} already-emitted samples"
            )
        self.sample_limit = limit

    def commit(self, start_sample: int, samples) -> list:
        """Commit ``samples`` starting at absolute index ``start_sample``.

        Returns the slice that is actually new (possibly empty). Re-delivering
        an already-committed range is tolerated and de-duplicated -- connectors
        do repeat -- but a *gap* is refused.
        """
        n = len(samples)
        end = start_sample + n
        if start_sample > self.next_sample_to_emit:
            raise StreamStateError(
                f"{self.request_id}: gap in PCM stream; expected sample "
                f"{self.next_sample_to_emit}, got {start_sample}"
            )
        if end <= self.next_sample_to_emit:
            return samples[:0]  # entirely a duplicate delivery

        skip = self.next_sample_to_emit - start_sample
        new = samples[skip:]
        if self.sample_limit is not None and end > self.sample_limit:
            keep = max(0, self.sample_limit - self.next_sample_to_emit)
            new = new[:keep]
        self.next_sample_to_emit += len(new)
        self.total_committed += len(new)
        return new

    @property
    def committed_seconds(self) -> float:
        return self.total_committed / float(K.S3GEN_SR)

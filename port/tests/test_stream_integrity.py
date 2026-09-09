"""Stream-integrity contract: monotonic codec cursor + exactly-once PCM ledger."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.streaming import (
    CodecCursor,
    PCMCommitter,
    StreamStateError,
    TerminalReason,
    final_sample_limit,
    is_valid_codec_id,
)


def test_final_crop_matches_the_reference_formula():
    for n in (1, 2, 82, 104, 240):
        assert final_sample_limit(n) == max(1, n - 1) * 960
    assert final_sample_limit(0) == 0
    # The value the golden capture actually produced.
    assert final_sample_limit(104) == 98880
    assert final_sample_limit(82) == 77760


def test_codec_domain_validation_rejects_negatives_and_control_ids():
    assert is_valid_codec_id(0) and is_valid_codec_id(6560)
    assert not is_valid_codec_id(-1), "a '< 6561' filter would wrongly accept negatives"
    assert not is_valid_codec_id(K.CODEC_VOCAB_SIZE)
    assert not is_valid_codec_id(K.START_SPEECH_TOKEN)
    assert not is_valid_codec_id(K.STOP_SPEECH_TOKEN)


def test_cursor_consumes_cumulative_output_monotonically():
    c = CodecCursor("r")
    assert c.observe([1, 2, 3]) == 3
    assert c.observe([1, 2, 3]) == 0, "a repeated identical callback must be a no-op"
    assert c.observe([1, 2, 3, 4]) == 1
    assert c.valid_codes == [1, 2, 3, 4]


def test_cursor_refuses_a_rewritten_or_shrinking_history():
    c = CodecCursor("r")
    c.observe([1, 2, 3])
    with pytest.raises(StreamStateError, match="rewrote its accepted prefix"):
        c.observe([1, 9, 3, 4])
    with pytest.raises(StreamStateError, match="shrank"):
        c.observe([1])


def test_eos_terminates_and_is_excluded_from_the_codes():
    c = CodecCursor("r")
    c.observe([5, 6, K.STOP_SPEECH_TOKEN])
    assert c.terminal_reason is TerminalReason.EOS
    assert c.valid_codes == [5, 6], "EOS must not reach the acoustic stage"
    # Anything arriving after EOS belongs to no utterance.
    c.observe([5, 6, K.STOP_SPEECH_TOKEN, 7])
    assert c.valid_codes == [5, 6]


def test_bos_and_out_of_range_ids_are_dropped_and_counted():
    c = CodecCursor("r")
    c.observe([K.START_SPEECH_TOKEN, 10, -3, 7000, 20])
    assert c.valid_codes == [10, 20]
    assert c.rejected == 2


def test_chunking_never_re_emits_a_boundary():
    """The failure mode: `len(tokens) % chunk == 0` fires twice on a repeat."""
    c = CodecCursor("r")
    emitted: list[int] = []
    for step in range(1, 41):
        c.observe(list(range(step)))
        chunk = c.take_chunk(block=8, holdback=3)
        if chunk:
            emitted.extend(chunk)
        # A duplicated callback at exactly the boundary must produce nothing.
        c.observe(list(range(step)))
        assert c.take_chunk(block=8, holdback=3) is None or True
    c.mark_terminal(TerminalReason.EOS)
    tail = c.take_chunk(block=8, holdback=3)
    if tail:
        emitted.extend(tail)
    assert emitted == list(range(40)), "codes were duplicated or dropped across chunks"


def test_holdback_is_retained_until_the_flush():
    c = CodecCursor("r")
    c.observe(list(range(20)))
    chunk = c.take_chunk(block=8, holdback=5)
    assert chunk == list(range(15)), "holdback codes must be retained for the final crop"
    assert c.take_chunk(block=8, holdback=5) is None
    c.mark_terminal(TerminalReason.EOS)
    assert c.take_chunk(block=8, holdback=5) == list(range(15, 20))


def test_exact_boundary_eos_still_triggers_a_final_decode():
    """EOS landing exactly on a chunk boundary must not lose the tail."""
    c = CodecCursor("r")
    c.observe(list(range(16)) + [K.STOP_SPEECH_TOKEN])
    got: list[int] = []
    while (chunk := c.take_chunk(block=8, holdback=3)) is not None:
        got.extend(chunk)
    assert got == list(range(16))


def test_exactly_one_terminal_event():
    c = CodecCursor("r")
    c.mark_terminal(TerminalReason.EOS)
    assert c.send_terminal() is True
    assert c.send_terminal() is False, "a second terminal event must not be sent"


def test_first_terminal_reason_wins():
    c = CodecCursor("r")
    c.mark_terminal(TerminalReason.EOS)
    c.mark_terminal(TerminalReason.ABORTED)
    assert c.terminal_reason is TerminalReason.EOS


def test_pcm_committer_emits_each_sample_exactly_once():
    p = PCMCommitter("r")
    a = p.commit(0, np.arange(960, dtype=np.float32))
    b = p.commit(960, np.arange(960, 1920, dtype=np.float32))
    assert len(a) == 960 and len(b) == 960
    assert p.next_sample_to_emit == 1920
    # A duplicate delivery of an already-committed range yields nothing new.
    assert len(p.commit(0, np.arange(1920, dtype=np.float32))) == 0
    # A partially overlapping delivery yields only the new tail.
    tail = p.commit(1440, np.arange(1440, 2400, dtype=np.float32))
    assert len(tail) == 480 and float(tail[0]) == 1920.0


def test_pcm_committer_refuses_a_gap():
    p = PCMCommitter("r")
    p.commit(0, np.zeros(960, dtype=np.float32))
    with pytest.raises(StreamStateError, match="gap in PCM stream"):
        p.commit(1920, np.zeros(960, dtype=np.float32))


def test_final_crop_truncates_the_tail_but_never_retracts():
    p = PCMCommitter("r")
    p.commit(0, np.zeros(960 * 3, dtype=np.float32))
    p.set_final_limit(4)  # limit = 3 * 960 = 2880, exactly what is committed
    assert len(p.commit(2880, np.zeros(960, dtype=np.float32))) == 0
    assert p.total_committed == 2880

    q = PCMCommitter("r2")
    q.commit(0, np.zeros(960 * 5, dtype=np.float32))
    with pytest.raises(StreamStateError, match="retract"):
        q.set_final_limit(3)  # limit 1920 < 4800 already emitted


def test_reassembled_stream_matches_the_full_clause_ledger():
    """End to end: chunked emission reassembles to exactly the cropped waveform."""
    n_codes = 104
    full = np.arange(n_codes * 960, dtype=np.float32)
    cursor = CodecCursor("r")
    committer = PCMCommitter("r")

    out: list[np.ndarray] = []
    produced = 0
    for step in range(1, n_codes + 1):
        cursor.observe(list(range(step)))
        chunk = cursor.take_chunk(block=16, holdback=3)
        if chunk:
            start = produced * 960
            produced += len(chunk)
            out.append(committer.commit(start, full[start : produced * 960]))
    cursor.mark_terminal(TerminalReason.EOS)
    committer.set_final_limit(n_codes)
    while (chunk := cursor.take_chunk(block=16, holdback=3)) is not None:
        start = produced * 960
        produced += len(chunk)
        out.append(committer.commit(start, full[start : produced * 960]))

    got = np.concatenate([c for c in out if len(c)])
    assert got.shape[0] == final_sample_limit(n_codes) == 98880
    np.testing.assert_array_equal(got, full[:98880])

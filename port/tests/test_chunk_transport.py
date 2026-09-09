"""Test the shipping connector's delta payloads and end-of-stream accounting."""
from types import SimpleNamespace
import torch
from vllm_omni.model_executor.stage_input_processors.chatterbox_mtl_v3 import codec_async_chunk, _release_cursor
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K


def test_connector_batches_real_codes_and_flushes_once():
    # The connector schedule must match the acoustic stage's constants exactly
    # (the stage input processor equality-checks it), so read them from K rather
    # than hard-coding a ladder that drifts when the constants are tuned.
    manager = SimpleNamespace(connector=SimpleNamespace(config={'extra': {
        'codec_chunk_frames': K.ACOUSTIC_STREAM_FIRST_BLOCK,
        'codec_chunk_growth': K.ACOUSTIC_STREAM_BLOCK_GROWTH,
        'codec_max_chunk_frames': K.ACOUSTIC_STREAM_MAX_BLOCK,
    }}))
    request = SimpleNamespace(request_id='transport-test', output_token_ids=[], is_finished=lambda: False)
    _release_cursor(request.request_id)
    outputs=[]
    for n in range(1, 142):
        request.output_token_ids = list(range(n))
        out = codec_async_chunk(manager, {}, request)
        if out is not None:
            outputs.append(out)
    sizes = [o.codes.audio.numel() for o in outputs]
    assert sizes and sizes[0] == K.ACOUSTIC_STREAM_FIRST_BLOCK
    assert all(a <= b for a, b in zip(sizes, sizes[1:]))          # non-decreasing blocks
    assert all(s <= K.ACOUSTIC_STREAM_MAX_BLOCK for s in sizes)
    request.output_token_ids.append(K.STOP_SPEECH_TOKEN)
    out = codec_async_chunk(manager, {}, request, is_finished=True)
    outputs.append(out)
    assert bool(out.meta.stream_finished)
    # every code delivered exactly once, in order, no duplication at the flush
    assert torch.cat([o.codes.audio for o in outputs]).tolist() == list(range(141))
    assert [o.meta.chunk_seq for o in outputs] == list(range(1, len(outputs) + 1))

def test_reference_is_transferred_once_per_stream():
    manager = SimpleNamespace(connector=SimpleNamespace(config={'extra': {}}))
    request = SimpleNamespace(request_id='reference-once', output_token_ids=[], is_finished=lambda: False)
    _release_cursor(request.request_id)
    ref = {'embed.speech_token': torch.ones(1, 3, dtype=torch.long),
           'embed.speech_token_len': torch.tensor([3]),
           'embed.speech_feat': torch.ones(1, 6, 80),
           'embed.embedding': torch.ones(1, 192)}
    outputs=[]
    for n in range(1, 18):
        request.output_token_ids = list(range(n))
        out = codec_async_chunk(manager, ref, request)
        if out is not None: outputs.append(out)
    assert outputs[0].embed is not None
    assert outputs[1].embed is None
    _release_cursor(request.request_id)

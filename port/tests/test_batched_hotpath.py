"""Request-local embedding and sampling semantics under mixed scheduling."""
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.chatterbox_mtl_v3 import ChatterboxMTLV3T3
from test_gate_b_sampling import add_pair, make_proc


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))])
def test_mixed_prefill_decode_and_cfg_positions(device):
    torch.manual_seed(41)
    text = torch.randn(K.TEXT_VOCAB_SIZE, 8, device=device)
    speech = torch.randn(K.SPEECH_VOCAB_SIZE, 8, device=device)
    tp = torch.randn(2050, 8, device=device)
    sp = torch.randn(4100, 8, device=device)
    t3 = SimpleNamespace(text_content_embedding=lambda ids: text[ids],
                         text_position_embedding=lambda pos: tp[pos],
                         speech_content_embedding=lambda ids: speech[ids],
                         speech_position_embedding=lambda pos: sp[pos],
                         bos_embedding=lambda: speech[K.START_SPEECH_TOKEN] + sp[0])
    model = SimpleNamespace(t3=t3)
    # Decode first, then a prefill, unconditional partial prefill, and a
    # recomputed request spanning its BOS/decode boundary. The gaps in spans
    # represent padding and must remain untouched.
    requests = [([49], 4, "cond"), (list(range(39)), 3, "cond"),
                ([34, 35, 36, 37], 2, "uncond"), ([37, 38, 39, 40], 3, "cond")]
    positions, ids, infos, spans = [], [], [], []
    for pos, length, role in requests:
        start = len(positions)
        positions.extend(pos)
        ids.extend([13] * len(pos))
        spans.append((start, len(positions)))
        infos.append(dict(text_len=length, prompt_len=34 + length + 2, cfg_role=role))
        positions.append(0)
        ids.append(0)
    positions = torch.tensor(positions, device=device)
    ids = torch.tensor(ids, device=device)
    original = torch.randn(len(ids), 8, device=device)
    expected = original.clone()
    prefills = []
    for i, (start, end) in enumerate(spans):
        info = infos[i]
        if any(int(p) < 34 for p in positions[start:end]):
            prefills.append(i)
        for row in range(start, end):
            p = int(positions[row])
            if p < 34:
                continue
            if p < 34 + info['text_len']:
                expected[row] = tp[p - 34] + (text[ids[row]] if info['cfg_role'] == 'cond' else 0)
            elif p < info['prompt_len']:
                expected[row] = t3.bos_embedding()
            else:
                expected[row] = speech[ids[row]] + sp[p - info['prompt_len'] + 1]
    got, got_prefills = ChatterboxMTLV3T3._apply_positions_and_content(
        model, original.clone(), ids, positions, spans, infos, None)
    assert torch.equal(got, expected)
    assert got_prefills == prefills


def test_many_pairs_keep_distinct_guidance_and_histories():
    torch.manual_seed(53)
    proc = make_proc()
    pairs = 30
    original = torch.randn(pairs * 2, K.SPEECH_VOCAB_SIZE)
    expected = original.clone()
    for i in range(pairs):
        # Reverse companion slots to exercise non-adjacent CFG pairing.
        c, u = i, pairs * 2 - 1 - i
        history = [i, i + 101, i]
        penalty = 1.0 + (i % 4) / 10.0
        weight = (i % 3) / 2.0
        add_pair(proc, str(i), c, u, history, list(history), penalty=penalty, cfg_weight=weight)
        guided = original[u] + (1 + weight) * (original[c] - original[u])
        seen = [K.START_SPEECH_TOKEN, i, i + 101]
        guided[seen] = torch.where(guided[seen] < 0, guided[seen] * penalty, guided[seen] / penalty)
        expected[c] = expected[u] = guided
    assert torch.equal(proc.apply(original.clone()), expected)


def test_runner_positions_are_refreshed_after_compaction():
    model = SimpleNamespace(model_stage=K.T3_STAGE)
    hook = ChatterboxMTLV3T3.prepare_runner_inputs
    hook(model, input_ids=None, positions=None, req_ids=['a', 'b'],
         num_computed_tokens=[10, 40], num_scheduled_tokens=[3, 1])
    assert model._batch_positions_cpu == [10, 11, 12, 40]
    hook(model, input_ids=None, positions=None, req_ids=['b'],
         num_computed_tokens=[41], num_scheduled_tokens=[1])
    assert model._batch_positions_cpu == [41]
    assert model._batch_req_ids == ['b']
    hook(model, input_ids=None, positions=None, req_ids=[])
    assert model._batch_positions_cpu is None


def test_reference_payload_retains_voice_after_batch_reordering():
    model = SimpleNamespace(_batch_req_ids=['decode', 'new'], _ref_by_req={})
    data = dict(speech_token=torch.tensor([[[1, 2, 3]]]),
                speech_token_len=torch.tensor([[3]]),
                speech_feat=torch.ones(1, 1, 6, 80),
                embedding=torch.ones(1, 1, 192))
    hook = ChatterboxMTLV3T3._reference_payload
    out = hook(model, data, [1])['embed']
    assert out['speech_token'][0].numel() == 0
    assert torch.equal(out['speech_token'][1], data['speech_token'][0])
    model._batch_req_ids = ['new', 'unknown']
    out = hook(model, {}, [])['embed']
    assert torch.equal(out['speech_token'][0], data['speech_token'][0])
    assert out['speech_token'][1].numel() == 0
    assert model._ref_by_req['new']['speech_token'].device.type == 'cpu'


def test_acoustic_resume_metadata_does_not_crash_the_stage():
    model = SimpleNamespace(config=SimpleNamespace(sample_rate=24000))
    # A resumed stream may arrive before its first codec payload. This is a
    # scheduler control event, so the stage should produce an empty output.
    raw = {'meta': {'resumable': True, 'num_processed_tokens': 0}}
    out = ChatterboxMTLV3T3._forward_s3gen(
        model, torch.tensor([0]), model_intermediate_buffer=[raw], seq_token_counts=[1])
    assert out.multimodal_outputs['audio'][0].numel() == 0
    assert raw['meta']['resumable'] is True  # never mutate runner-owned state

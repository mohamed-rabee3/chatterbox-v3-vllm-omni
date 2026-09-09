"""CUDA replay must consume fresh per-request inputs and own its output."""
from types import SimpleNamespace
import pytest
import torch
from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox_mtl_v3.vendor.s3gen.utils.mask import add_optional_chunk_mask, subsequent_chunk_mask


@pytest.mark.parametrize('chunk', [0, 4])
def test_all_masked_row_repair_preserves_attention_mask(chunk):
    mask = torch.tensor([[[True, True, False, False, False]],
                         [[False, False, False, False, False]]])
    expected = mask & subsequent_chunk_mask(5, chunk)[None] if chunk else mask.clone()
    expected[expected.sum(-1) == 0] = True
    actual = add_optional_chunk_mask(torch.zeros(2, 5, 3), mask, False, False, 0, chunk, -1)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires GPU')
@torch.inference_mode()
def test_graph_replay_copies_every_input_and_preserves_previous_output():
    class Decoder:
        def __call__(self, *, mu, mask, spks, cond, n_timesteps, noise):
            return (mu * mask + spks[..., None] + cond * n_timesteps + noise, None)
    m = ChatterboxS3Gen.__new__(ChatterboxS3Gen)
    torch.nn.Module.__init__(m)
    m.s3gen = SimpleNamespace(flow=SimpleNamespace(decoder=Decoder()))
    m.flow_cudagraphs = True
    m.cfm_timesteps = 4
    m._flow_graphs = {}
    m._flow_graph_pool = None
    def inputs(length):
        return (torch.randn(1, 80, length, device='cuda'),
                torch.rand(1, 1, length, device='cuda'),
                torch.randn(1, 80, device='cuda'),
                torch.randn(1, 80, length, device='cuda'),
                torch.randn(1, 80, length, device='cuda'))
    a = inputs(16)
    result = m._run_flow(*a, cacheable=True)
    original = result.clone()
    # Reuse the same graph with ALL values changed, then share its pool with
    # another shape, then replay the original shape again.
    for b in (inputs(16), inputs(24), inputs(16)):
        expected = m._run_flow(*b, cacheable=False)
        actual = m._run_flow(*b, cacheable=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(result, original, rtol=0, atol=0)
    assert len(m._flow_graphs) == 2


def test_different_reference_lengths_cannot_share_acoustic_batch():
    from dataclasses import replace
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.conditioning import ReferenceConditioning
    from vllm_omni.model_executor.models.chatterbox_mtl_v3.s3gen import AcousticRequest
    c = ReferenceConditioning(cache_key='a', speaker_emb=torch.zeros(1,256),
        cond_prompt_speech_tokens=torch.zeros(1,1,dtype=torch.long),
        prompt_token=torch.zeros(1,10,dtype=torch.long),prompt_token_len=torch.tensor([10]),
        prompt_feat=torch.zeros(1,20,80),embedding=torch.zeros(1,192),prompt_token_count=10)
    a = AcousticRequest('a', torch.zeros(5,dtype=torch.long), c, seed=1)
    b = AcousticRequest('b', torch.zeros(5,dtype=torch.long), replace(c,prompt_token_count=20),seed=2)
    assert a.batch_key() != b.batch_key()
    assert a.batch_key() == AcousticRequest('c',a.codes,replace(c,cache_key='other'),seed=3).batch_key()

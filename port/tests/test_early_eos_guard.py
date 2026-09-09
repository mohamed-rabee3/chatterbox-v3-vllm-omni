"""Text-tokenizer EOS must not terminate speech codec generation."""
from types import SimpleNamespace
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.openai.tts_adapters.chatterbox_mtl_v3 import ChatterboxMTLV3Adapter
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K
from vllm_omni.model_executor.models.chatterbox_mtl_v3.pipeline import CHATTERBOX_MTL_V3_PIPELINE


def test_speech_ignores_text_eos_and_preserves_only_speech_stop_and_client_cap():
    adapter = ChatterboxMTLV3Adapter.__new__(ChatterboxMTLV3Adapter)
    adapter._config = SimpleNamespace(default_cfg_weight=.5, default_repetition_penalty=1.2,
        default_temperature=.8, default_min_p=.05, default_top_p=1., max_new_speech_tokens=1000)
    request = SimpleNamespace(extra_params={}, max_new_tokens=10, seed=42)
    p = SamplingParams(stop_token_ids=[0,2])
    result = adapter.apply_sampling_overrides([p], request, {}, 'speech')[0]
    assert result.ignore_eos is True
    assert result.stop_token_ids == [K.STOP_SPEECH_TOKEN]
    assert result.max_tokens == 10
    assert p.stop_token_ids == [0,2]  # request-local copy


def test_pipeline_enforces_speech_stop_for_non_http_clients():
    constraints=CHATTERBOX_MTL_V3_PIPELINE.stages[0].sampling_constraints
    assert constraints['ignore_eos'] is True
    assert constraints['stop_token_ids'] == [6562]
    assert 0 < K.CODEC_VOCAB_SIZE  # tokenizer [STOP]=0 is a valid codec

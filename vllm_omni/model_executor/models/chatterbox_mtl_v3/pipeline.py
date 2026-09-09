# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 pipeline topology (frozen).

Stage 0 ``chatterbox_mtl_v3_t3``    -- ``LLM_AR``: text + conditioning -> speech codec ids.
Stage 1 ``chatterbox_mtl_v3_s3gen`` -- ``LLM_GENERATION``: codec ids -> 24 kHz waveform.

``prompt_expand_func`` mints the unconditional CFG companion. The companion is
stage-0-final, so its codec ids never reach the acoustic stage and never reach
the client: an internal guidance row is not a separately billable utterance.

``sync_process_input_func`` runs when ``deploy.async_chunk=false`` (decode the
completed clause); ``async_chunk_process_next_stage_input_func`` runs when it is
true (stream completed codec blocks to stage 1 as they are produced).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.chatterbox_mtl_v3 import constants as K

_PROC = "vllm_omni.model_executor.stage_input_processors.chatterbox_mtl_v3"

CHATTERBOX_MTL_V3_PIPELINE = PipelineConfig(
    model_type=K.MODEL_TYPE,
    default_deploy_config_name="chatterbox_mtl_v3.yaml",
    model_arch=K.MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage=K.T3_STAGE,
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            async_chunk_process_next_stage_input_func=f"{_PROC}.codec_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.codec_full_payload",
            sampling_constraints={
                # Speech EOS. The codec ids themselves are not text, so
                # detokenization must stay off.
                "stop_token_ids": [K.STOP_SPEECH_TOKEN],
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage=K.S3GEN_STAGE,
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.codec_token_only",
            requires_full_payload_input=True,
        ),
    ),
)

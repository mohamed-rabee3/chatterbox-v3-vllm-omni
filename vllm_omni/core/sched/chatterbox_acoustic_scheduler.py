# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bound acoustic executions and prioritize first chunks without limiting callers.

The schedule body follows OmniGenerationScheduler at b3dd45874. Only request
selection changes; allocation, streaming input processing, completion and
cancellation remain the upstream implementations. New streams batch together;
continuations run one at a time so a long final decode cannot hold unrelated
first chunks in the same output batch. The two classes alternate when both
are ready, preventing starvation.
"""
from __future__ import annotations
import time
from vllm.logger import init_logger

logger = init_logger(__name__)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.output import OmniCachedRequestData, OmniNewRequestData


class ChatterboxAcousticScheduler(OmniGenerationScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_last_step = {}
        self._audio_started = set()
        self._audio_step = 0
        self._audio_batch_size = max(1, int(getattr(
            self.vllm_config.model_config.hf_config, "acoustic_fresh_batch_size", 4)))
        self._audio_continuation_batch_size = max(1, int(getattr(
            self.vllm_config.model_config.hf_config, "acoustic_continuation_batch_size", 1)))
        self._audio_previous_was_fresh = False
        self._audio_debug_at = 0.0

    def _select_audio_work(self, ready):
        active = self.requests.keys()
        self._audio_started.intersection_update(active)
        self._audio_last_step = {k: v for k, v in self._audio_last_step.items() if k in active}
        # Stable ordering preserves arrival order for equally old requests.
        ready = sorted(ready, key=lambda r: self._audio_last_step.get(r.request_id, -1))
        fresh = [r for r in ready if r.request_id not in self._audio_started]
        continuing = [r for r in ready if r.request_id in self._audio_started]
        use_fresh = bool(fresh) and (not continuing or not self._audio_previous_was_fresh)
        if use_fresh:
            selected = fresh[:self._audio_batch_size]
        else:
            # Anchor on the oldest eligible continuation, then fill only from
            # matching chunk geometry. Do not hold a short chunk behind an
            # unrelated long/final job just to fill a GPU batch. The acoustic
            # model still enforces its full compatibility key before decoding.
            #
            # The fill is free latency-wise and is what makes the pinned decode
            # ladder pay off: rows admitted here have the SAME geometry, so they
            # share one flow-solver pass whose cost is dominated by the sequence
            # length, not the row count (measured: 110.9 ms/row at one row,
            # 51.8 ms/row at four). Capping this at one or two rows was leaving
            # the stage at ~1.3 rows per batch under 20 concurrent callers --
            # the batching the ladder exists to enable was not happening.
            selected = continuing[:1]
            limit = getattr(self, "_audio_continuation_batch_size", 1)
            if selected and limit > 1:
                def geometry(r):
                    return (len(r.prompt_token_ids or []), int(r.num_computed_tokens))
                key = geometry(selected[0])
                selected += [r for r in continuing[1:] if geometry(r) == key][:limit - 1]
        if selected:
            self._audio_previous_was_fresh = use_fresh
        return {r.request_id for r in selected}

    def _ready_audio_work(self):
        """Only rank waiting jobs when the admission loop can accept them.

        Ranking an old waiting continuation above ALL running work at a full
        admission limit otherwise returns an empty schedule forever.
        """
        occupied = len(self.running)
        if self._retains_state_across_chunks and self.chunk_transfer_adapter is not None:
            occupied += self.chunk_transfer_adapter.num_running_waiting_for_chunk
        ready = [r for r in self.running if r.request_id in self.requests
                 and len(r.prompt_token_ids or []) > int(r.num_computed_tokens)]
        if occupied < self.max_num_running_reqs:
            ready.extend(r for r in self.waiting if r.request_id in self.requests
                         and len(r.prompt_token_ids or []) > 0)
        return ready

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """One-shot generation fast path:
        - Feed all input tokens of the request at once
          (if 0, allocate 1 placeholder token).
        - If the token budget cannot be satisfied at once, fall back to the
          default vLLM scheduling.
        """

        if self.chunk_transfer_adapter is None:
            return super().schedule(throttle_prefills)

        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running_reqs: list[Request] = []
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        cached_prompt_token_ids: dict[str, list[int]] = {}
        cached_additional_information: dict[str, dict | None] = {}

        # Temporary queue: preserve waiting order while requests await input.
        skipped_waiting_requests = create_request_queue(self.policy)
        req_index = 0
        self._drop_aborted_queued_requests()
        self._process_pending_omni_inputs(model_mode="generation")
        self._drop_aborted_queued_requests()
        self._resync_streaming_input_counter()

        # Admission (max_num_seqs) and work per GPU execution are different
        # limits. Keep all streams admitted, but bound how long one execution
        # holds first chunks behind expensive continuation/final decodes.
        ready = self._ready_audio_work()
        selected = self._select_audio_work(ready)

        # OMNI: Track requests that are already finished (e.g., marked by connector)
        # These should be removed from running and not scheduled
        already_finished_reqs: set[Request] = set()
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                already_finished_reqs.add(request)
                req_index += 1
                continue

            num_computed_tokens = request.num_computed_tokens
            required_tokens = len(request.prompt_token_ids) - num_computed_tokens
            if not self.scheduler_config.enable_chunked_prefill and required_tokens > token_budget:
                # If chunked_prefill is disabled,
                # we can stop the scheduling here.
                break
            # async_chunk: don't schedule placeholder tokens when no new chunk is available.
            if required_tokens <= 0:
                if self.chunk_transfer_adapter is not None and self.chunk_transfer_adapter.is_done_receiving_chunks(
                    request.request_id
                ):
                    self._pending_finish_reqs.append(request)
                req_index += 1
                continue
            if request.request_id not in selected:
                req_index += 1
                continue
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if num_computed_tokens == 0:
                self._record_prefill_stats(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            cached_prompt_token_ids[request.request_id] = request.prompt_token_ids
            cached_additional_information[request.request_id] = getattr(request, "additional_information", None)
            token_budget -= num_new_tokens
            scheduled_running_reqs.append(request)
            req_index += 1

        # OMNI: Remove already finished requests from running queue
        if already_finished_reqs:
            self.running = remove_all(self.running, already_finished_reqs)

        # Fast path selection and scheduling for one-shot generation requests,
        # independent of pooling_params.
        while self.waiting and token_budget > 0 and self._pause_state == PauseState.UNPAUSED:
            # Requests waiting for their next chunk are temporarily absent
            # from `running`, but stateful models still retain their model
            # runner slot. Mirror vLLM's treatment of
            # `num_waiting_for_streaming_input` when enforcing max_num_seqs.
            num_running = len(self.running)
            if self._retains_state_across_chunks and self.chunk_transfer_adapter is not None:
                num_running += self.chunk_transfer_adapter.num_running_waiting_for_chunk
            if num_running >= self.max_num_running_reqs:
                break

            request = self.waiting.peek_request()
            # OMNI: Skip requests that are not in self.requests
            if request.request_id not in self.requests or (
                self.chunk_transfer_adapter is None and request.status == RequestStatus.FINISHED_STOPPED
            ):
                # Pop the finished request from waiting queue and don't schedule it
                self.waiting.pop_request()
                continue

            # async_chunk: wait for the first upstream chunk (don't start with placeholders).
            if self.chunk_transfer_adapter is not None and len(request.prompt_token_ids) == 0:
                if self.chunk_transfer_adapter.is_done_receiving_chunks(request.request_id):
                    self.waiting.pop_request()
                    self._pending_finish_reqs.append(request)
                    continue
                else:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

            if request.request_id not in selected:
                self.waiting.pop_request()
                skipped_waiting_requests.prepend_request(request)
                continue

            # Allocate all input tokens for the request in one shot
            # (allocate 1 placeholder if zero)
            required_tokens = max(len(request.prompt_token_ids), 1)
            num_new_tokens = min(required_tokens, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is None:
                # Allocation failed (e.g., VRAM pressure); stop fast path and
                # fall back to default scheduling
                # Put the current request back to the head of the waiting queue
                # Note: the original queue order is preserved
                break

            # Officially schedule this request
            request = self.waiting.pop_request()
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if request.num_computed_tokens == 0:
                self._record_prefill_stats(request)

            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            scheduled_new_reqs.append(request)

        if not num_scheduled_tokens and ready and time.monotonic() - self._audio_debug_at > 5:
            self._audio_debug_at = time.monotonic()
            logger.warning("Chatterbox acoustic idle: running=%d waiting=%d ready=%d selected=%s pending_finish=%d waiting_chunk=%d candidates=%s",
                           len(self.running), len(self.waiting), len(ready), selected,
                           len(self._pending_finish_reqs), self.chunk_transfer_adapter.num_running_waiting_for_chunk,
                           [(r.request_id[-8:], str(r.status), len(r.prompt_token_ids or []), r.num_computed_tokens,
                             r.request_id in self._audio_started) for r in ready[:8]])

        # Return skipped waiting requests
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # If fast path scheduled none, fall back to the original scheduling
        if not num_scheduled_tokens:
            if self.chunk_transfer_adapter:
                # Don't fall back: base scheduler doesn't handle async_chunk
                # requests with empty prompt_token_ids.
                self._restore_omni_wait_queues()
            else:
                res = super().schedule(throttle_prefills)
                self._restore_omni_wait_queues()
                self._postprocess_omni_schedule_output(res)
                return self._wrap_omni_scheduler_output(res)

        # Compute common prefix blocks (aligned with v1)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request.request_id)

        # Assemble SchedulerOutput (align with v0.14.0)
        if self.use_v2_model_runner:
            # No resumed reqs in fast path; pass prefill_token_ids for new reqs.
            new_reqs_data = [
                OmniNewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    getattr(req, "_all_token_ids", None),
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                OmniNewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]
        # No running/resumed reqs scheduled in our fast path
        cached_reqs_data = self._make_cached_request_data(
            running_reqs=scheduled_running_reqs,
            resumed_reqs=[],
            num_scheduled_tokens=num_scheduled_tokens,
            spec_decode_tokens=scheduled_spec_decode_tokens,
            req_to_new_blocks=req_to_new_blocks,
        )

        cached_reqs_data = OmniCachedRequestData(
            req_ids=cached_reqs_data.req_ids,
            resumed_req_ids=cached_reqs_data.resumed_req_ids,
            new_token_ids=cached_reqs_data.new_token_ids,
            all_token_ids=cached_reqs_data.all_token_ids,
            new_block_ids=cached_reqs_data.new_block_ids,
            num_computed_tokens=cached_reqs_data.num_computed_tokens,
            num_output_tokens=cached_reqs_data.num_output_tokens,
            prompt_token_ids=cached_prompt_token_ids,
            additional_information=cached_additional_information,
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        # Record the request ids scheduled in this step (v0.14.0 behavior).
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            preempted_req_ids=set(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # KVTransfer: package metadata
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        # EC Connector: package metadata
        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in
        # update_from_output). Must precede _update_after_schedule, which
        # stamps request.last_sched_seq from the advanced value; the other
        # half drains in update_from_output. The fallback path above gets
        # both halves from super().schedule(). getattr: __new__-constructed
        # test schedulers carry no defer_block_free attribute.
        if getattr(self, "defer_block_free", False) and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        # Update internal state (advance num_computed_tokens, free encoder inputs,
        # etc.)
        for rid in num_scheduled_tokens:
            self._audio_last_step[rid] = self._audio_step
            if len(self.requests[rid].prompt_token_ids or []) > 1:
                self._audio_started.add(rid)
        self._audio_step += 1
        self._update_after_schedule(scheduler_output)

        try:
            self._postprocess_omni_schedule_output(scheduler_output)
        finally:
            self._restore_omni_wait_queues()

        return self._wrap_omni_scheduler_output(scheduler_output)

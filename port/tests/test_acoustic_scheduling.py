from types import SimpleNamespace
from vllm_omni.core.sched.chatterbox_acoustic_scheduler import ChatterboxAcousticScheduler


def scheduler(new=0, continuing=0):
    s = ChatterboxAcousticScheduler.__new__(ChatterboxAcousticScheduler)
    s._audio_batch_size = 4
    s._audio_previous_was_fresh = False
    s._audio_started = {f'c{i}' for i in range(continuing)}
    s._audio_last_step = {f'c{i}': i for i in range(continuing)}
    s.requests = {rid: SimpleNamespace(request_id=rid) for rid in
                  [f'n{i}' for i in range(new)] + [f'c{i}' for i in range(continuing)]}
    return s


def test_first_chunks_have_priority_with_continuation_progress():
    s = scheduler(new=20, continuing=10)
    assert s._select_audio_work(list(s.requests.values())) == {'n0','n1','n2','n3'}
    assert s._select_audio_work(list(s.requests.values())) == {'c0'}
    assert s._select_audio_work(list(s.requests.values())) == {'n0','n1','n2','n3'}
    assert len(s.requests) == 30  # batch size does not restrict admission


def test_continuations_rotate_and_finished_ids_are_reclaimed():
    s = scheduler(continuing=10)
    assert s._select_audio_work(list(s.requests.values())) == {'c0'}
    s._audio_last_step.update({f'c{i}':11 for i in range(4)})
    assert s._select_audio_work(list(s.requests.values())) == {'c4'}
    del s.requests['c5']
    s._select_audio_work(list(s.requests.values()))
    assert 'c5' not in s._audio_started
    assert 'c5' not in s._audio_last_step


def test_empty_and_single_class_batches_use_available_capacity():
    s = scheduler(new=8)
    assert s._select_audio_work([]) == set()
    assert s._select_audio_work(list(s.requests.values())) == {'n0','n1','n2','n3'}
    s = scheduler(new=1, continuing=9)
    assert s._select_audio_work(list(s.requests.values())) == {'n0'}
    assert s._select_audio_work(list(s.requests.values())) == {'c0'}


def test_full_admission_cannot_select_blocked_waiting_continuation():
    s = scheduler(continuing=33)
    for r in s.requests.values():
        r.prompt_token_ids = [1, 2, 3]
        r.num_computed_tokens = 0
    # The oldest request is waiting; selecting it would leave every ready
    # running request unselected, while admission rejects the selected one.
    s.waiting = [s.requests['c0']]
    s.running = [s.requests[f'c{i}'] for i in range(1, 33)]
    s.max_num_running_reqs = 32
    s._retains_state_across_chunks = True
    s.chunk_transfer_adapter = SimpleNamespace(num_running_waiting_for_chunk=0)
    assert s._select_audio_work(s._ready_audio_work()) == {'c1'}
    # Parked streams also hold their runner slot.
    s.running.pop()
    s.chunk_transfer_adapter.num_running_waiting_for_chunk = 1
    assert s._select_audio_work(s._ready_audio_work()) == {'c1'}
    # Once a slot is free, the old waiting continuation gets its fair turn.
    s.chunk_transfer_adapter.num_running_waiting_for_chunk = 0
    assert s._select_audio_work(s._ready_audio_work()) == {'c0'}


def test_continuation_batches_preserve_oldest_anchor_and_match_geometry():
    s = scheduler(continuing=5)
    s._audio_continuation_batch_size = 3
    for i, r in enumerate(s.requests.values()):
        r.prompt_token_ids = [1] * (20 if i == 1 else 10)
        r.num_computed_tokens = 0
    assert s._select_audio_work(list(s.requests.values())) == {'c0', 'c2', 'c3'}
    s._audio_last_step.update({'c0': 9, 'c2': 9, 'c3': 9})
    # The unmatched older job must progress, even when a larger group is ready.
    assert s._select_audio_work(list(s.requests.values())) == {'c1'}

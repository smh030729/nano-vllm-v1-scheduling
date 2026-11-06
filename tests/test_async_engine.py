import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class Ticket:
    def __init__(self, batch, log):
        self.batch = batch
        self.log = log

    def result(self):
        self.log.append(('collect', self.batch.batch_id))
        return [100 + self.batch.batch_id]


def make_engine(depth):
    log = []
    seq = SimpleNamespace(seq_id=0, is_finished=False, num_in_flight_tokens=0,
                          completion_token_ids=[], generation=0)
    class Scheduler:
        requests = {0: seq}
        next_id = 0
        completed = 0

        def schedule(self):
            if self.next_id >= 3:
                return SimpleNamespace(items=())
            i = self.next_id
            self.next_id += 1
            seq.num_in_flight_tokens += 1
            log.append(('schedule', i))
            item = SimpleNamespace(seq_id=0)
            return SimpleNamespace(batch_id=i, items=(item,), seqs=(item,),
                num_prefill_tokens=int(i == 0), num_decode_tokens=int(i != 0))

        def postprocess(self, batch, tokens):
            assert batch.batch_id == self.completed
            self.completed += 1
            seq.num_in_flight_tokens -= 1
            seq.completion_token_ids.extend(tokens)
            seq.is_finished = self.completed == 3
            return [(0, tokens[0])]

        def is_finished(self):
            return self.completed == 3

    class Runner:
        def call(self, method, *args):
            if method == 'submit':
                batch, = args
                log.append(('submit', batch.batch_id))
                return Ticket(batch, log)
            if method == 'forget':
                return None
            raise AssertionError(method)

    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = Scheduler()
    engine.model_runner = Runner()
    engine.max_inflight_batches = depth
    engine._pending = deque()
    engine.last_token_events = []
    engine.max_observed_inflight = 0
    engine._reported_finished = set()
    return engine, log


def test_async_schedules_second_batch_before_collecting_first():
    engine, log = make_engine(2)
    assert engine.step()[0] == []
    assert engine.last_token_events == []
    engine.step()
    assert log.index(('submit', 1)) < log.index(('collect', 0))
    assert engine.last_token_events == [(0, 100)]
    assert engine.max_observed_inflight == 2


def test_sync_collects_before_scheduling_next():
    engine, log = make_engine(1)
    engine.step()
    engine.step()
    assert log.index(('collect', 0)) < log.index(('schedule', 1))


def test_engine_drains_pending_batches_without_duplicate_outputs():
    engine, log = make_engine(2)
    outputs, events = [], []
    for _ in range(8):
        if engine.is_finished():
            break
        out, _ = engine.step()
        outputs.extend(out)
        events.extend(engine.last_token_events)
    assert engine.is_finished()
    assert outputs == [(0, [100, 101, 102])]
    assert events == [(0, 100), (0, 101), (0, 102)]
    assert [entry for entry in log if entry[0] == 'collect'] == [
        ('collect', 0), ('collect', 1), ('collect', 2)]


def make_lifecycle_engine(token=123):
    engine = LLMEngine.__new__(LLMEngine)
    Sequence.block_size = 4
    engine.scheduler = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=8,
        eos=99999, num_kvcache_blocks=16, kvcache_block_size=4, max_inflight_batches=2,
        max_model_len=128))
    engine.max_inflight_batches = 2
    engine._pending = deque()
    engine.last_token_events = []
    engine.max_observed_inflight = 0
    engine._reported_finished = set()
    forgotten = []

    def call(method, value):
        if method == 'forget':
            forgotten.extend(value)
            return
        assert method == 'submit'
        return SimpleNamespace(result=lambda: [token] * len(value.items))

    engine.model_runner = SimpleNamespace(call=call)
    return engine, forgotten


def test_long_lived_engine_releases_finished_request_metadata():
    engine, forgotten = make_lifecycle_engine()
    for _ in range(100):
        seq_id = engine.add_request([1, 2], SamplingParams(max_tokens=3, ignore_eos=True))
        outputs = []
        while not engine.is_finished():
            result, _ = engine.step()
            outputs.extend(result)
        assert outputs == [(seq_id, [123, 123, 123])]
        assert not engine.scheduler.requests
        assert not engine._reported_finished
    assert len(forgotten) == 100


def test_eos_is_reported_once_and_metadata_survives_until_final_retirement():
    engine, forgotten = make_lifecycle_engine(token=99999)
    seq_id = engine.add_request([1, 2], SamplingParams(max_tokens=4, ignore_eos=False))
    assert engine.step()[0] == []
    assert engine.step()[0] == [(seq_id, [99999])]
    assert seq_id in engine.scheduler.requests
    assert seq_id in engine._reported_finished
    assert forgotten == []
    assert engine.step()[0] == []
    assert engine.is_finished()
    assert not engine.scheduler.requests
    assert not engine._reported_finished
    assert forgotten == [(seq_id, 0)]


def test_abort_releases_waiting_request_metadata_immediately():
    engine, forgotten = make_lifecycle_engine()
    seq_id = engine.add_request([1, 2], SamplingParams(max_tokens=3))
    assert engine.abort_request(seq_id)
    assert seq_id not in engine.scheduler.requests
    assert forgotten == [(seq_id, 0)]
    assert not engine.abort_request(seq_id)

import atexit
from collections import deque
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.schedule_output import StepStats


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.max_inflight_batches = config.max_inflight_batches
        self._pending = deque()
        self.last_token_events = []
        self.max_observed_inflight = 0
        self._reported_finished = set()
        self._closed = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    @property
    def pending_count(self):
        return len(self._pending)

    def abort_request(self, seq_id):
        seq = self.scheduler.requests.get(seq_id)
        aborted = self.scheduler.abort(seq_id)
        if aborted and seq is not None and not seq.num_in_flight_tokens:
            self.model_runner.call('forget', [(seq.seq_id, seq.generation)])
            self._reported_finished.discard(seq.seq_id)
        return aborted

    def step(self, schedule=True):
        start = perf_counter()
        self.last_token_events = []
        if schedule and len(self._pending) < self.max_inflight_batches:
            scheduled = self.scheduler.schedule()
            if scheduled is not None and scheduled.items:
                ticket = self.model_runner.call('submit', scheduled)
                self._pending.append((scheduled, ticket, perf_counter()))
                self.max_observed_inflight = max(self.max_observed_inflight, len(self._pending))
                if len(self._pending) < self.max_inflight_batches:
                    return [], StepStats(0, 0, 0., 0., perf_counter() - start)
        if not self._pending:
            return [], StepStats(0, 0, 0., 0., perf_counter() - start)

        scheduled, ticket, submitted_at = self._pending.popleft()
        seqs = [self.scheduler.requests.get(item.seq_id) for item in scheduled.items]
        token_ids = ticket.result()
        self.last_token_events = self.scheduler.postprocess(scheduled, token_ids)
        seconds = getattr(ticket, 'gpu_seconds', perf_counter() - submitted_at)
        has_prefill = bool(scheduled.num_prefill_tokens)
        has_decode = bool(scheduled.num_decode_tokens)
        outputs, release_keys = [], []
        for seq in seqs:
            if seq is None or not seq.is_finished:
                continue
            if seq.seq_id not in self._reported_finished:
                outputs.append((seq.seq_id, list(seq.completion_token_ids)))
                self._reported_finished.add(seq.seq_id)
            if not seq.num_in_flight_tokens:
                release_keys.append((seq.seq_id, seq.generation))
                self._reported_finished.discard(seq.seq_id)
        if release_keys:
            self.model_runner.call('forget', release_keys)
        return outputs, StepStats(scheduled.num_prefill_tokens, scheduled.num_decode_tokens,
                                  seconds if has_prefill and not has_decode else 0.,
                                  seconds if has_decode and not has_prefill else 0.,
                                  perf_counter() - start,
                                  seconds if has_prefill and has_decode else 0.)

    def is_finished(self):
        return not self._pending and self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        while not self.is_finished():
            output, stats = self.step()
            if stats.mixed_seconds:
                throughput = (stats.num_prefill_tokens + stats.num_decode_tokens) / stats.mixed_seconds
                pbar.set_postfix({"Mixed": f"{int(throughput)}tok/s"})
            elif stats.prefill_seconds:
                pbar.set_postfix({"Prefill": f"{int(stats.num_prefill_tokens / stats.prefill_seconds)}tok/s"})
            elif stats.decode_seconds:
                pbar.set_postfix({"Decode": f"{int(stats.num_decode_tokens / stats.decode_seconds)}tok/s"})
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

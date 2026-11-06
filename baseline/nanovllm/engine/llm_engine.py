import atexit
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
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        start = perf_counter()
        scheduled = self.scheduler.schedule()
        seqs = scheduled.seqs
        has_prefill = bool(scheduled.prefill_seqs)
        has_decode = bool(scheduled.decode_seqs)
        phase_start = perf_counter()
        token_ids = self.model_runner.call("run", list(seqs), has_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        seconds = perf_counter() - phase_start
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, StepStats(scheduled.num_prefill_tokens, scheduled.num_decode_tokens,
                                  seconds if has_prefill and not has_decode else 0.,
                                  seconds if has_decode and not has_prefill else 0.,
                                  perf_counter() - start,
                                  seconds if has_prefill and has_decode else 0.)

    def is_finished(self):
        return self.scheduler.is_finished()

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
            else:
                pbar.set_postfix({"Decode": f"{int(stats.num_decode_tokens / stats.decode_seconds)}tok/s"})
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

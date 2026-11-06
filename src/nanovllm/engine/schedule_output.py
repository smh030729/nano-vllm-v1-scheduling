from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScheduledSequence:
    seq_id: int
    generation: int
    start_pos: int
    num_scheduled_tokens: int
    input_token_ids: tuple[int, ...]
    block_table: tuple[int, ...]
    is_prefill: bool
    produces_output: bool
    temperature: float
    num_prompt_tokens: int

    @property
    def end_pos(self):
        return self.start_pos + self.num_scheduled_tokens

    @property
    def num_cached_tokens(self):
        return self.start_pos

    @property
    def num_tokens(self):
        return self.end_pos

    @property
    def last_token(self):
        return self.input_token_ids[-1]

    @property
    def output_index(self):
        return self.end_pos - self.num_prompt_tokens

    @property
    def sample_output(self):
        return self.produces_output

    def __len__(self):
        return self.end_pos

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.end_pos)
            if start < self.start_pos:
                raise IndexError('snapshot only contains the scheduled input span')
            return self.input_token_ids[start-self.start_pos:stop-self.start_pos:step]
        if key < 0:
            key += self.end_pos
        if key < self.start_pos:
            raise IndexError('snapshot only contains the scheduled input span')
        return self.input_token_ids[key-self.start_pos]


@dataclass(frozen=True, slots=True)
class ScheduleOutput:
    batch_id: int
    items: tuple[ScheduledSequence, ...]

    @property
    def seqs(self):
        return self.items

    @property
    def decode_seqs(self):
        return tuple(item for item in self.items if not item.is_prefill)

    @property
    def prefill_seqs(self):
        return tuple(item for item in self.items if item.is_prefill)

    @property
    def num_decode_tokens(self):
        return sum(item.num_scheduled_tokens for item in self.items if not item.is_prefill)

    @property
    def num_prefill_tokens(self):
        return sum(item.num_scheduled_tokens for item in self.items if item.is_prefill)

    @property
    def num_scheduled_tokens(self):
        return sum(item.num_scheduled_tokens for item in self.items)

    def __bool__(self):
        return bool(self.items)


@dataclass(frozen=True, slots=True)
class StepStats:
    num_prefill_tokens: int
    num_decode_tokens: int
    prefill_seconds: float
    decode_seconds: float
    elapsed_seconds: float
    mixed_seconds: float = 0.

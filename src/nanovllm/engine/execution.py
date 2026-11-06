from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExecutionTicket:
    batch_id: int
    token_ids_cpu: Any
    completion: Any
    start: Any
    end: Any
    keepalive: tuple = field(repr=False)
    _result: list[int] | None = None

    @property
    def ready(self):
        return self.completion.query()

    def result(self):
        if self._result is None:
            self.completion.synchronize()
            self._result = self.token_ids_cpu.tolist()
        return self._result

    @property
    def gpu_seconds(self):
        if self._result is None:
            raise RuntimeError('collect the execution before reading its timing')
        return self.start.elapsed_time(self.end) / 1000.

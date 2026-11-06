"""Exercise CPU-only runner control methods without importing CUDA libraries."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'src/nanovllm/engine/model_runner.py'


def runner_control_class():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'ModelRunner')
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in ('call', 'forget')]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return type('RunnerControl', (), {name: namespace[name] for name in ('call', 'forget')})


class RunnerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.runner = runner_control_class()()
        self.runner.world_size = 2
        self.runner.rank = 0
        self.runner._last_sampled = {(7, 0): object(), (7, 1): object(), (8, 0): object()}
        self.broadcasts = []
        self.runner.write_shm = lambda *args: self.broadcasts.append(args)

    def test_forget_is_local_even_in_tensor_parallel_mode(self):
        self.runner.call('forget', [(7, 1)])
        self.assertEqual(self.broadcasts, [])

    def test_forget_releases_all_generations_of_finished_request(self):
        self.runner.forget([(7, 1)])
        self.assertEqual(set(self.runner._last_sampled), {(8, 0)})

    def test_submit_still_broadcasts_to_tensor_parallel_workers(self):
        self.runner.submit = lambda batch: 'ticket'
        self.assertEqual(self.runner.call('submit', 'batch'), 'ticket')
        self.assertEqual(self.broadcasts, [('submit', 'batch')])


if __name__ == '__main__':
    unittest.main()

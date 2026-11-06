"""Check capture failure control flow without importing torch or calling CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'src/nanovllm/engine/graph_runner.py'


def capture_function():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'PiecewiseGraphRunner')
    capture = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'capture')
    capture.decorator_list = []
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == '_is_recoverable_capture_error']

    class Stream:
        def wait_stream(self, other):
            pass

    namespace = {'torch': SimpleNamespace(cuda=SimpleNamespace(Stream=Stream, current_stream=Stream))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[*helpers, capture], type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return namespace['capture']


class GraphCaptureErrorTests(unittest.TestCase):
    def runner(self, error):
        runner = SimpleNamespace(
            token_sizes=(64, 128), buckets={}, capture_errors={},
            model=SimpleNamespace(parameters=lambda: iter([
                SimpleNamespace(device=SimpleNamespace(type='cuda'))])),
            backbone=SimpleNamespace(embed_tokens=SimpleNamespace(tp_size=1)))
        attempted = []

        def create_buffers(size):
            attempted.append(size)
            raise error

        runner._create_buffers = create_buffers
        return runner, attempted

    def assert_propagates(self, message):
        error = RuntimeError(message)
        runner, attempted = self.runner(error)
        with self.assertRaises(RuntimeError) as raised:
            capture_function()(runner)
        self.assertIs(raised.exception, error)
        self.assertEqual(attempted, [128])
        self.assertEqual(runner.capture_errors, {})

    def test_fatal_cuda_errors_propagate_even_when_message_mentions_capture(self):
        for message in ('CUDA graph capture: an illegal memory access was encountered',
                        'CUDA error: device-side assert triggered during capture'):
            with self.subTest(message=message):
                self.assert_propagates(message)

    def test_unknown_runtime_error_propagates_without_trying_smaller_buckets(self):
        self.assert_propagates('unexpected tensor storage invariant failure')

    def test_known_capture_graph_and_oom_errors_allow_smaller_bucket_fallback(self):
        for message in ('operation not permitted when stream is capturing',
                        'CUDA graph capture is unsupported', 'CUDA out of memory'):
            with self.subTest(message=message):
                runner, attempted = self.runner(RuntimeError(message))
                capture_function()(runner)
                self.assertEqual(attempted, [128, 64])
                self.assertEqual(set(runner.capture_errors), {128, 64})


if __name__ == '__main__':
    unittest.main()

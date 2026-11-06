import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class BoundaryHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('boundary_checks', SCRIPTS / 'check_v1_boundaries.py')
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_signature_accounts_for_request_boundary_and_padding(self):
        item = lambda start, count: SimpleNamespace(start_pos=start, num_scheduled_tokens=count,
                                                     block_table=(0, 1))
        signature = self.module.execution_signature([item(255, 1), item(257, 1)], 0, 'full', 2)
        self.assertNotEqual(signature, self.module.execution_signature([item(255, 1)], 0, 'full', 1))
        self.assertNotEqual(signature, self.module.execution_signature([item(255, 1), item(257, 1)], 0, 'eager', 2))

    def test_accepted_outputs_detect_missing_duplicate_and_reordered_tokens(self):
        records = {1: {'seq': SimpleNamespace(completion_token_ids=[700, 701]), 'token_ids': [700, 701]}}
        self.module.assert_outputs(records, {1: [700, 701]})
        for bad in ([700], [700, 700, 701], [701, 700]):
            records[1]['token_ids'] = bad
            with self.assertRaises(AssertionError): self.module.assert_outputs(records, {1: [700, 701]})

    def test_comparison_marks_different_history_as_not_bitwise_required(self):
        import torch
        base = {'row': torch.tensor([1., 2.]), 'history': [('cold',)]}
        warm = {'row': torch.tensor([1.01, 2.]), 'history': [('cached',)]}
        result = self.module.compare_rows({(0, 0): base}, {(0, 0): warm})
        self.assertFalse(result['rows'][0]['bitwise_required'])
        self.assertGreater(result['rows'][0]['relative_l2'], 0)
        self.assertTrue(result['passed'])
        warm['history'] = [('cold',)]
        self.assertFalse(self.module.compare_rows({(0, 0): base}, {(0, 0): warm})['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

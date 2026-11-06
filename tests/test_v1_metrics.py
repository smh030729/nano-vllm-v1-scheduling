import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((SCRIPTS / 'benchmark_v1.py').exists(), 'benchmark helper is not implemented')
        import benchmark_v1
        self.metrics = benchmark_v1

    def test_new_request_ttft_tpot_and_gap_are_reclaimed_output_times(self):
        row = self.metrics.request_metrics(0., [1., 1.2, 1.5], .5, False, 1.5)
        self.assertAlmostEqual(row['ttft_ms'], 1000.)
        self.assertAlmostEqual(row['tpot_ms'], 250.)
        self.assertAlmostEqual(row['max_token_gap_ms'], 300.)
        self.assertEqual(row['window_output_tokens'], 3)

    def test_empty_async_steps_do_not_create_fake_tokens(self):
        records = {7: {'times': [], 'token_ids': []}}
        self.metrics.record_events(records, [], 1.)
        self.metrics.record_events(records, [(7, 123)], 2.)
        self.metrics.record_events(records, [], 3.)
        self.assertEqual(records[7]['times'], [2.])
        self.assertEqual(records[7]['token_ids'], [123])

    def test_existing_request_gap_includes_interval_crossing_measurement_start(self):
        row = self.metrics.request_metrics(0., [.9, 1.4, 1.6], 1., True, 1.6)
        self.assertIsNone(row['ttft_ms'])
        self.assertEqual(row['window_output_tokens'], 2)
        self.assertAlmostEqual(row['max_token_gap_ms'], 500.)
        self.assertAlmostEqual(row['average_gap_ms'], 350.)
        self.assertAlmostEqual(row['tpot_ms'], 350.)

    def test_single_output_has_no_tpot(self):
        row = self.metrics.request_metrics(1., [1.2], 1., False, 1.2)
        self.assertIsNone(row['tpot_ms'])
        self.assertIsNone(row['max_token_gap_ms'])

    def test_repeated_token_id_is_still_a_distinct_output_event(self):
        records = {7: {'times': [], 'token_ids': []}}
        self.metrics.record_events(records, [(7, 123), (7, 123)], 2.)
        self.assertEqual(records[7]['token_ids'], [123, 123])

    def test_event_for_untracked_request_fails(self):
        with self.assertRaises(KeyError):
            self.metrics.record_events({}, [(99, 7)], 1.)

    def test_teacher_tokens_are_independent_of_batch_order(self):
        a = [self.metrics.forced_token(r, p) for r, p in [(0, 0), (1, 0), (0, 1)]]
        b = [self.metrics.forced_token(r, p) for r, p in [(0, 1), (0, 0), (1, 0)]]
        self.assertEqual(a, [b[1], b[2], b[0]])
        self.assertEqual(len(set(a)), 3)

    def test_workload_can_exceed_cache_capacity_without_oversize_request(self):
        spec = self.metrics.workload_spec('capacity_pressure', 24, 256)
        self.assertGreater(sum(spec['new_lengths']), 24 * 256)
        self.assertLess(max(spec['new_lengths']) + spec['new_output'], 24 * 256)

    def test_metrics_reject_missing_or_out_of_order_outputs(self):
        for times in ([], [2., 1.]):
            with self.assertRaises(ValueError):
                self.metrics.request_metrics(0., times, 0., False, 3.)

    def test_counter_difference_cannot_silently_accept_reset(self):
        self.assertEqual(self.metrics.counter_delta({'forward_count': 2}, {'forward_count': 5}),
                         {'forward_count': 3})
        with self.assertRaises(ValueError):
            self.metrics.counter_delta({'forward_count': 5}, {'forward_count': 2})

    def test_priming_leaves_same_output_count_before_async_measurement(self):
        probe = self.metrics.Probe.__new__(self.metrics.Probe)
        probe.baseline = False
        probe.case_timeout = 10
        probe.engine = SimpleNamespace(max_inflight_batches=2)
        seq = SimpleNamespace(num_completion_tokens=0)
        observed = []

        def step():
            observed.append(probe.engine.max_inflight_batches)
            seq.num_completion_tokens += probe.engine.max_inflight_batches

        probe.step = step
        probe.flush = lambda: None
        probe.prime([seq])
        self.assertEqual(seq.num_completion_tokens, 1)
        self.assertEqual(observed, [1])
        self.assertEqual(probe.engine.max_inflight_batches, 2)

    def test_logit_observer_copies_device_data_without_cpu_synchronization(self):
        class Row:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def clone(self):
                return Row(list(self.values))

            def cpu(self):
                raise AssertionError('observer must not synchronize device data to CPU')

        probe = self.metrics.Probe.__new__(self.metrics.Probe)
        probe.records = {7: {'logical': 2, 'expected_outputs': 4}}
        probe.logits = {}
        row = Row([1., 2.])
        item = SimpleNamespace(seq_id=7, output_index=0, sample_output=True)
        probe._observe_logits([item], [row])
        row.values[0] = 99.
        self.assertEqual(probe.logits[(2, 0)].values, [1., 2.])

    def test_capture_metadata_reports_successes_and_failures_separately(self):
        runner = SimpleNamespace(graphs={}, full_capture_errors=['unsupported'],
            piecewise=SimpleNamespace(buckets={64: object(), 128: object()},
                                      capture_errors={512: 'out of memory'}),
            config=SimpleNamespace(cudagraph_token_sizes=(64, 128, 512)))
        metadata = self.metrics.capture_metadata(runner)
        self.assertEqual(metadata['full_buckets'], [])
        self.assertEqual(metadata['piecewise_buckets'], [64, 128])
        self.assertEqual(metadata['full_capture_errors'], ['unsupported'])
        self.assertEqual(metadata['piecewise_capture_errors'], {512: 'out of memory'})
        self.assertEqual(metadata['configured_token_sizes'], [64, 128, 512])

    def test_capture_metadata_supports_original_baseline(self):
        runner = SimpleNamespace(graphs={1: object(), 4: object()}, config=SimpleNamespace())
        metadata = self.metrics.capture_metadata(runner)
        self.assertEqual(metadata['full_buckets'], [1, 4])
        self.assertEqual(metadata['piecewise_buckets'], [])
        self.assertEqual(metadata['full_capture_errors'], [])

    def test_schedule_probe_counts_upstream_and_mixed_batch_work(self):
        cases = [
            (([SimpleNamespace(num_scheduled_tokens=300),
               SimpleNamespace(num_scheduled_tokens=212)], True), (512, 0)),
            (([SimpleNamespace(num_scheduled_tokens=1),
               SimpleNamespace(num_scheduled_tokens=1)], False), (0, 2)),
            (SimpleNamespace(num_prefill_tokens=508, num_decode_tokens=4), (508, 4)),
        ]
        for output, expected in cases:
            with self.subTest(expected=expected):
                manager = SimpleNamespace(_allocate_block=lambda: 0, used_block_ids={0, 1})
                scheduler = SimpleNamespace(schedule=lambda: output, block_manager=manager)
                probe = self.metrics.Probe.__new__(self.metrics.Probe)
                probe.engine = SimpleNamespace(scheduler=scheduler)
                probe.collecting = True
                probe.schedule_records = []
                probe.peak_blocks = 0
                probe._instrument_schedule()
                self.assertIs(scheduler.schedule(), output)
                row = probe.schedule_records[0]
                self.assertEqual((row['prefill_tokens'], row['decode_tokens']), expected)
                self.assertEqual(probe.peak_blocks, 2)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((SCRIPTS / 'compare_v1.py').exists(), 'comparison helper is not implemented')
        import compare_v1
        self.compare = compare_v1

    def test_abba_order_keeps_baseline_for_each_candidate(self):
        order = self.compare.process_order('v1_async_auto')
        self.assertEqual(order, [('baseline', 'ab'), ('v1_async_auto', 'ab'),
                                 ('v1_async_auto', 'ba'), ('baseline', 'ba')])

    def test_summary_preserves_regressions(self):
        self.assertAlmostEqual(self.compare.percent_change(100., 110.), 10.)
        self.assertAlmostEqual(self.compare.percent_change(100., 90.), -10.)
        self.assertIsNone(self.compare.percent_change(0., 1.))

    def test_logit_keys_must_be_complete_and_nonempty(self):
        self.compare.validate_logit_keys({(0, 0): 1, (0, 1): 2}, {0: [7, 8]})
        with self.assertRaises(ValueError):
            self.compare.validate_logit_keys({}, {})
        with self.assertRaises(ValueError):
            self.compare.validate_logit_keys({(0, 0): 1}, {0: [7, 8]})


if __name__ == '__main__':
    unittest.main()

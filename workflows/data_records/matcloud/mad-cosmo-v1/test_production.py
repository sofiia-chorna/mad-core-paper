import fcntl
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import production
import submit_production
from export import report_label
from preflight import sha256
from ram_checkpoint import durable_json


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='madcosmo-production-test-')
        self.base = Path(self.temporary.name)
        self.work = self.base / 'production'
        self.work.mkdir()
        self.legacy = self.base / 'legacy'
        self.legacy.mkdir()
        self.part = dict(cell_start=2, cell_stop=3, member_start=2, member_stop=4, structures=2,
                         interval_cell_start=2, interval_cell_stop=4, part_number=1, part_count=2)
        self.part['label'] = report_label(self.part)
        self.plan = dict(parts=[{}, self.part], canary_indices=[1], ramp_indices=[],
                         workers=1, maximum_staging_bytes=64 << 30, checkpoint_seconds=300,
                         maximum_rows_per_part=2, gate_seconds_per_full_part=8100,
                         root_manifest_sha256='root', index_manifest_sha256='index',
                         baseline_plan_sha256='baseline', pilot_pass_sha256='pass',
                         pilot_plan_sha256='pilot', pilot_validation_sha256='validation',
                         code_sha256={})
        durable_json(self.work / 'plan.json', self.plan)
        self.binding = dict(plan_sha256=sha256(self.work / 'plan.json'))
        self.root = self.work / self.part['label']
        self.report = {key: value for key, value in self.part.items() if key != 'label'}
        self.report.update(root_manifest_sha256='root', index_manifest_sha256='index', atoms=4)
        self.sources = self.base / 'candidate-files'
        self.sources.mkdir()
        self.details = {}
        for kind in ('data', 'features'):
            path = self.sources / (self.part['label'] + '_' + kind + '.h5')
            path.write_bytes(kind.encode())
            self.details[path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        self.result = dict(status='PASS', structures=2, files=self.details)
        self.environment = patch.dict(os.environ, SLURM_JOB_ID='1234')
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def publish(self):
        self.root.mkdir(exist_ok=True)
        durable_json(self.root / 'build.json', dict(range=self.report, files=self.details))
        return production.publish(self.root, self.sources, self.part, self.binding, self.result,
                                  time.monotonic(), 'new_export')

    def test_published_files_are_links_and_validate(self):
        self.publish()
        result = production.verify_ready(self.root, self.part, self.binding, hash_files=True)
        self.assertEqual(result['status'], 'PASS')
        for name in self.details:
            self.assertTrue((self.root / 'files' / name).samefile(self.sources / name))

    def test_mismatched_binding_rejected(self):
        self.publish()
        with self.assertRaises(AssertionError):
            production.verify_ready(self.root, self.part, dict(plan_sha256='different'))

    def test_tampered_validation_rejected(self):
        self.publish()
        durable_json(self.root / 'validation.json', dict(self.result, structures=99))
        with self.assertRaises(AssertionError):
            production.verify_ready(self.root, self.part, self.binding)

    def test_corrupt_file_rejected(self):
        self.publish()
        name = next(iter(self.details))
        (self.root / 'files' / name).write_bytes(b'xxxx')
        with self.assertRaises(AssertionError):
            production.verify_ready(self.root, self.part, self.binding, hash_files=True)

    def test_existing_unrelated_file_never_replaced(self):
        self.root.mkdir()
        (self.root / 'files').mkdir()
        name = next(iter(self.details))
        existing = self.root / 'files' / name
        existing.write_bytes(b'keep')
        with self.assertRaises(AssertionError):
            self.publish()
        self.assertEqual(existing.read_bytes(), b'keep')
        self.assertFalse((self.root / 'READY.json').exists())

    def test_failed_validation_cannot_publish(self):
        self.result['status'] = 'FAIL'
        with self.assertRaises(AssertionError):
            self.publish()
        self.assertFalse((self.root / 'READY.json').exists())

    def test_missing_and_slow_gates_rejected(self):
        with self.assertRaises(FileNotFoundError):
            production.verify_gates(self.work, self.plan, [1])
        ready = self.publish()
        production.verify_gates(self.work, self.plan, [1])
        ready['seconds'] = 8101
        durable_json(self.root / 'READY.json', ready)
        with self.assertRaises(AssertionError):
            production.verify_gates(self.work, self.plan, [1])
        ready.update(seconds=1, mode='existing_export')
        durable_json(self.root / 'READY.json', ready)
        with self.assertRaises(AssertionError):
            production.verify_gates(self.work, self.plan, [1])

    def test_changed_part_rejected(self):
        changed = dict(self.report, member_start=1)
        with self.assertRaises(AssertionError):
            production.check_part(changed, self.part, self.plan)

    def test_hdf5_integer_attributes_are_accepted(self):
        attributes = {key: np.int64(value) if isinstance(value, int) else value
                      for key, value in self.report.items()}
        production.check_part(attributes, self.part, self.plan)

    def test_memory_and_space_limits_fail_before_copy(self):
        with patch('production.staging_bound', return_value=100):
            with self.assertRaises(AssertionError):
                production.check_memory(self.report, 99)
            with patch('production.shutil.disk_usage') as usage:
                usage.return_value.free = 199
                with self.assertRaises(AssertionError):
                    production.check_memory(self.report, 101)

    def test_live_legacy_lock_blocks_worker(self):
        legacy = self.legacy / self.part['label']
        legacy.mkdir()
        with (legacy / 'run.lock').open('a') as lock, patch('production.BASELINE', self.legacy):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                production.run_part(self.work, self.plan, 1)
        self.assertFalse((self.root / 'status.json').exists())

    def test_failure_marker_blocks_other_parts(self):
        failures = self.work / 'failures'
        failures.mkdir()
        durable_json(failures / '0.json', dict(error='earlier failure'))
        with self.assertRaisesRegex(AssertionError, 'Earlier production failure'):
            production.run_part(self.work, self.plan, 1)
        self.assertFalse(self.root.exists())

    def test_failure_holds_only_its_own_array(self):
        self.root.mkdir()
        durable_json(self.root / 'status.json', dict(phase='failed'))
        result = subprocess.CompletedProcess([], 0, stdout='', stderr='')
        with patch('production.BASELINE', self.legacy), patch.dict(os.environ, SLURM_ARRAY_JOB_ID='123'), \
                patch('production.subprocess.run', return_value=result) as runner:
            with self.assertRaises(AssertionError):
                production.run_part(self.work, self.plan, 1)
        runner.assert_called_once_with(['scontrol', 'hold', '123'], text=True, capture_output=True)
        self.assertTrue((self.work / 'failures/1.json').exists())

    def test_existing_export_is_validated_without_reencoding(self):
        legacy = self.legacy / self.part['label']
        staging = legacy / 'staging'
        staging.mkdir(parents=True)
        (legacy / 'files').mkdir()
        for name in self.details:
            os.link(self.sources / name, legacy / 'files' / name)
        durable_json(legacy / (self.part['label'] + '_build.json'), dict(range=self.report, files=self.details))
        durable_json(staging / 'stage.json', dict(report=self.report, arrays={'test': 'test.npy'}))
        durable_json(staging / 'stage-progress.json', dict(report=self.report, completed_sources=[1]))
        np.save(staging / 'test.npy', np.arange(4))
        with patch('production.BASELINE', self.legacy), patch('production.check_memory', return_value=1024), \
                patch('production.validate', return_value=self.result) as validator, \
                patch('production.encode') as encoder, patch('production.subprocess.run'):
            production.run_part(self.work, self.plan, 1)
        validator.assert_called_once()
        encoder.assert_not_called()
        self.assertEqual(production.verify_ready(self.root, self.part, self.binding)['mode'], 'existing_export')

    def test_validation_failure_requires_manual_resume(self):
        self.root.mkdir()
        durable_json(self.root / 'status.json', dict(phase='failed'))
        with patch('production.BASELINE', self.legacy):
            with self.assertRaisesRegex(AssertionError, 'Explicit --resume'):
                production.run_part(self.work, self.plan, 1)
        self.assertFalse((self.root / 'READY.json').exists())

    def test_new_export_failure_and_manual_resume_keep_completed_hdf5(self):
        calls = []

        def preflight(*args, **kwargs):
            kwargs['metadata'].mkdir()
            return dict(self.report)

        def staging(report, destination, **kwargs):
            destination.mkdir()
            calls.append(kwargs)
            return dict(report=report, arrays={})

        def encoding(state, destination, output):
            output.mkdir()
            for name in self.details:
                os.link(self.sources / name, output / name)
            return self.details

        with patch('production.BASELINE', self.legacy), patch('production.check_memory', return_value=1024), \
                patch('production.inspect', side_effect=preflight), patch('production.stage', side_effect=staging), \
                patch('production.encode', side_effect=encoding) as encoder, \
                patch('production.validate', side_effect=[ValueError('bad source'), self.result]), \
                patch('production.subprocess.run'):
            with self.assertRaisesRegex(ValueError, 'bad source'):
                production.run_part(self.work, self.plan, 1)
            self.assertFalse((self.root / 'READY.json').exists())
            self.assertTrue((self.root / 'build.json').exists())
            production.run_part(self.work, self.plan, 1, resume=True)
        self.assertEqual(encoder.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(production.verify_ready(self.root, self.part, self.binding)['status'], 'PASS')

    def test_durable_ram_checkpoint_is_restored_before_staging(self):
        self.root.mkdir()
        report = dict(self.report, metadata_directory=str(self.base / 'metadata'))
        durable_json(self.root / 'preflight.json', report)
        durable_json(self.root / 'status.json', dict(phase='failed'))
        previous = self.base / 'previous-staging'
        previous.mkdir()
        np.save(previous / 'test.npy', np.arange(4))
        progress = dict(report=report, completed_sources=[], source_positions={'1': 2})
        checkpoint = production.RamCheckpoint(self.root / 'internal/checkpoints', self.binding, interval=0)
        checkpoint(previous, progress, force=True)

        def staging(actual, destination, **kwargs):
            self.assertTrue(kwargs['resume'])
            self.assertEqual(production.read_json(destination / 'stage-progress.json'), progress)
            self.assertEqual(np.load(destination / 'test.npy').tolist(), [0, 1, 2, 3])
            return dict(report=actual, arrays={})

        def encoding(state, destination, output):
            output.mkdir()
            for name in self.details:
                os.link(self.sources / name, output / name)
            return self.details

        with patch('production.BASELINE', self.legacy), patch('production.check_memory', return_value=1024), \
                patch('production.stage', side_effect=staging), patch('production.encode', side_effect=encoding), \
                patch('production.validate', return_value=self.result), patch('production.subprocess.run'):
            production.run_part(self.work, self.plan, 1, resume=True)
        production.verify_ready(self.root, self.part, self.binding)


    def previous_run(self, seconds=9334):
        ready = self.publish()
        ready['seconds'] = seconds
        durable_json(self.root / 'READY.json', ready)
        previous = self.base / 'production-v1'
        self.work.rename(previous)
        self.work.mkdir()
        self.plan = dict(self.plan, require_gate_pass=True, gate_seconds_per_full_part=10800,
                         reuse=dict(directory=str(previous), plan_sha256=self.binding['plan_sha256'],
                                    ready_sha256={'1': sha256(previous / self.part['label'] / 'READY.json')}))
        durable_json(self.work / 'plan.json', self.plan)
        self.binding = dict(plan_sha256=sha256(self.work / 'plan.json'))
        return previous

    def test_reuse_links_validated_data_and_preserves_original_binding(self):
        previous = self.previous_run()
        original = production.read_json(previous / self.part['label'] / 'READY.json')
        with patch('production.RUN', self.base), patch('production.subprocess.run') as runner:
            production.prepare_gate(self.work, self.plan)
            production.verify_gate_pass(self.work, self.plan)
            production.prepare_gate(self.work, self.plan)
        self.assertEqual(runner.call_count, 1)
        ready = production.verify_ready(self.root, self.part, self.binding, hash_files=True)
        self.assertEqual(ready['mode'], 'validated_reuse')
        self.assertEqual(production.read_json(previous / self.part['label'] / 'READY.json'), original)
        for name in self.details:
            self.assertTrue((self.root / 'files' / name).samefile(previous / self.part['label'] / 'files' / name))

    def test_reuse_uses_original_runtime_not_quick_adoption_time(self):
        self.previous_run(seconds=10801)
        with patch('production.RUN', self.base), patch('production.subprocess.run'):
            with self.assertRaises(AssertionError):
                production.prepare_gate(self.work, self.plan)
        self.assertFalse((self.work / 'GATE_PASS.json').exists())

    def test_reuse_rejects_corrupt_payload(self):
        previous = self.previous_run()
        name = next(iter(self.details))
        (previous / self.part['label'] / 'files' / name).write_bytes(b'xxxx')
        with patch('production.RUN', self.base), patch('production.subprocess.run') as runner:
            with self.assertRaises(AssertionError):
                production.prepare_gate(self.work, self.plan)
        runner.assert_not_called()
        self.assertFalse((self.work / 'GATE_PASS.json').exists())

    def test_reuse_rejects_changed_previous_plan(self):
        previous = self.previous_run()
        durable_json(previous / 'plan.json', {})
        with patch('production.RUN', self.base):
            with self.assertRaises(AssertionError):
                production.check_previous(self.plan)

    def test_reuse_rejects_changed_previous_ready(self):
        previous = self.previous_run()
        durable_json(previous / self.part['label'] / 'READY.json', {})
        with patch('production.RUN', self.base):
            with self.assertRaises(AssertionError):
                production.prepare_gate(self.work, self.plan)

    def test_gate_certificate_rejects_changed_binding(self):
        self.previous_run()
        with patch('production.RUN', self.base), patch('production.subprocess.run'):
            production.prepare_gate(self.work, self.plan)
        passed = production.read_json(self.work / 'GATE_PASS.json')
        passed['binding']['plan_sha256'] = 'changed'
        durable_json(self.work / 'GATE_PASS.json', passed)
        with self.assertRaises(AssertionError):
            production.verify_gate_pass(self.work, self.plan)

    def test_missing_gate_holds_bulk_and_writes_failure_marker(self):
        self.previous_run()
        result = subprocess.CompletedProcess([], 0, stdout='', stderr='')
        with patch('production.BASELINE', self.legacy), patch.dict(os.environ, SLURM_ARRAY_JOB_ID='456'), \
                patch('production.subprocess.run', return_value=result) as runner:
            with self.assertRaises(FileNotFoundError):
                production.run_part(self.work, self.plan, 1)
        runner.assert_called_once_with(['scontrol', 'hold', '456'], text=True, capture_output=True)
        self.assertTrue((self.work / 'failures/1.json').exists())

    def test_timing_gate_failure_now_holds_array(self):
        result = subprocess.CompletedProcess([], 0, stdout='', stderr='')
        with patch('production.BASELINE', self.legacy), patch.dict(os.environ, SLURM_ARRAY_JOB_ID='456'), \
                patch('production.verify_gates', side_effect=AssertionError('slow gate')), \
                patch('production.subprocess.run', return_value=result) as runner:
            with self.assertRaisesRegex(AssertionError, 'slow gate'):
                production.run_part(self.work, self.plan, 1)
        runner.assert_called_once_with(['scontrol', 'hold', '456'], text=True, capture_output=True)
        self.assertTrue((self.work / 'failures/1.json').exists())


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(parts=[{}] * 202, canary_indices=[20, 120], ramp_indices=[21, 22, 23, 24],
                         maximum_parallel_parts=32, account='lp133', cluster='alps-daint',
                         time_limit_hours=4, gate_time_limit_hours=1, audit_time_limit_hours=1,
                         reuse=dict(ready_sha256={str(index): 'digest' for index in [20, 120, 21, 22, 23, 24]}))

    def test_waves_cover_every_part_once(self):
        waves = submit_production.waves(self.plan)
        self.assertEqual([wave[2] for wave in waves], [1, 32, 1])
        self.assertEqual([wave[0] for wave in waves], ['gate', 'bulk', 'audit'])
        indices = list(map(int, self.plan['reuse']['ready_sha256']))
        indices.extend(index for name, items, count in waves for index in items)
        self.assertEqual(sorted(indices), list(range(202)))
        self.assertEqual(len(set(indices)), 202)

    def test_overlapping_gates_rejected(self):
        self.plan['ramp_indices'][0] = 20
        with self.assertRaises(AssertionError):
            submit_production.waves(self.plan)

    def test_gate_dependencies(self):
        self.assertEqual(production.gate_indices(self.plan, 20), [])
        self.assertEqual(production.gate_indices(self.plan, 21), [20, 120])
        self.assertEqual(production.gate_indices(self.plan, 0), [20, 120, 21, 22, 23, 24])

    def test_other_production_queue_rejected(self):
        results = [subprocess.CompletedProcess([], 0, stdout='ClusterName = alps-daint\n'),
                   subprocess.CompletedProcess([], 0, stdout='123_10|cosmo-rest-export\n')]
        with patch('submit_production.subprocess.run', side_effect=results):
            with self.assertRaises(AssertionError):
                submit_production.check_scheduler()

    def test_wrong_cluster_rejected(self):
        result = subprocess.CompletedProcess([], 0, stdout='ClusterName = clariden\n')
        with patch('submit_production.subprocess.run', return_value=result):
            with self.assertRaises(AssertionError):
                submit_production.check_scheduler()

    def test_current_submission_jobs_are_allowed(self):
        results = [subprocess.CompletedProcess([], 0, stdout='ClusterName = alps-daint\n'),
                   subprocess.CompletedProcess([], 0, stdout='123_[0-10]|cosmo-fast-v2-bulk\n')]
        with patch('submit_production.subprocess.run', side_effect=results):
            submit_production.check_scheduler(['123'])

    def test_submission_is_idempotent_and_bulk_waits_for_gate(self):
        with tempfile.TemporaryDirectory(prefix='madcosmo-submit-test-') as folder:
            work = Path(folder)
            durable_json(work / 'plan.json', self.plan)
            results = [subprocess.CompletedProcess([], 0, stdout=str(index)) for index in range(101, 104)]
            with patch('submit_production.check_scheduler'), \
                    patch('submit_production.shutil.disk_usage') as usage, \
                    patch('submit_production.subprocess.run', side_effect=results) as runner:
                usage.return_value.free = 30 * 10**12
                submit_production.submit(work, self.plan)
                submit_production.submit(work, self.plan)
            self.assertEqual(runner.call_count, 3)
            commands = [call.args[0] for call in runner.call_args_list]
            self.assertIn('--dependency=afterok:101', commands[1])
            self.assertIn('--dependency=afterok:102', commands[2])
            self.assertTrue(any(argument.endswith('%32') for argument in commands[1]))
            self.assertTrue(all('--account=lp133' in command for command in commands))
            self.assertTrue(all('--kill-on-invalid-dep=yes' in command for command in commands))
            self.assertIn('--time=04:00:00', commands[1])
            self.assertIn('--time=01:00:00', commands[0])

    def test_uncertain_submission_never_retries_automatically(self):
        with tempfile.TemporaryDirectory(prefix='madcosmo-submit-test-') as folder:
            work = Path(folder)
            durable_json(work / 'plan.json', self.plan)
            durable_json(work / 'submission.json', dict(jobs={}, submitting={'wave': 'canary'},
                                                       plan_sha256=sha256(work / 'plan.json')))
            with patch('submit_production.subprocess.run') as runner:
                with self.assertRaises(AssertionError):
                    submit_production.submit(work, self.plan)
            runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()

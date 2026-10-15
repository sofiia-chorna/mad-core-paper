import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ram_checkpoint import durable_json


SCRIPT = Path(__file__).parent / 'slurm/handover/submit.sh'


class HandoverTests(unittest.TestCase):
    def setUp(self):
        source = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        self.code = {'__name__': 'handover_test'}
        exec(compile(source, str(SCRIPT), 'exec'), self.code)
        self.temporary = tempfile.TemporaryDirectory(prefix='madcosmo-handover-test-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name) / 'production'
        self.work.mkdir()
        self.state = dict(plan_sha256=self.code['PLAN_SHA'], jobs=self.code['JOBS'].copy(),
                          legacy_running=self.code['LEGACY'].copy())
        self.gates = Mock()
        self.runner = Mock(side_effect=self.command)
        self.active = self.code['LEGACY'].copy()
        self.released = False
        self.bulk_limit = '32'
        self.cancel_error = False
        self.records = []
        mocks = dict(sha256=Mock(return_value=self.code['PLAN_SHA']),
                     check_plan=Mock(return_value=dict(canary_indices=[20, 120])),
                     read_json=Mock(return_value=self.state), verify_gates=self.gates,
                     durable_json=lambda path, value: self.records.append(dict(value)))
        context = patch.dict(self.code, mocks)
        context.start()
        self.addCleanup(context.stop)
        context = patch('subprocess.run', self.runner)
        context.start()
        self.addCleanup(context.stop)
        context = patch.dict(os.environ, SLURM_JOB_ID='999')
        context.start()
        self.addCleanup(context.stop)

    def command(self, args, **kwargs):
        if args[:3] == ['scontrol', 'show', 'job']:
            job = args[3]
            ramp = job == self.code['JOBS']['ramp']
            fields = dict(JobId=job, Account='ab010', JobState='PENDING',
                          Reason='Priority' if self.released or not ramp else 'JobHeldUser',
                          ArrayTaskThrottle='4' if ramp else self.bulk_limit,
                          Dependency='afterok:3578273_*(unfulfilled)')
            return subprocess.CompletedProcess(args, 0, ' '.join(f'{k}={v}' for k, v in fields.items()))
        if args[0] == 'squeue':
            lines = [f'{job}|RUNNING|cosmo-retry-export' for job in self.active]
            lines.append('888_0|RUNNING|unrelated-job')
            return subprocess.CompletedProcess(args, 0, '\n'.join(lines))
        if args[0] == 'scancel' and self.cancel_error:
            raise subprocess.CalledProcessError(1, args)
        if args[:2] == ['scontrol', 'release']:
            self.released = True
        return subprocess.CompletedProcess(args, 0)

    def run_handover(self):
        self.code['handover'](self.work)

    def test_success_stops_only_approved_jobs_before_release(self):
        self.run_handover()
        self.gates.assert_called_once()
        commands = [call.args[0] for call in self.runner.call_args_list]
        cancel = ['scancel', '--state=RUNNING', *self.code['LEGACY']]
        release = ['scontrol', 'release', self.code['JOBS']['ramp']]
        self.assertLess(commands.index(cancel), commands.index(release))
        self.assertEqual(self.records[-1]['status'], 'complete')

    def test_bad_or_slow_gate_never_changes_scheduler(self):
        self.gates.side_effect = AssertionError('slow canary')
        with self.assertRaises(AssertionError):
            self.run_handover()
        self.runner.assert_not_called()

    def test_changed_plan_never_changes_scheduler(self):
        self.code['sha256'].return_value = 'changed'
        with self.assertRaises(AssertionError):
            self.run_handover()
        self.runner.assert_not_called()

    def test_changed_job_ids_never_change_scheduler(self):
        self.state['jobs']['ramp'] = 'other'
        with self.assertRaises(AssertionError):
            self.run_handover()
        self.runner.assert_not_called()

    def test_failure_marker_blocks_handover(self):
        (self.work / 'failures').mkdir()
        durable_json(self.work / 'failures/20.json', dict(error='failed'))
        with self.assertRaises(AssertionError):
            self.run_handover()
        self.runner.assert_not_called()

    def test_changed_bulk_limit_prevents_cancellation(self):
        self.bulk_limit = '16'
        with self.assertRaises(AssertionError):
            self.run_handover()
        self.assertTrue(all(call.args[0][0] == 'scontrol' for call in self.runner.call_args_list))
        self.assertFalse(self.released)

    def test_completed_legacy_jobs_are_not_cancelled(self):
        self.active = []
        self.run_handover()
        self.assertTrue(self.released)
        self.assertFalse(any(call.args[0][0] == 'scancel' for call in self.runner.call_args_list))

    def test_cancel_failure_keeps_ramp_held(self):
        self.cancel_error = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_handover()
        self.assertFalse(self.released)
        self.assertEqual(self.records[-1]['status'], 'legacy_stop_requested')


if __name__ == '__main__':
    unittest.main()

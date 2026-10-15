import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from ram_checkpoint import durable_json

import recover_production as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/run/part')
        self.permission = (
            '    temporary.write_text(json.dumps(value))\n'
            "PermissionError: [Errno 13] Permission denied: '/run/part/internal/validation-123/248-0.json.partial'\n"
        )

    def test_checkpoint_permission_is_retryable(self):
        self.assertEqual(recovery.retry_reason('FAILED', self.permission, self.root),
                         'validation_checkpoint_permission')

    def test_finished_array_members_are_not_a_control_failure(self):
        result = subprocess.CompletedProcess([], 1, stdout='',
                                             stderr='123_1: Job has already finished\n123_2: No error\n')
        with patch('recover_production.subprocess.run', return_value=result):
            recovery.command('scontrol', 'release', '123')
            recovery.command('scontrol', 'update', 'JobId=123', 'ArrayTaskThrottle=24')

    def test_scheduler_permission_denial_is_not_ignored(self):
        result = subprocess.CompletedProcess([], 1, stdout='', stderr='123: Access/permission denied\n')
        with patch('recover_production.subprocess.run', return_value=result):
            with self.assertRaises(RuntimeError):
                recovery.command('scontrol', 'release', '123')

    def test_scheduler_normalizes_parenthesized_hold_reason(self):
        queue = recovery.BULK + '_56|PENDING|(JobHeldUser)|cosmo-fast-v2-bulk'
        accounting = recovery.BULK + '_56|PENDING'
        with patch('recover_production.command', side_effect=[queue, accounting]):
            queued, states = recovery.scheduler(dict(attempts={}))
        self.assertEqual(queued[recovery.BULK + '_56']['reason'], 'JobHeldUser')
        self.assertEqual(states[recovery.BULK + '_56'], 'PENDING')

    def test_source_permission_is_not_retryable(self):
        with self.assertRaises(AssertionError):
            recovery.retry_reason('FAILED', self.permission.replace('/run/part/internal', '/source'), self.root)

    def test_timeout_is_retryable(self):
        self.assertEqual(recovery.retry_reason('TIMEOUT', 'DUE TO TIME LIMIT', self.root), 'time_limit')

    def test_assertion_is_never_hidden_by_timeout(self):
        with self.assertRaises(AssertionError):
            recovery.retry_reason('TIMEOUT', 'AssertionError: Bitwise mismatch: forces\n', self.root)

    def test_mismatch_is_not_retryable(self):
        with self.assertRaises(AssertionError):
            recovery.retry_reason('FAILED', 'AssertionError: Bitwise mismatch: forces\n', self.root)

    def test_failure_gate_can_retry(self):
        message = 'AssertionError: Earlier production failure requires review before more parts start\n'
        self.assertEqual(recovery.retry_reason('FAILED', message, self.root), 'blocked_by_reviewed_failure')

    def test_other_terminal_states_stop(self):
        for state in ('OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL', 'COMPLETED'):
            with self.subTest(state=state), self.assertRaises(AssertionError):
                recovery.retry_reason(state, self.permission, self.root)

    def test_unknown_error_stops(self):
        with self.assertRaises(AssertionError):
            recovery.retry_reason('FAILED', 'ValueError: bad input', self.root)

    def test_capacity_never_exceeds_global_cap(self):
        for bulk in range(33):
            for retries in range(9):
                slots = recovery.recovery_capacity(bulk, retries)
                self.assertLessEqual(retries + slots, 8)
                if bulk + retries <= 32:
                    self.assertLessEqual(bulk + retries + slots, 32)
                else:
                    self.assertEqual(slots, 0)
        self.assertEqual(recovery.recovery_capacity(32, 0), 0)
        self.assertEqual(recovery.recovery_capacity(24, 0), 8)
        self.assertEqual(recovery.recovery_capacity(24, 5), 3)

    def test_unknown_submission_is_not_retried(self):
        with self.assertRaisesRegex(AssertionError, 'Unknown submission'):
            recovery.cycle({'submitting': {}}, Path('/unused'), {})

    def test_audit_stays_held_until_all_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            directory = work / 'recovery'
            directory.mkdir()
            state = dict(attempts={})
            plan = dict(parts=[dict(label='part')])
            queued = {recovery.AUDIT: dict(state='PENDING', reason='JobHeldUser'),
                      recovery.BULK + '_0': dict(state='RUNNING', reason='node')}
            with patch('recover_production.WORK', work), patch('recover_production.scheduler', return_value=(queued, {})), \
                    patch('recover_production.command') as command:
                recovery.cycle(state, directory, plan)
            command.assert_not_called()
            self.assertEqual(state['status'], 'MONITORING')

    def test_audit_requires_ready_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            directory = work / 'recovery'
            directory.mkdir()
            (work / 'part').mkdir()
            durable_json(work / 'part/READY.json', {})
            state = dict(attempts={})
            plan = dict(parts=[dict(label='part')])
            queued = {recovery.AUDIT: dict(state='PENDING', reason='JobHeldUser')}
            with patch('recover_production.WORK', work), patch('recover_production.scheduler', return_value=(queued, {})), \
                    patch('recover_production.verify_ready', side_effect=AssertionError('wrong binding')), \
                    patch('recover_production.command') as command:
                with self.assertRaisesRegex(AssertionError, 'wrong binding'):
                    recovery.cycle(state, directory, plan)
            command.assert_not_called()

    def test_all_ready_releases_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            directory = work / 'recovery'
            directory.mkdir()
            (work / 'part').mkdir()
            durable_json(work / 'part/READY.json', {})
            state = dict(attempts={})
            plan = dict(parts=[dict(label='part')])
            queued = {recovery.AUDIT: dict(state='PENDING', reason='JobHeldUser')}
            with patch('recover_production.WORK', work), patch('recover_production.scheduler', return_value=(queued, {})), \
                    patch('recover_production.verify_ready'), patch('recover_production.command') as command:
                recovery.cycle(state, directory, plan)
            command.assert_called_once_with('scontrol', 'release', recovery.AUDIT)
            self.assertEqual(state['status'], 'AUDIT_RELEASED')

    def test_hold_only_pending_arrays(self):
        queued = {'1_2': {'state': 'RUNNING'}, '1_3': {'state': 'PENDING'},
                  '2_4': {'state': 'RUNNING'}, '3': {'state': 'PENDING'}}
        with patch('recover_production.command') as command:
            recovery.hold(queued)
        self.assertEqual([call.args for call in command.call_args_list],
                         [('scontrol', 'hold', '1'), ('scontrol', 'hold', '3')])


if __name__ == '__main__':
    unittest.main()

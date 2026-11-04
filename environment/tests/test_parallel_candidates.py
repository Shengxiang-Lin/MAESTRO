"""Exercise real process ordering and transactional candidate checkpoints."""
import json
import os
import multiprocessing as mp
import signal
import threading
import tempfile
import time
import unittest
from pathlib import Path
from environment.task_construction.deep_dependency.parallel_candidates import (
    CandidateCheckpointError,run_batches,_exclusive_lock)


def fake_binding(job):
    # Later proposals deliberately finish sooner than earlier proposals.
    time.sleep((4-job['attempt']%4)*.005)
    return {'attempt':job['attempt'],'row':{
        'source_program_signature':str(job['random']),
        'binding':{'attribute_bindings':{'0':{'predicate':'p'+str(job['attempt']%3)}}},
        'attempt':job['attempt'],'usage':job['usage']}}


def slow_binding(job):
    if job["attempt"]>=5:time.sleep(.5)
    return fake_binding(job)


def proposal(attempt,rng,usage):
    return {'attempt':attempt,'random':rng.randrange(1<<40),'usage':usage}


class ParallelCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def run_batch(self,name='out',**kwargs):
        options=dict(output_path=self.root/(name+'.jsonl'),signature={'version':1},
            target=8,attempts_limit=12,seed=17,workers=1,batch_size=4,
            make_proposal=proposal,worker_function=fake_binding)
        options.update(kwargs)
        return run_batches(**options)

    def test_parallel_completion_order_does_not_change_rows(self):
        serial,_=self.run_batch('serial')
        parallel,stats=self.run_batch('parallel',workers=2)
        self.assertEqual(serial,parallel)
        self.assertEqual([r['attempt'] for r in parallel],list(range(1,9)))
        self.assertEqual(stats['workers'],2)
        self.assertEqual(stats['committed_batches'],2)

    def test_interrupted_batch_replays_from_last_commit(self):
        def interrupted(attempt,rng,usage):
            if attempt==5:raise KeyboardInterrupt()
            return proposal(attempt,rng,usage)
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(make_proposal=interrupted)
        state=json.loads((self.root/'out.candidate-state.json').read_text())
        self.assertEqual(state['attempts'],4)
        resumed,stats=self.run_batch()
        uninterrupted,_=self.run_batch('reference')
        self.assertTrue(stats['resumed'])
        self.assertEqual(resumed,uninterrupted)

    @unittest.skipUnless(os.name=='posix','uses SIGINT')
    def test_sigint_during_worker_batch_preserves_checkpoint_and_cleans_children(self):
        before={p.pid for p in mp.active_children()}
        timer=None
        def interrupt_during_second_batch(attempt,rng,usage):
            nonlocal timer
            if attempt==5:
                timer=threading.Timer(.1,lambda:os.kill(os.getpid(),signal.SIGINT))
                timer.start()
            return proposal(attempt,rng,usage)
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(workers=2,make_proposal=interrupt_during_second_batch,worker_function=slow_binding)
        if timer is not None:timer.join()
        state=json.loads((self.root/'out.candidate-state.json').read_text())
        self.assertEqual(state['attempts'],4)
        self.assertEqual({p.pid for p in mp.active_children()},before)
        resumed,_=self.run_batch(workers=2)
        reference,_=self.run_batch('reference',workers=2)
        self.assertEqual(resumed,reference)

    def test_uncommitted_partial_tail_is_discarded(self):
        self.run_batch(attempts_limit=4)
        with (self.root/'out.candidate-progress.jsonl').open('ab') as handle:
            handle.write(b'{"uncommitted":true}\ntruncated')
        resumed,_=self.run_batch()
        reference,_=self.run_batch('reference')
        self.assertEqual(resumed,reference)

    def test_committed_tampering_is_rejected_without_truncation(self):
        self.run_batch(attempts_limit=4)
        path=self.root/'out.candidate-progress.jsonl'
        before=path.read_bytes().replace(b'candidate-000001',b'candidate-000009')
        path.write_bytes(before)
        with self.assertRaisesRegex(CandidateCheckpointError,'hash mismatch'):
            self.run_batch()
        self.assertEqual(path.read_bytes(),before)

    def test_changed_signature_requires_explicit_restart(self):
        self.run_batch(attempts_limit=4)
        with self.assertRaisesRegex(CandidateCheckpointError,'changed'):
            self.run_batch(signature={'version':2})
        rows,stats=self.run_batch(signature={'version':2},restart=True)
        self.assertEqual(len(rows),8)
        self.assertFalse(stats['resumed'])

    def test_completed_run_does_not_repeat_binding(self):
        rows,_=self.run_batch()
        def unexpected(job):raise AssertionError('completed candidates were rebound')
        resumed,stats=self.run_batch(worker_function=unexpected)
        self.assertTrue(stats['resumed'])
        self.assertEqual(rows,resumed)

    @unittest.skipUnless(os.name=='posix','requires advisory file locks')
    def test_second_writer_is_rejected(self):
        with _exclusive_lock(self.root/'out.candidate-state.lock'):
            with self.assertRaisesRegex(CandidateCheckpointError,'another process'):
                self.run_batch()


if __name__=='__main__':unittest.main()

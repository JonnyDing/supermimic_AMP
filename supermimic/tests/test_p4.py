import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from mjlab.managers.termination_manager import TerminationManager
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager
from supermimic.mdp.recovery import advance_recovery
from supermimic.mdp.amp import ExpertPairs
from supermimic.evaluation.metrics import acceptance


class RecoveryTest(unittest.TestCase):
    def test_native_state_machine_recovery_deadline_timeout(self):
        enabled=torch.tensor([True,True,False,True])
        native=DelayedTerminationManager(SimpleNamespace(),enabled,250)
        counter=torch.tensor([248,249,0,7]);native._delay_counters.copy_(counter)
        scenarios=[([1,1,1,0],[0,0,0,0]),([1,1,0,0],[0,0,1,1]),([0,0,0,1],[0,0,0,0]),([1,0,1,0],[1,0,0,0])]
        for bad,timeouts in scenarios:
            bad=torch.tensor(bad,dtype=torch.bool);timeouts=torch.tensor(timeouts,dtype=torch.bool)
            def compute(manager):
                manager._terminated_buf=bad.clone();manager._truncated_buf=timeouts.clone()
                return bad|timeouts
            counter,terminated,truncated=advance_recovery(bad,timeouts,enabled,counter)
            with patch.object(TerminationManager,'compute',compute): done=native.compute()
            self.assertTrue(torch.equal(counter,native._delay_counters))
            self.assertTrue(torch.equal(terminated,native._terminated_buf))
            self.assertTrue(torch.equal(truncated,native._truncated_buf))
            self.assertTrue(torch.equal(done,terminated|truncated))

    def test_full_reward_gate_rejects_recovery_regression_and_nonfinite(self):
        base=dict(mean_task_return=100.,xy_rmse=1.,yaw_rmse=1.,velocity_survival_rate=1.,recovery_success_rate=1.,
                  velocity_full_return=50.,recovery_full_return=10.,velocity_style_reward=.1,recovery_style_reward=.1)
        self.assertTrue(acceptance(base,{**base,'recovery_full_return':9.8})['passed'])
        self.assertFalse(acceptance(base,{**base,'recovery_full_return':9.79})['passed'])
        self.assertFalse(acceptance(base,{**base,'recovery_full_return':float('inf')})['passed'])

    def test_expert_pairs_never_cross_clip_or_repeat_terminal(self):
        experts=ExpertPairs([torch.arange(3)[:,None],torch.arange(100,105)[:,None]],['a','b'])
        for i in range(4):
            s,ns=experts.sample(64)
            self.assertTrue(torch.equal(ns-s,torch.ones_like(s)))
            self.assertTrue((s<3).all() if i%2==0 else (s>=100).all())


if __name__=='__main__':unittest.main()

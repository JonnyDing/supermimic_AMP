"""Task migration contracts: terminal targets, BPTT, schemas and checkpoints."""
from types import SimpleNamespace

import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn

from supermimic.core.task_model import TaskWorldModel, TaskImagination, TaskHeads, task_rollout_loss, task_context
from supermimic.core.imagination import ValueModel, freeze_parameters
from supermimic.core.p4 import sample_starts
from supermimic.core.policy import create_policy, load_policy
from supermimic.core.replay import SequenceReplay


def fixture(frame_dim=98):
    batch, horizon = 4, 2
    history = torch.randn(batch, 4, frame_dim)
    task = dict(command=torch.zeros(batch, 3), heading_target=torch.zeros(batch),
                command_time_left=torch.full((batch,), 5.), is_heading_env=torch.zeros(batch, dtype=torch.bool),
                is_standing_env=torch.zeros(batch, dtype=torch.bool), recovery_enabled=torch.zeros(batch, dtype=torch.bool),
                recovery_counter=torch.zeros(batch, dtype=torch.long), episode_step=torch.zeros(batch, dtype=torch.long))
    state = torch.randn(batch, 71)*.01
    state[:, 2] = .8; state[:, 3:7] = state.new_tensor([1, 0, 0, 0])
    context = task_context(torch.zeros(batch, 2), history, task, 250)
    wm = TaskWorldModel(29, context.shape[-1], hidden=(16,16), frame_dim=frame_dim, statistics_mode='running')
    wm.update_stats(state, state, torch.zeros(batch,29), context)
    wm.update_task_stats(history, torch.randn(batch), torch.ones(batch,dtype=torch.bool))
    window = dict(state=state[:,None].expand(-1,horizon,-1), next_state=state[:,None].expand(-1,horizon,-1),
                  context=context[:,None].expand(-1,horizon,-1), action=torch.randn(batch,horizon,29),
                  reward=torch.zeros(batch,horizon), terminated=torch.zeros(batch,horizon,dtype=torch.bool),
                  truncated=torch.zeros(batch,horizon,dtype=torch.bool),
                  **{'labels/feet_in_contact':torch.zeros(batch,horizon,2),
                     'labels/self_collision_raw':torch.zeros(batch,horizon),
                     'next_task/observation_history':history[:,None].expand(-1,horizon,-1,-1).clone(),
                     'next_task/recovery_counter':torch.zeros(batch,horizon)})
    starts = dict(state=state[:,None], context=context[:,None],
                  **{'task/observation_history':history[:,None], 'task/previous_action':torch.zeros(batch,1,29)})
    starts.update({'task/'+k:v[:,None] for k,v in task.items()})
    return wm, window, starts


def check_imagination_propagates_actor_gradient_with_frozen_models(frame_dim,action_start):
    torch.manual_seed(42)
    wm, window, starts = fixture(frame_dim)
    # A supervised task-head update precedes actor optimization, as in training.
    opt = torch.optim.Adam(wm.parameters(),lr=.001)
    task_rollout_loss(wm,window)[0].backward(); opt.step(); opt.zero_grad(set_to_none=True)
    actor = nn.Sequential(nn.Linear(4*frame_dim,29),nn.Tanh())
    critic = ValueModel(wm.input_mean.numel())
    heads = SimpleNamespace(max_episode_steps=250,dt=.02,heading_stiffness=.5,yaw_range=(-1,1))
    heads.command = lambda state, task: TaskHeads.command(heads,state,task)
    engine = TaskImagination(wm,actor,nn.Identity(),critic,heads,precision='fp32',horizon=3)
    engine.action_start = action_start
    with freeze_parameters(wm,critic):
        result = engine(starts)
        (-result['objective']).backward()
    assert torch.isfinite(result['rewards']).all()
    assert result['critic_inputs'].shape[:2] == (4,4)
    assert sum(p.grad.abs().sum() for p in actor.parameters()) > 0
    assert all(p.grad is None for p in wm.parameters())


def check_terminal_observation_is_not_a_learning_target():
    wm, window, _ = fixture()
    window['terminated'].fill_(True)
    first = task_rollout_loss(wm,window)[0]
    window['next_task/observation_history'].fill_(12345.)
    second = task_rollout_loss(wm,window)[0]
    torch.testing.assert_close(first,second)


def check_sampling_without_recovery_partition():
    wm, window, starts = fixture()
    values = {k:v[:,0] for k,v in starts.items()}
    values.update(action=torch.zeros(4,29),next_state=values['state'],
                  terminated=torch.zeros(4,dtype=torch.bool),truncated=torch.zeros(4,dtype=torch.bool),
                  dynamics_valid=torch.ones(4,dtype=torch.bool),episode_id=torch.zeros(4,dtype=torch.long))
    replay=SequenceReplay(2,4)
    replay.add_values(values)
    out=sample_starts(replay,[0,1,2],SimpleNamespace(dt=.02),batch=8,horizon=2)
    assert len(out['state'])==8
    assert (out['_env_ids']<3).all()


def check_variable_policy_checkpoint(tmp_path):
    policy,norm=create_policy('cpu',392,113)
    checkpoint=tmp_path/'policy.pt'
    torch.save(dict(model_state_dict=policy.state_dict(),obs_norm_state_dict=norm.state_dict()),checkpoint)
    restored,rnorm=load_policy(checkpoint,'cpu')
    obs=torch.randn(2,392)
    torch.testing.assert_close(policy.act_inference(norm(obs)),restored.act_inference(rnorm(obs)))


class TaskModelTests(unittest.TestCase):
    def test_bptt_for_all_new_observation_schemas(self):
        for frame_dim, action_start in [(98,69),(283,67),(285,69)]:
            with self.subTest(frame_dim=frame_dim):
                check_imagination_propagates_actor_gradient_with_frozen_models(frame_dim,action_start)

    def test_task_configuration_and_cli(self):
        import src.tasks
        from scripts.train import TrainConfig
        from supermimic.cli import SUPPORTED_TASKS, build_arguments
        from supermimic.train_fast import arguments
        for task in SUPPORTED_TASKS:
            with self.subTest(task=task):
                cfg=TrainConfig.from_task(task)
                self.assertEqual(cfg.agent.algorithm.class_name,'WMRL')
                self.assertEqual(cfg.agent.class_name,'SuperMimic')
                self.assertFalse(hasattr(cfg.agent.algorithm,'clip_param'))
                self.assertNotIn('push_robot',cfg.env.events)
                self.assertEqual(cfg.env.observations['actor'].history_ordering,'time')
                if task.endswith('Rough'):
                    self.assertIn('terrain_levels',cfg.env.curriculum)
                    self.assertIn('height_scan',cfg.env.observations['actor'].terms)
                parsed,_=arguments(build_arguments(task,cfg,Path('/tmp/not-created-by-cli-test'),'cuda:0'))
                self.assertEqual(parsed.task,task)

    def test_terminal_targets(self):
        check_terminal_observation_is_not_a_learning_target()

    def test_no_recovery_sampling(self):
        check_sampling_without_recovery_partition()

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            check_variable_policy_checkpoint(Path(directory))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()

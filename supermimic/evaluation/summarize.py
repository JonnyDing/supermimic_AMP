"""Render completed fixed-scenario results without rerunning simulation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['status'] != 'completed':
        raise ValueError('Refusing to summarize incomplete evaluation')
    summary, protocol = report['summary'], report['protocol']
    recovery_mean = summary['recovery_time_success_mean_s']
    recovery_text = '无成功样本' if recovery_mean is None else f'{recovery_mean:.3f} 秒'
    lines = ['# PPO 固定场景基线', '',
             f"Checkpoint：`{protocol['checkpoint']['path']}`。种子：{protocol['seeds']}。", '',
             f"速度场景：{summary['velocity_episodes']} 次，20 秒存活率 {summary['velocity_survival_rate']:.0%}。",
             f"恢复场景：{summary['recovery_episodes']} 次，成功率 {summary['recovery_success_rate']:.0%}；成功判定平均耗时 {recovery_text}。", '',
             '恢复判定：5 秒内，连续 1 秒高度 ≥ 0.5 m、倾角 ≤ 70°；耗时包含稳定判定的 1 秒。', '',
             '| 指令 vx / vy / wz | XY 速度 RMSE（m/s） | yaw rate RMSE（rad/s） | 存活 |',
             '|---|---:|---:|---:|']
    for command in protocol['commands']:
        rows = [r for r in report['episodes'] if r['kind'] == 'velocity' and r['command'] == command]
        avg = lambda key: sum(r[key] for r in rows) / len(rows)
        lines.append(f"| {command} | {avg('velocity_xy_rmse_m_s'):.3f} | {avg('yaw_rate_rmse_rad_s'):.3f} | {sum(r['survived_full_horizon'] for r in rows)}/{len(rows)} |")
    lines += ['', '表中误差先按 episode 求 RMSE，再对三个种子取均值。速度使用 torso heading 坐标，yaw rate 使用 torso 世界坐标 z 分量。', '',
              '当前策略的侧向移动与原地转向跟踪较弱。存活率与恢复成功率不能替代指令跟踪质量。', '',
              f"恢复帧：`{protocol['recovery_frames']}`，来源：`{protocol['motion_path']}`。", '',
              f"实测 rollout 总耗时：{summary['evaluation_rollout_wall_seconds']:.2f} 秒，不含环境构建与模型加载；这是评估耗时，不是训练 GPU 时间。", '',
              '范围：关闭观测噪声与推扰，保留启动物理随机化。固定 8 帧来自同一个恢复动作，不代表任意倒地恢复；没有地形、推扰或训练速度对比。', '',
              '复现：', '', '```bash',
              '/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.scenarios --output /tmp/ppo_fixed_scenarios.json',
              '```', '', '完整逐场景指标、动作变化、接触足滑指标见同名 JSON，协议在 `.protocol.json` 中，均保留 checkpoint 和数据哈希。', '']
    args.report.with_suffix('.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

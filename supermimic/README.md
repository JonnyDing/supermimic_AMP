# AMP_mjlab / SuperMimic

> 已清理未使用的 RSL-RL 循环网络、蒸馏/RND、旧 replay/rollout storage、日志适配器和 motion loader。
> 播放入口仅负责策略预览；ONNX 导出使用 `scripts/export_sim2sim.py`。
> 阶段训练脚本保留为历史实验入口，当前核心不再从这些入口导入公共函数。

> 当前训练后端仅为 WMRL。PPO/AMP-PPO 优化器及训练 Runner 已移除；
> `core/policy.py` 提供公共 checkpoint 加载，`evaluation/scenarios.py` 保留固定场景基线评估。
> 原策略仍用于初始化和冻结 teacher。ActorCritic、归一化及 AMP 判别器保持兼容。
> `rsl_rl.runners.OnPolicyRunner` 仅为 mjlab 自动导入所需的占位类，不能训练。
> 下文的 PPO 训练入口描述属于历史记录，不代表当前功能。

状态：P0–P4 核心链路已验证，验收目标为媲美 PPO。长训沿用 `python scripts/train.py Unitree-G1-AMP-Flat`；性能结果、14,000 轮命令及续训说明见 [PERFORMANCE.md](PERFORMANCE.md)。下文保留各阶段历史诊断记录。

目标：将 SuperMimic 的真实仿真采集 → 世界模型多步拟合 → 想象 rollout / BPTT → critic 更新迁入本项目，最终保留速度控制、AMP 风格约束与跌倒恢复。Flat 优先，Rough 后续。原 AMP-PPO 作为对照。

## 2026-10-27：终端输出与 critic 尖峰诊断

原训练命令保持可用。终端显示进度、ETA、采样速度、critic/WM/判别器 loss、梯度与回报；不显示异常保护计数、排除比例或目标值幅度。`--supermimic.log-every=25` 控制输出间隔。训练指标仅通过 TensorBoard 持久记录：`Train/*` 为标量，`Diagnostics/*` 为详细事件。新训练不再写 metrics/guard/critic 的 JSONL 日志；`config.json`、`training.json` 和检查点保留配置及完成/失败状态。loss 是区间均值，`*_peak` 和 `*_abs_max` 是区间最大值。历史报告中的 JSONL 属于此前的实验格式。

4096 环境的 500 轮诊断在第 494 轮复现 critic loss 5284.52：一条恢复想象轨迹的动作幅度从 13 增长到 2176，奖励最低 -1841、GAE 目标最低 -1908，continuation 始终为 1。真实采集拒绝绝对值达到 100 或非有限的动作，想象端此前没有相同约束。现在两端共用动作有效性规则；任一步动作无效的整条想象轨迹从 actor 和 critic 更新中排除，整批无效直接报错。有效奖励和目标不做裁剪。该保护不能代替世界模型精度与恢复状态分布的改善。

同一模型、同一批 512 条起点的[对照探针](reports/critic_guard_paired_probe.json)：排除 1 条越界轨迹后，更新前 critic MSE 从 60325.06 降到 0.12076。修复后另一轮 4096 环境、500 次更新的 critic loss 单次峰值为 0.17792，最后 25 次均值为 0.03343；两次训练并非逐步确定性配对，因果判断以同批探针为依据。[编译前向与梯度一致性检查](reports/critic_guard_compiled_parity.json)通过，23 项单元测试通过。

这验证了已复现尖峰的来源和防护效果，尚未验证修复后的 14000 轮稳定性或真实仿真相对 PPO 的性能。旧 14000 轮日志缺少尖峰时刻的逐轨迹信息，不能断言每次历史尖峰都由这一原因造成。

## 目录与职责

```text
supermimic/
  config.py           算法配置，默认值尚未经过目标环境调优
  contracts.py        状态、transition、仿真适配器与任务头协议
  __main__.py         打印配置；不是训练入口
  adapters/           mjlab 1.2 状态采集、reset、机器人名称映射、DR 读取
  core/               世界模型、序列 replay、BPTT、critic、checkpoint
  mdp/                可微 FK、观测、奖励、恢复状态与终止
  amp/                可微判别器奖励与真实数据判别器训练
  evaluation/         固定评估场景、指标、ONNX 验证
  IMPLEMENTATION_PLAN.md
```

目前 adapters/mjlab.py、mdp/kinematics.py、evaluation/baseline.py 和 evaluation/probe.py 已有实现，其余训练模块仍待实现。PhysicsCollector 目前提供物理 transition，历史已保存于 task/next_task 的 observation_history（B,4,96）；PhysicsTransition 仍是具体采集接口，未声明实现 contracts.py 的通用任务协议。

在仓库根目录执行：

```bash
python -m supermimic --print-config
```

setup.py 已加入 supermimic 子包发现；在现有 mjlab 环境以 --no-deps --no-build-isolation 重新安装后，已验证可从 /tmp 导入，不升级依赖。

## 架构约束

- 保持目标项目 mjlab 1.2.0 / mujoco-warp 3.8.1 / warp-lang 1.12.0 依赖，暂不跟随参考项目升级。
- 仿真适配器负责 reset；训练器只消费 reset 前的完整 transition。terminated 与 truncated 分开保存。
- 世界模型学习物理状态；恢复计数、命令计时和参考进度由显式任务逻辑推进。
- 状态使用 root link 世界坐标速度，位置减 env origin，内部四元数 wxyz；关节与 body 按名称映射。
- actor 更新冻结 WM、critic、判别器参数，保留它们对输入的导数；归一化统计同样冻结。
- AMP 判别器只用真实策略数据与专家数据训练；想象数据只计算可微奖励。
- 现有 actor 的观测顺序、4 帧 time-history、动作尺度优先保持；tracking 基准可单独使用论文 10 帧配置。
- 奖励时间缩放按现有项目实际行为对齐，包括 AMP 混合，不重复乘 dt。
- 不直接复用旧世界模型权重：目标执行器、摩擦、模型资产不同，需要重新采集与训练。
- checkpoint 保存 actor / WM / critic / discriminator、优化器、normalizer、RNG、配置及状态 schema；buffer 不保存时明确标识恢复后需要重新填充。

## 参考

- 本地论文：`/home/ubuntu/Downloads/CoRL2026_SuperMimic.pdf`，§3 与附录 A–D。
- 本地复现：`/home/ubuntu/Sia/supermimic_vibed/supermimic/`。
- mjlab 参考：[InstinctMJ supermimic](https://github.com/project-instinct/InstinctMJ/tree/supermimic/src/instinct_mj/rl/supermimic)，本次审阅提交 `7ca851e002980011f24a1664011305d0ba94aefc`。

移植实际代码时保留源文件许可与归属；参考分支的视觉、地形、instinct_rl 依赖不整体引入。

## 当前验证与剩余工作

结果见 [P1 GPU 检查](reports/p1_probe.json) 和 [基线来源清单](reports/baseline_manifest.json)。来源清单冻结代码/数据/checkpoint 哈希，并关联已完成的固定场景 PPO 评分。checkpoint 按最新运行目录和最大迭代号选择，不代表最佳策略。

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.baseline --output /tmp/baseline_manifest.json
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.probe --output /tmp/p1_probe.json
```

probe 使用 4 个环境、关闭推扰、缩短 timeout，并强制一次跌倒用于检查；不是训练或性能评估。当前检查包含 root 坐标、FK 位姿/速度、有限差分梯度、快照独立性、连续 transition、timeout 和真正 terminated 的 reset 前捕获。

终止帧历史与接触标签已补齐，固定场景评估见下文。接触传感器和奖励的原始读取时刻已明确区分：mjlab 1.2 的派生量在奖励计算时滞后一个物理子步。PhysicsCollector 拒绝带 step/interval 事件的环境，避免把未建模推扰混入动力学数据。

## 终止历史、接触监督与 PPO 基线

- `task['observation_history']` 保存真实 actor 历史；非终止 `next_task` 保存 env 返回的下一历史。终止行保留上一历史最后 3 帧，并追加从 reset 前原始物理状态重建的第 4 帧。
- 终止帧使用结束 transition 的命令（不执行新 episode 的命令采样）。IMU 按目标模型 pelvis 对齐姿态计算，遵守现有观测项顺序、偏差开关、噪声、clip 和 scale。噪声采样恢复 RNG，不修改实时观测管理器历史。
- 接触标签在 reward compute 后、reset/forward/sense 前保存，包含足部 found/force、接触与腾空计时、自碰撞 force history，以及原始 foot_slip/self_collision 奖励目标。该时刻的子步传感器标签不冒充 post-forward 标签。
- probe 验证终止历史移位与最后动作、无 RNG/实时 history 副作用、接触奖励公式一致性，以及之前的 FK 和 reset 检查。

[PPO 结果摘要](reports/ppo_fixed_scenarios.md) · [逐场景结果](reports/ppo_fixed_scenarios.json) · [冻结协议](reports/ppo_fixed_scenarios.protocol.json)

本次 3 个种子，共 15 个速度场景与 24 个恢复场景。20 秒速度场景存活率与预设恢复成功率均为 100%，但侧向指令与原地转向跟踪明显不足，详见逐指令误差。恢复帧只来自一个 clip，且关闭噪声/推扰，不作泛化结论。

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.scenarios --output /tmp/ppo_fixed_scenarios.json
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.summarize /tmp/ppo_fixed_scenarios.json
```

P2 已通过 SequenceReplay.add 对接 PhysicsTransition，完成多步世界模型诊断。P3 已开展短时域 BPTT 与配对真实评估；固定场景评分不替代后续推扰/噪声评估。

## P2 世界模型

实现文件：`core/replay.py`（序列 ring buffer）、`core/state.py`（状态编码/积分）、`core/world_model.py`（动态维度 MLP 与接触头）、`train_world_model.py`（离线诊断采集/训练）、`evaluation/world_model.py`（独立验证）。旋转工具来自 supermimic_vibed，MIT 许可保存在 `licenses/`。

[实验报告](reports/p2_world_model.md) · [分类误差与元数据](reports/p2_world_model.json)

Replay 同时保存原始物理状态、actor 历史、任务状态和接触标签；按 episode_id、terminated、truncated、dynamics_valid 筛选完整窗口，支持物理存储环形覆盖。没有合法窗口时明确报错，不无限重采样。

默认诊断只运行 3×512 世界模型，论文配置仍在 config.py 中。采集使用原 PPO 作为数据源，不更新 actor/判别器。checkpoint 包含网络架构、归一化、优化器、RNG 与环境划分；支持保存重载和独立评估，尚未提供训练恢复 CLI。

32 步倒地误差仍较大；P3 应先验证短时域 actor 梯度与真实仿真收益，不能只凭 loss 下降扩展完整恢复训练。

## P3 短时域 BPTT

- `core/imagination.py`：保持状态/历史计算图的短时域展开、参数冻结、终端 value bootstrap、GAE targets 与独立 critic 更新。
- `mdp/velocity.py`：现有线/角速度跟踪、动作变化与 joint-limit 四项可微奖励，以及与原 actor 一致的 96 维单帧观测。
- `train_bptt.py`：从 PPO warm start，使用 P2 的冻结 WM；开发种子选择候选后，才运行不同种子的配对真实仿真。
- `evaluation/bptt.py`：独立重载候选 checkpoint，与原 PPO 比较初态和 context 哈希一致的场景。
- `evaluation/verify_velocity_heads.py`：在同一 post-forward 时刻对比原生奖励项，避免把一个物理子步的时差当成公式误差。

P3 是 upright velocity 消融：不更新 WM，不训练 AMP，不使用恢复目标。真实仿真仍运行完整目标执行器；收益评分使用同一四项任务目标并加硬终止惩罚。原 PPO checkpoint 和原训练入口没有被替换。

[配对评估报告](reports/p3_bptt.md) · [奖励头验证](reports/p3_heads_probe.json)

历史 P3 验收线在试验前固定：真实回报提升 >0.5%，至少 2/3 测试种子改善，存活率不下降，XY/yaw 平均 RMSE 不超过基线的 105%。工程检查通过不意味着这条收益线通过；请以报告的实测结果为准。

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_bptt --output logs/supermimic/p3_new_run
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_bptt --output logs/supermimic/p3_no_bootstrap_new --bootstrap-weight 0 --test-seeds 52 53 54
```

已保存 actor、normalizer、P3 critic 和优化器 checkpoint；当前没有 CLI 续训或 ONNX 导出。后续需要先补充在线采集/WM 刷新与 value 梯度校验，不能仅因离线想象目标上升就进入完整 AMP/恢复训练。

## 在线交替更新与新验收标准

用户已明确无需超越 PPO。后续运行采用固定工程容差：平均任务回报下降 ≤2%，XY/yaw RMSE 各增加 ≤5%，存活率和恢复成功率不下降。旧报告保留原来的严格收益判定；新旧要求不能混淆。

`train_online.py` 从 P3 actor_10 与 P2 世界模型启动；当前 actor 采集真实 transition，两个 WM minibatch、一次 H=8 BPTT actor 更新、一次 GAE critic 更新交替执行。默认 actor 学习率 1e-7、64 环境、64 步预填充、100 次交替更新，每轮再采集两步；48/16 环境隔离训练/验证。PPO normalizer 和 WM 归一化统计冻结，原 PPO actor 只作锚定参照。checkpoint 记录网络、优化器、RNG；replay 另存。尚无模拟器精确续训 CLI。

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_online --output logs/supermimic/p3_online_new
```

最终 checkpoint 在评估前固定，并通过原生加载接口重载；以新种子 62/63/64 对原 PPO 做相同初态与上下文的配对真实评估。训练仍是速度目标消融，恢复只作回归测试；尚不代表完整 AMP、推扰或从零训练的算法结论。

首轮在线学习率 1e-6 试验未通过：XY 误差增加约 6.37%，恢复成功率从 100% 降到 95.83%；一个恢复 episode 出现动作平方溢出。失败结果完整保存在 [首轮报告](reports/p3_online_initial.md)，没有因序列化异常丢弃。降低学习率的复验使用新种子，避免在原测试集上选择结果。

学习率 1e-7 的复验已通过：三个新测试种子 72/73/74 的单种子与汇总检查均满足新容差。平均任务回报 -0.129%，XY RMSE +1.98%，yaw RMSE -0.43%，速度存活率及固定恢复成功率均为 100%。见 [在线闭环报告](reports/p3_online.md)。最终策略为 `logs/supermimic/p3_online_conservative_20261026/actor_final.pt`。这完成了 P3 的小规模在线闭环验收，下一阶段进入 P4 的可微 AMP 与恢复目标。

复验命令：
```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_online --output logs/supermimic/p3_online_repeat --actor-lr 1e-7 --test-seeds 72 73 74
```

## P4：可微 AMP、判别器和恢复目标

- `mdp/amp.py`：与原任务一致的 195 维 AMP 特征、保留输入导数的混合奖励、真实策略/专家 transition 判别器更新。原 PPO 的 `predict_amp_reward` 保持不变。
- `mdp/recovery.py`：250 步延迟计数、恢复清零、timeout 不延期、恢复奖励切换、heading 控制；完整任务奖励的可微代理。
- `core/p4.py`：约 40% 起点来自恢复环境，critic 输入包含恢复计数、命令时间和 heading 状态。命令重采样边界不跨越。
- `train_p4.py`：在线真实采集 → WM → 判别器 → BPTT actor → GAE critic。actor 更新冻结 WM、critic、判别器参数与 AMP 统计。

判别器沿用原实现的 least-squares loss、对原始专家特征的梯度惩罚、trunk/head weight decay，以及用归一化后的当前策略/专家特征更新 AMP normalizer 的原有行为。后者是现有 PPO 的实现细节，本次为保持一致而保留，不将其当作通用推荐。专家按原实现轮换 clip；采样仅取同 clip 的相邻 50 Hz 帧，排除末帧重复配对。

接触概率代替足底二值接触；`4 × 自碰撞概率` 近似四个物理子步的碰撞计数；关节加速度采用控制步有限差分；软终止用于想象梯度。真实评估使用原环境奖励和硬终止。使用真实传感器/加速度替换代理时，逐项奖励已通过同一时刻的数值对齐：[检查记录](reports/p4_parity.json)。

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.verify_p4 --output /tmp/p4_parity.json
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_p4 --output logs/supermimic/p4_new
```

P4 的真实评估让原 PPO 与候选共用原始冻结判别器和 AMP normalizer；既检查之前的速度/恢复指标，也检查完整 AMP 混合回报（速度、恢复分别下降不超过 2%）。评估包含固定横向推扰：第 5 秒对速度场景加入 `(0.6, 0.3) m/s` 根速度增量。训练仍关闭随机 interval 推扰，保留 startup DR；这是一组推扰回归测试，不能等同于完成全随机推扰训练。

P4 小规模在线闭环已通过：三组固定场景与一组固定横向推扰，逐组及汇总均满足预设容差。固定场景完整 AMP 速度回报 -0.081%，恢复回报 +2.94%；速度存活率均为 100%，恢复成功率从 PPO 的 23/24 到 P4 的 24/24。14 项测试、奖励前向对齐和端到端梯度检查通过。见 [P4 实验报告](reports/p4_amp_recovery.md)。候选策略：`logs/supermimic/p4_amp_recovery_20261026/actor_final.pt`。

完整随机推扰训练、接触/加速度代理消融、多训练种子和长预算学习效率尚未完成；本轮不宣称这些能力。保存策略的端到端梯度复验：
```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.verify_p4_gradient logs/supermimic/p4_amp_recovery_20261026 --output /tmp/p4_gradient.json
```

## Independent MuJoCo playback

See [deployment/README.md](deployment/README.md) for `scripts/sim2sim.py`, native/headless commands, control contracts and current validation limits.

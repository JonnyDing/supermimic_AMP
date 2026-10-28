# Python ONNX + C++ SDK2 communication

Board entrypoint (manual hardware run; not executed during development):

```bash
conda activate robomimic
cd ~/supermimic_amp/current
python deploy/deploy_g1_cpp.py --net eth0 --policy deploy/wmrl/policy.onnx
```

No shell launcher or manual exports required. `cpp_runtime.py` selects the board's
matching `/usr/local/lib/libddscxx.so` and `libddsc.so` before importing the native
module. Python DDS's `/opt/cyclonedds-0.10.2` is NOT mixed into this process.
The environment is changed for this process only. Python ONNX inference is unchanged.

## Reference and scope

The designated reference is
`~/supermimic_amp/RoboMimic_Deploy_bjc/deploy_real/deploy_real.py` (identical to
`~/RoboMimic_Deploy_bjc/deploy_real/deploy_real.py`). That reference imports
**unitree_sdk2py**, not unitree_cpp. Its receive callback stores the latest state
and updates the remote; it does not compute Python CRC on every incoming state.
The original WMRL controller added this per-state check, measured at ~1.87 ms on
board, limiting receive throughput to ~540 Hz for an approximately 1 kHz stream.

This native extension preserves the reference's DDS topics, HG message types,
PR mode, mode_machine feedback, 29 SDK motor indices, pelvis wxyz quaternion,
gyro, and joystick axes (`ly,-lx,-rx`). It moves DDS deserialization, CRC, receive
buffering and send serialization into C++; it does not import RoboJuDo's controller,
call MotionSwitcher, or reproduce another policy's observation dimensions/gains.
WMRL keeps its own 4-frame observation, ONNX normalization and exported parameters.

The reference's FSM has START for POS_RESET and R1+A for LOCO. The existing WMRL
operator workflow remains START -> nominal pose -> A -> policy; SELECT/B stop.
This is explicit, not a claim that every reference FSM detail is identical.

## Native behavior

- Constructor subscribes, optionally creates a publisher, but sends no commands.
- Read-only mode creates no LowCmd publisher at all.
- Receive callback validates CRC, latches B/SELECT, and overwrites the latest state
  without the Python GIL or a Python queue. Duplicate ticks don't refresh age.
- Python sends one desired position command per 20 ms policy period.
- Native watchdog checks every 5 ms. After first command, no fresh state or no new
  Python command for >100 ms latches a stop. It attempts damping for about 1 second
  using exported KD, KP=0, tau=0; it never keeps replaying the previous policy target.
- DDS Write executes outside the state mutex. A separate writer mutex orders policy
  and watchdog sends. Callbacks never need a Python lock.
- Closing a read-only or never-started controller does not initiate motor commands.
- SIGKILL/power/network loss cannot guarantee damping; independent robot protection
  remains required. No automatic motion-controller release or ownership checking.

## Build

Built against the existing board SDK headers/static library and matching native DDS
libraries in /usr/local, with robomimic's Python 3.8 and pybind11. Existing SDK packages
and system libraries are unchanged.

```bash
cmake -S deploy/cpp -B deploy/cpp/build \
  -DPYTHON_EXECUTABLE="$HOME/miniconda3/envs/robomimic/bin/python" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/cpp/build -j2
```

Runtime files: deploy_g1_cpp.py, policy_core.py (shared policy logic), cpp_runtime.py,
wmrl_transport.cpython-38-aarch64-linux-gnu.so, policy.onnx, policy.json. The obsolete pure Python DDS entrypoint has been removed.

## Directory scope

This directory contains the deployment entrypoint, shared policy logic, native
runtime loader, C++ transport source/build configuration, and policy assets.
Standalone environment checks, stream diagnostics, replay verification tools and
the native test peer have been removed. Rebuild the native extension from the
current source before deployment.

Historical JSON reports and the recorded trace remain as archived evidence;
their script paths refer to removed verification tools. Real-robot ONNX closed-loop
behavior has not been validated by this cleanup.

"""Python ONNX + native SDK2 transport. No Python DDS imports or MotionSwitcher calls."""
import argparse
import time
from types import SimpleNamespace
from cpp_runtime import prepare
prepare()
import numpy as np
from wmrl_transport import Bridge
from policy_core import PolicyController, Stop, START, A


class Controller(PolicyController):
    def __init__(self, policy, net='eth0', domain=0, read_only=False, **kwargs):
        self.net,self.domain,self.read_only=net,domain,read_only
        super().__init__(policy,**kwargs)

    def _init_transport(self):
        kd=np.zeros(29)
        kd[self.motor_ids]=self.kd
        self.bridge=Bridge(self.net,self.domain,self.read_only,kd.tolist())

    def _read(self):
        d=self.bridge.snapshot()
        self.native_state=d
        if not d['have']:
            return None
        s=SimpleNamespace(tick=d['tick'],mode_machine=d['mode_machine'],wireless_remote=d['remote'],
                          imu_state=SimpleNamespace(quaternion=d['quat'],gyroscope=d['gyro']),
                          motor_state=[SimpleNamespace(q=q,dq=dq) for q,dq in zip(d['q'],d['dq'])])
        self.low_state=s
        self.buttons=int.from_bytes(d['remote'][2:4],'little')
        return s

    def wait_connected(self,timeout=10.):
        end=time.monotonic()+timeout
        while self._read() is None:
            if time.monotonic()>end:raise TimeoutError('No native LowState')
            time.sleep(.005)
        return self.snapshot()

    def snapshot(self):
        s=self._read()
        if self.native_state['stopped']:raise Stop(self.native_state['reason'])
        if s is None or self.native_state['age']>.1:raise Stop('Native LowState stale > 100 ms')
        return s

    def pressed(self,button):
        self._read()
        return bool(self.buttons & (1<<button))

    def write(self,q,kp,kd,mode_machine):
        def sdk_order(v):
            result=np.zeros(29);result[self.motor_ids]=v
            return result.tolist()
        try:
            self.bridge.send(sdk_order(q),sdk_order(kp),sdk_order(kd),False)
        except RuntimeError as exc:
            raise Stop(str(exc)) from exc

    def damp(self,seconds=1.):
        if self.read_only:return
        d=self.bridge.snapshot()
        if not d['have']:return
        z=[0.]*29
        for _ in range(max(1,round(seconds/self.dt))):
            self.bridge.send(z,z,z,True)  # Native saved damping gains are used.
            time.sleep(self.dt)

    def close(self):
        self.bridge.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--net',default='eth0')
    p.add_argument('--policy',required=True)
    p.add_argument('--velocity-scale',nargs=3,type=float,default=[1.,1.,1.])
    p.add_argument('--deadzone',type=float,default=.01)
    p.add_argument('--max-tilt-deg',type=float)
    a=p.parse_args()
    if a.net=='lo':p.error('Deployment requires a robot network interface')
    if a.max_tilt_deg is not None and not 0<a.max_tilt_deg<=180:p.error('Invalid tilt angle')
    c=Controller(a.policy,net=a.net,velocity_scale=a.velocity_scale,deadzone=a.deadzone,max_tilt_deg=a.max_tilt_deg)
    try:
        c.wait_connected();z=np.zeros(29)
        print('C++ transport connected. Zero torque; START: nominal pose; B/SELECT: stop.',flush=True)
        c.wait_button(START,z,z,z);c.move_to_start()
        print('Holding pose. Center sticks, then A: policy.',flush=True)
        c.wait_button(A,c.default_q,c.kp,c.kd);c.run()
    except (Stop,KeyboardInterrupt) as exc:print('Stopped:',exc,flush=True)
    finally:
        try:c.damp()
        finally:c.close()


if __name__=='__main__':main()

# MuJoCo_VEH/scripts/control/torque_sync_tcs.py
from __future__ import annotations
import numpy as np
import mujoco

class TorqueSyncTCS:
    def __init__(self, model,
                 wheel_joint_names=("fl_wheel","fr_wheel","rl_wheel","rr_wheel"),
                 motor_actuator_names=("fl_motor","fr_motor","rl_motor","rr_motor"),
                 wdiff_th=5.0,    # 좌우 바퀴 각속도 차 임계값 [rad/s]
                 tcs_reduce=0.15, # 빠른 쪽 토크 감쇠율 (0.15=15%)
                 ramp_time=0.4,   # 가속 토크 램프 시간 [s]
                 lpf_alpha=0.2):  # 토크 저역통과(0=끄기, 0.1~0.3 권장)
        self.model = model
        self.aid_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_actuator_names[0])
        self.aid_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_actuator_names[1])
        self.aid_rl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_actuator_names[2])
        self.aid_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_actuator_names[3])

        self.did_fl = self._wheel_dof(model, wheel_joint_names[0])
        self.did_fr = self._wheel_dof(model, wheel_joint_names[1])
        self.did_rl = self._wheel_dof(model, wheel_joint_names[2])
        self.did_rr = self._wheel_dof(model, wheel_joint_names[3])

        self.wdiff_th  = float(wdiff_th)
        self.tcs_reduce = float(tcs_reduce)
        self.ramp_time = float(ramp_time)
        self.lpf_alpha = float(lpf_alpha)

        self.t0 = None
        self._prev = None  # for LPF

    def _wheel_dof(self, model, joint_name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"joint not found: {joint_name}")
        return int(model.jnt_dofadr[jid])

    def reset(self, data):
        self.t0 = float(data.time)
        self._prev = None

    def _clip_to_range(self, ctrl: np.ndarray):
        cr = self.model.actuator_ctrlrange
        np.clip(ctrl, cr[:,0], cr[:,1], out=ctrl)

    def apply(self, data, ctrl: np.ndarray):
        """
        compose_control() 등으로 ctrl을 만든 *직후* mj_step 이전에 호출.
        ctrl은 in-place로 수정한다.
        """
        if self.t0 is None:
            self.reset(data)

        # 1) 바퀴 각속도
        w_fl = data.qvel[self.did_fl]
        w_fr = data.qvel[self.did_fr]
        w_rl = data.qvel[self.did_rl]
        w_rr = data.qvel[self.did_rr]

        # 2) 현재 토크
        tau_fl = ctrl[self.aid_fl]; tau_fr = ctrl[self.aid_fr]
        tau_rl = ctrl[self.aid_rl]; tau_rr = ctrl[self.aid_rr]

        # 3) 좌/우 평균 동기화 (앞, 뒤 각각)
        tau_f = 0.5*(tau_fl + tau_fr); tau_r = 0.5*(tau_rl + tau_rr)
        tau_fl = tau_fr = tau_f
        tau_rl = tau_rr = tau_r

        # 4) 간단 TCS: 좌우 각속도 차가 크면 빠른 쪽만 감쇠
        def tcs_pair(t_l, t_r, w_l, w_r):
            if abs(w_l - w_r) > self.wdiff_th:
                if w_l > w_r: t_l *= (1.0 - self.tcs_reduce)
                else:         t_r *= (1.0 - self.tcs_reduce)
            return t_l, t_r
        tau_fl, tau_fr = tcs_pair(tau_fl, tau_fr, w_fl, w_fr)
        tau_rl, tau_rr = tcs_pair(tau_rl, tau_rr, w_rl, w_rr)

        # 5) 토크 램프 (초기 가속 임펄스 줄이기)
        elapsed = max(0.0, float(data.time) - float(self.t0))
        if self.ramp_time > 1e-6:
            ramp = min(1.0, elapsed / self.ramp_time)
            tau_fl *= ramp; tau_fr *= ramp; tau_rl *= ramp; tau_rr *= ramp

        # 6) 저역통과
        if self.lpf_alpha > 0.0:
            import numpy as np
            if self._prev is None:
                self._prev = np.array([tau_fl, tau_fr, tau_rl, tau_rr], dtype=float)
            curr = np.array([tau_fl, tau_fr, tau_rl, tau_rr], dtype=float)
            sm = (1.0 - self.lpf_alpha) * self._prev + self.lpf_alpha * curr
            tau_fl, tau_fr, tau_rl, tau_rr = sm.tolist()
            self._prev = sm

        # 7) ctrl 반영 + 8) 안전 클리핑
        ctrl[self.aid_fl] = tau_fl; ctrl[self.aid_fr] = tau_fr
        ctrl[self.aid_rl] = tau_rl; ctrl[self.aid_rr] = tau_rr
        self._clip_to_range(ctrl)

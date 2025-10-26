# scripts/vehicle/ebrake.py

from __future__ import annotations
import math
import mujoco
import numpy as np
from dataclasses import dataclass

@dataclass
class WheelState:
    w: float     # 바퀴 각속도 [rad/s]
    R: float     # 바퀴 반경 [m]
    load: float  # 바퀴 수직하중 [N] (없으면 대략 m*g/바퀴수)

class EBrake: 
    """
    전자식 브레이크(Brake-by-Wire) 최소구성:
      - 페달맵: pedal(0~1) -> 차량 종감속 목표 ax_ref
      - EBD: 앞/뒤 축 제동 분배(front_bias)
      - 회생제동 우선, 잔여를 마찰제동으로
      - ABS: 바퀴별 slip 제어 (간단형)
      - 액추에이터 지연: 1차 지연(τ)
    """
    def __init__(self, model=None, data=None, **kwargs):
        """
        Args:
            sim: MuJoCo 시뮬레이터 인스턴스 (mjcf or mujoco.MjSim)
            mass_kg: 차량 질량 [kg]
            frictionloss_max: 최대 브레이크 마찰손실값 [N·m·s/rad]
            tau_actuator: 제동 반응 시간상수 [s]
        """
        self.model = model
        self.data = data
        self.m = kwargs.get("mass_kg", 1300.0)
        self.frictionloss_max = kwargs.get("frictionloss_max", 2500.0)
        self.brake_tau = kwargs.get("tau_actuator", 0.05)

        # 휠 조인트 이름 (모델 이름과 반드시 일치해야 함)
        self.wheel_joints = ["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"]

        # 내부 상태
        self._brake_cmd_prev = 0.0

    # ---------- 페달 입력 → 제동 강도 변환 ----------
    def pedal_to_brake_strength(self, pedal: float) -> float:
        pedal = np.clip(pedal, 0.0, 1.0)
        shaped = pedal * pedal * (3 - 2 * pedal)  # smoothstep curve
        return shaped

    # ---------- 1차 지연 ----------
    def _first_order(self, prev, cmd, dt):
        a = dt / (self.brake_tau + dt)
        return (1 - a) * prev + a * cmd

    # ---------- frictionloss 적용 ----------
    def apply_brake(self, pedal: float, dt: float):
        """
        브레이크 페달 입력(0~1)에 따라 frictionloss를 실시간 갱신
        """
        if self.model is None or self.data is None:
            raise ValueError("EBrake requires valid MuJoCo model/data objects")

        # 1) 페달 입력 → 제동강도 계산
        brake_cmd = self.pedal_to_brake_strength(pedal)
        brake_cmd = self._first_order(self._brake_cmd_prev, brake_cmd, dt)
        self._brake_cmd_prev = brake_cmd

        # 2) 마찰손실 값 계산 (패드 압착 효과)
        friction_val = brake_cmd * self.frictionloss_max

        # 3) 각 바퀴 조인트에 적용
        for jname in self.wheel_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            dof_id = self.model.jnt_dofadr[jid]
            self.model.dof_frictionloss[dof_id] = friction_val

    # ---------- (옵션) 회생제동 병용 ----------
    def apply_regen_torque(self, regen_torque: float):
        """
        회생제동 토크를 따로 적용하고 싶을 때 (모터축에 추가)
        """
        pass  # 필요시 별도 구현

    # ---------- (예전 인터페이스 호환용) ----------
    def compute_wheel_torques(self, **kwargs) -> list[float]:
        """
        구형 코드 호환용 더미 함수 (토크 기반 대신 frictionloss 기반)
        """
        pedal = kwargs.get("pedal", 0.0)
        dt = kwargs.get("dt", 0.001)
        self.apply_brake(pedal, dt)
        # frictionloss 방식은 직접 토크 반환 안 함
        return [0.0, 0.0, 0.0, 0.0]
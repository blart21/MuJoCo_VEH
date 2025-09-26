# scripts/vehicle/ebrake.py

from __future__ import annotations
import math
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
    def __init__(
        self,
        mass_kg: float,
        wheel_inertia: float = 1.2,
        n_front_wheels: int = 2,
        n_rear_wheels: int = 2,
        front_bias: float = 0.6,          # 앞축 60%
        pedal_ax_max: float = -8.0,       # pedal=1일 때 목표 종감속 [m/s^2] (음수=감속)
        regen_max_torque: float = 1200.0, # 바퀴당 회생 최대토크 [N·m]
        mu_tire: float = 0.9,
        abs_enable: bool = True,
        slip_target: float = 0.18,
        abs_kp: float = 180.0,
        abs_ki: float = 0.0,
        tau_actuator: float = 0.04,       # 유압/모듈레이터 지연(1차) [s]
    ):
        self.m = mass_kg
        self.Jw = wheel_inertia
        self.nf = n_front_wheels
        self.nr = n_rear_wheels
        self.front_bias = front_bias
        self.pedal_ax_max = float(pedal_ax_max)
        self.regen_max = regen_max_torque
        self.mu = mu_tire
        self.abs_enable = abs_enable
        self.slip_target = slip_target
        self.abs_kp = abs_kp
        self.abs_ki = abs_ki
        self.tau = max(1e-3, tau_actuator)

        # 내부 상태
        self._abs_int = [0.0, 0.0, 0.0, 0.0]  # 바퀴별 적분
        self._T_cmd_prev = [0.0, 0.0, 0.0, 0.0]  # 1차 지연용

    # ---------- 상위계층: 페달 → 목표 종감속 ----------
    def pedal_to_ax(self, pedal: float) -> float:
        pedal = max(0.0, min(1.0, pedal))
        # 간단 S-curve 맵(저페달에서 부드럽게)
        x = pedal
        shaped = x*x*(3 - 2*x)   # smoothstep
        return shaped * self.pedal_ax_max  # 음수(감속)

    # ---------- 토크 분배(EBD) ----------
    def distribute_axles(self, Fx_total: float) -> tuple[float, float]:
        Fx_front = Fx_total * self.front_bias
        Fx_rear  = Fx_total * (1.0 - self.front_bias)
        return Fx_front, Fx_rear

    # ---------- 회생 우선 블렌딩 ----------
    def blend_regen_friction(
        self, Fx_axle: float, wheel_states: list[WheelState], is_drive_axle: bool
    ) -> tuple[list[float], list[float]]:
        """
        축에 요구되는 제동력 Fx_axle(>0)를 바퀴별 회생/마찰 토크로 분해.
        회생은 '구동축'에서만 사용(보통 후륜모터/전륜모터 선택).
        """
        n = len(wheel_states)
        Fx_axle = max(0.0, Fx_axle)
        T_regen = [0.0]*n
        T_fric  = [0.0]*n

        # 바퀴별 요구토크(균등)
        for i, ws in enumerate(wheel_states):
            Ti_req = Fx_axle / n * ws.R
            if is_drive_axle:
                Ti_rg = min(self.regen_max, Ti_req)
            else:
                Ti_rg = 0.0
            T_regen[i] = Ti_rg
            T_fric[i]  = max(0.0, Ti_req - Ti_rg)
        return T_regen, T_fric

    # ---------- ABS 슬립 제어 ----------
    def _abs_wheel(self, T_cmd: float, veh_v: float, ws: WheelState, idx: int, dt: float) -> float:
        if not self.abs_enable or veh_v < 0.5:
            return max(0.0, T_cmd)
        v = max(veh_v, 1e-3)
        slip = (v - ws.w * ws.R) / v
        err = self.slip_target - slip
        self._abs_int[idx] += err * dt
        scale = 1.0 + (self.abs_kp * err + self.abs_ki * self._abs_int[idx]) / max(abs(T_cmd) + 1e-6, 1.0)
        T = T_cmd * max(0.0, min(1.5, scale))
        return max(0.0, T)

    # ---------- 액추에이터 1차 지연 ----------
    def _first_order(self, u_prev: float, u_cmd: float, dt: float) -> float:
        a = dt / (self.tau + dt)
        return (1 - a) * u_prev + a * u_cmd

    # ---------- 메인 API ----------
    def compute_wheel_torques(
        self,
        pedal: float,
        veh_v: float,
        wheels_front: list[WheelState],
        wheels_rear: list[WheelState],
        dt: float,
        drive_axle: str = "rear",  # "front"|"rear"
    ) -> list[float]:
        """
        반환: [Tfl, Tfr, Trl, Trr] (모두 제동토크, +방향=바퀴 회전 감속)
        """
        ax_ref = self.pedal_to_ax(pedal)          # 음수
        Fx_total = -ax_ref * self.m               # 제동력이므로 양수
        Fx_f, Fx_r = self.distribute_axles(Fx_total)

        # 축별 회생/마찰 블렌딩
        T_rg_f, T_fr_f = self.blend_regen_friction(Fx_f, wheels_front, is_drive_axle=(drive_axle=="front"))
        T_rg_r, T_fr_r = self.blend_regen_friction(Fx_r, wheels_rear,  is_drive_axle=(drive_axle=="rear"))

        # 바퀴별 합성 토크 + ABS + 액추에이터 지연
        T_cmd = [
            T_rg_f[0] + T_fr_f[0],
            T_rg_f[-1] + T_fr_f[-1],
            T_rg_r[0] + T_fr_r[0],
            T_rg_r[-1] + T_fr_r[-1],
        ]
        ws_all = [*wheels_front, *wheels_rear]

        T_out = [0.0]*4
        for i in range(4):
            T_abs = self._abs_wheel(T_cmd[i], veh_v, ws_all[i], i, dt)
            T_filt = self._first_order(self._T_cmd_prev[i], T_abs, dt)
            self._T_cmd_prev[i] = T_filt
            T_out[i] = T_filt

        return T_out  # [FL, FR, RL, RR]

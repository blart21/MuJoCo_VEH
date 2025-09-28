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
    def __init__(self, **kwargs): 
        """
        Args:
            mass_kg: 차량 질량 [kg]
            wheel_inertia: 바퀴 관성모멘트 [kg·m^2]
            n_front_wheels: 앞축 바퀴 수
            n_rear_wheels: 뒷축 바퀴 수
            front_bias: 앞축 제동 분배 비율
            pedal_ax_max: 페달 최대 종감속 [m/s^2]
            regen_max_torque: 회생제동 최대토크 [N·m]
            mu_tire: 타이어 마찰계수
            abs_enable: ABS 사용 여부
            slip_target: 슬립 목표값
            abs_kp: ABS 제어 피드백 이득
            abs_ki: ABS 제어 피드백 적분 이득
            tau_actuator: 액추에이터 지연 시간 [s]
            wheel_radius: 바퀴 반경 [m]
        """

        self.m = kwargs.get("mass_kg", 1500.0)
        self.Jw = kwargs.get("wheel_inertia", 1.2)
        self.nf = kwargs.get("n_front_wheels", 2)
        self.nr = kwargs.get("n_rear_wheels", 2)
        self.front_bias = kwargs.get("front_bias", 0.6)
        self.pedal_ax_max = float(kwargs.get("pedal_ax_max", -8.0))
        self.regen_max = kwargs.get("regen_max_torque", 1200.0)
        self.mu = kwargs.get("mu_tire", 0.9)
        self.abs_enable = kwargs.get("abs_enable", True)
        self.slip_target = kwargs.get("slip_target", 0.18)
        self.abs_kp = kwargs.get("abs_kp", 180.0)
        self.abs_ki = kwargs.get("abs_ki", 0.0)
        self.tau = max(1e-3, kwargs.get("tau_actuator", 0.04))
        self.wheel_radius = kwargs.get("wheel_radius", 0.3)

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
    def blend_regen_friction(self, 
    Fx_axle: float, wheel_states: list[WheelState], is_drive_axle: bool
    ) -> tuple[list[float], list[float]]:
        """
        축에 요구되는 제동력 Fx_axle(>0)를 바퀴별 회생/마찰 토크로 분해.
        회생은 '구동축'에서만 사용(보통 후륜모터/전륜모터 선택).

        Args:
            Fx_axle: 축에 요구되는 제동력 [N]
            wheel_states: 바퀴 상태 리스트
            is_drive_axle: 구동축 여부
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
    def _abs_wheel(self, **kwargs) -> float:
        T_cmd = kwargs.get("T_cmd", 0.0)
        veh_v = kwargs.get("veh_v", 0.0)
        ws = kwargs.get("ws", WheelState(w=0.0, R=0.0, load=0.0))
        idx = kwargs.get("idx", 0)
        dt = kwargs.get("dt", 0.0)

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
    def _first_order(self,
    u_prev: float, u_cmd: float, dt: float
    ) -> float:
        a = dt / (self.tau + dt)
        return (1 - a) * u_prev + a * u_cmd

    # ---------- 메인 API ----------
    def compute_wheel_torques(self,**kwargs) -> list[float]:
        """
        반환: [Tfl, Tfr, Trl, Trr] (모두 제동토크, +방향=바퀴 회전 감속)
        """
        pedal = kwargs.get("pedal", 0.0)
        veh_v = kwargs.get("veh_v", 0.0)
        wheels_front = kwargs.get("wheels_front", [])
        wheels_rear = kwargs.get("wheels_rear", [])
        dt = kwargs.get("dt", 0.0)
        drive_axle = kwargs.get("drive_axle", "rear")

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
            T_abs = self._abs_wheel(
                                        T_cmd=T_cmd[i],
                                        veh_v=veh_v,
                                        ws=ws_all[i],
                                        idx=i,
                                        dt=dt
                                    )
            T_filt = self._first_order(self._T_cmd_prev[i], T_abs, dt)
            self._T_cmd_prev[i] = T_filt
            T_out[i] = T_filt

        return T_out  # [FL, FR, RL, RR]

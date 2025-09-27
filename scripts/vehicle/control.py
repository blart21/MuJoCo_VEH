# scripts/vehicle/control.py

import numpy as np

def compose_control(action, suspension, **kwargs):
    """
    속도, 서스펜션, 조향 신호를 10개 actuator 입력 배열로 변환
      - speed_ctrl: 가속/후진 토크 (모터 토크 명령)
      - suspension_forces: 4개 액티브 서스펜션 힘
      - steer: 좌우 조향 입력 (-1.0 ~ 1.0)
      - brake: 0~1 브레이크 입력
      - brake_scale: 브레이크 토크 크기 스케일
    """
    controls = []

    speed_ctrl = action.get("throttle", 0.0) - action.get("reverse", 0.0)
    steer = action.get("steer", 0.0)
    brake = action.get("brake", 0.0)
    brake_scale = kwargs.get("brake_scale", 300.0)

    # ---- 모터 토크 (가속 - 브레이크) ----
    brake_torque = brake * brake_scale
    wheel_torque = speed_ctrl - brake_torque

    # 0~3: 4개 바퀴 구동 모터
    controls.append(wheel_torque)  # fl_motor
    controls.append(wheel_torque)  # fr_motor
    controls.append(wheel_torque)  # rl_motor
    controls.append(wheel_torque)  # rr_motor

    # 4~7: 액티브 서스펜션 force (FL, FR, RL, RR)
    forces = np.zeros(4)
    n = min(len(suspension), 4)
    forces[:n] = suspension[:n]
    controls.extend(forces)

    # 8~9: 앞바퀴 조향 (steer_fl_motor, steer_fr_motor)
    # 입력 steer(-1~1)을 rad 값으로 매핑
    max_steer = 0.8
    steer_val = np.clip(steer, -1.0, 1.0) * max_steer
    controls.append(steer_val)  # steer_fl
    controls.append(steer_val)  # steer_fr

    return np.array(controls)

def compute_suspension_forces(action=None, state=None):
    """
    서스펜션 4코너(FL, FR, RL, RR)에 가할 힘을 계산
    """
    if action is not None:
        return np.array(action[:4])
    return np.zeros(4)

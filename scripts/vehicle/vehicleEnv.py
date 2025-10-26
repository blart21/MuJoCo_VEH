# scripts/vehicle/vehicleEnv.py
from __future__ import annotations
import os
import numpy as np
import mujoco

from perception import LidarSensor
from .control import compose_control
from .ebrake import EBrake
from .aeb import AEBRadarMulti


class VehicleEnv:
    """
    단일 차량 + AEB 시뮬레이션 환경 래퍼.

    - scene/base_scene.xml 에 vehicle_active / vehicle_static / actuator 를 인라인 치환하여
      하나의 XML로 구성한 뒤 MjModel 생성
    - EBrake(마찰손실) + AEB(AEBRadarMulti)의 제동을 병행
    - step(action) 에서 baseline control → 페달 브레이크 → AEB 순서로 적용
    """

    def __init__(self, **kwargs):
        # ---------- 경로 ----------
        self.vehicle_active_path = kwargs.get("vehicle_active_path", "../models/vehicle/vehicle_active.xml")
        self.vehicle_static_path = kwargs.get("vehicle_static_path", "../models/vehicle/vehicle_static.xml")
        self.actuator_path       = kwargs.get("actuator_path",       "../models/vehicle/actuator.xml")
        self.scene_path          = kwargs.get("scene_path",          "../models/scene/base_scene.xml")

        # ---------- 모델 로딩/초기화 ----------
        xml = self._compose_model()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)

        # ---------- E-Brake(마찰손실) ----------
        self.ebrake = EBrake(
            model=self.model,
            data=self.data,
            frictionloss_max=2500.0,         # 제동 효과 강도
            tau_actuator=0.05,               # 응답 지연(작을수록 빠름)
            wheel_joint_names=["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"],
        )

        # (선택) 라이다 래퍼
        self.lidar = LidarSensor(self.model, self.data)

        # ---------- AEB(상·하 듀얼 라이다) ----------
        # - 하단 라이다(lidar_low)만 임계/거리 오버라이드로 더 민감하게
        self.aeb = AEBRadarMulti(
            site_names=("lidar_high", "lidar_low"),
            tilt_deg=0.0,
            ema_alpha=0.30,
            self_clearance=0.12,
            motor_brake_K=2000.0,
            clamp_ctrl=10000.0,
            zero_drive_when_aeb=True,
            static_brake_torque=5000.0,
            static_brake_vmin=0.05,
            verbose=False,
            per_site_cfg={
                "lidar_low": {                 # ▼ 저/낮은 물체 대응 강화
                    "dmin_on_override": 12.0,  # 켜짐 임계 거리
                    "dmin_off_override": 14.0, # 꺼짐 임계 거리(히스테리시스)
                    "max_dist_override": 90.0, # 최대 레이 길이
                }
            },
        )

        # (디버그) 휠 액추에이터 목록
        self._wheel_act_ids = []
        for name in ("fl_motor", "fr_motor", "rl_motor", "rr_motor"):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._wheel_act_ids.append(aid)

        print(
            "[VehicleEnv] AEB wired: sites=('lidar_high','lidar_low'), wheel_actuators:",
            [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self._wheel_act_ids],
            flush=True,
        )

        self.done = False

    # ---------- 모델 합성 ----------
    def _compose_model(self) -> str:
        """
        base_scene.xml 내부의 플레이스홀더를 실제 vehicle/actuator XML로 치환하여
        단일 XML 문자열을 반환.
        """
        base_dir = os.path.dirname(os.path.dirname(__file__))  # scripts/ 기준 상위가 프로젝트 루트

        active_path   = os.path.join(base_dir, self.vehicle_active_path)
        static_path   = os.path.join(base_dir, self.vehicle_static_path)
        actuator_path = os.path.join(base_dir, self.actuator_path)
        scene_path    = os.path.join(base_dir, self.scene_path)

        with open(active_path,   "r", encoding="utf-8") as f: active_xml   = f.read()
        with open(static_path,   "r", encoding="utf-8") as f: static_xml   = f.read()
        with open(actuator_path, "r", encoding="utf-8") as f: actuator_xml = f.read()
        with open(scene_path,    "r", encoding="utf-8") as f: scene_xml    = f.read()

        scene_xml = scene_xml.replace("<!--VEHICLE_ACTIVE INCLUDE-->", active_xml)
        scene_xml = scene_xml.replace("<!--VEHICLE_STATIC INCLUDE-->", static_xml)
        scene_xml = scene_xml.replace("<!--ACTUATOR INCLUDE-->",     actuator_xml)
        return scene_xml

    # ---------- 리셋 ----------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.done = False
        return self._get_obs()

    # ---------- 스텝 ----------
    def step(self, action: dict):
        """
        Args
        ----
        action: dict
            {"throttle": float, "reverse": float, "steer": float, "brake": float}
        """
        dt = float(self.model.opt.timestep)

        # 1) 기본 control 구성(엔진/조향/서스펜션)
        suspension = [0.0, 0.0, 0.0, 0.0]
        ctrl = compose_control(action, suspension)

        # 2) baseline ctrl 먼저 적용 (이후 AEB가 덮어씀)
        self.data.ctrl[:] = ctrl

        # 3) 운전자 브레이크(마찰손실 기반)
        self.ebrake.apply_brake(action.get("brake", 0.0), dt)

        # 4) AEB (활성 시 frictionloss + 휠 역토크 병행)
        info_aeb = self.aeb.apply(
            self.ebrake, t=self.data.time, model=self.model, data=self.data, dt=dt, brake_level=0.95
        )

        # 5) 물리 스텝
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done, info = 0.0, self.done, {"aeb": info_aeb}
        return obs, reward, done, info

    # ---------- 관측값 ----------
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

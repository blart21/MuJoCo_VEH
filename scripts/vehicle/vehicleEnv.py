# scripts/vehicle/vehicle.py

import mujoco
import numpy as np
import os

from perception import LidarSensor
from .control import compose_control
from .ebrake import EBrake, WheelState
class VehicleEnv:

    # Vehicle Environment 초기화
    def __init__(self, **kwargs):
        """
        Vehicle Environment 초기화

        Args:
            vehicle_active_path (str): Vehicle Active Model 경로
            vehicle_static_path (str): Vehicle Static Model 경로
            actuator_path (str): Actuator XML 경로
            scene_path (str): Scene Model 경로

        Returns:
            None

        """
        # 경로 
        self.vehicle_active_path = kwargs.get("vehicle_active_path", "../models/vehicle/vehicle_active.xml")
        self.vehicle_static_path = kwargs.get("vehicle_static_path", "../models/vehicle/vehicle_static.xml")
        self.actuator_path = kwargs.get("actuator_path", "../models/vehicle/actuator.xml")
        self.scene_path = kwargs.get("scene_path", "../models/scene/base_scene.xml")

        # Vehicle Model 구성
        xml = self._compose_model()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        sim = type("SimpleSim", (), {"model": self.model, "data": self.data})()
        self.ebrake = EBrake(model=self.model, data=self.data)

        #-------------------------------------------------
        self.lidar = LidarSensor(self.model, self.data) # Lidar 초기화
        #-------------------------------------------------
        self.done = False 

    # Vehicle Model 구성 
    def _compose_model(self) -> str:
        
        base_dir = os.path.dirname(os.path.dirname(__file__))  # 프로젝트 루트

        # 각 XML 경로
        active_path = os.path.join(base_dir, self.vehicle_active_path)
        static_path = os.path.join(base_dir, self.vehicle_static_path)
        actuator_path = os.path.join(base_dir, self.actuator_path)
        scene_path = os.path.join(base_dir, self.scene_path)

        # 파일 읽기
        with open(active_path, "r", encoding="utf-8") as f   : active_xml = f.read()
        with open(static_path, "r", encoding="utf-8") as f   : static_xml = f.read()
        with open(actuator_path, "r", encoding="utf-8") as f : actuator_xml = f.read()
        with open(scene_path, "r", encoding="utf-8") as f    : scene_xml = f.read()

        # placeholder 치환
        scene_xml = scene_xml.replace("<!--VEHICLE_ACTIVE INCLUDE-->", active_xml)
        scene_xml = scene_xml.replace("<!--VEHICLE_STATIC INCLUDE-->", static_xml)
        scene_xml = scene_xml.replace("<!--ACTUATOR INCLUDE-->", actuator_xml)

        return scene_xml

    # Simulation Reset
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.done = False
        return self._get_obs()

    # Vehicle Environment 진행 
    def step(self, action):
        """
        Args:
            action (dict): {"throttle", "reverse", "steer", "brake"}
        """
        # 1) 기본 control 계산
        suspension = [0.0, 0.0, 0.0, 0.0]
        ctrl = compose_control(action, suspension)

        # 2) 브레이크 계산
        veh_v = np.linalg.norm(self.data.qvel[:2])  # XY 속도
        wheel_states = [WheelState(w=0.0, R=0.3, load=self.ebrake.m * 9.81 / 4)] * 4
        torques = self.ebrake.compute_wheel_torques(
            pedal=action.get("brake", 0.0),
            veh_v=veh_v,
            wheels_front=wheel_states[:2],
            wheels_rear=wheel_states[2:],
            dt=self.model.opt.timestep
        )

        ctrl[:4] -= torques

        # MuJoCo step
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done, info = 0.0, self.done, {}

        return obs, reward, done, info

    # Vehicle Observation 반환 
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

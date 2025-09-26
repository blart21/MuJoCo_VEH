# scripts/vehicle/vehicle.py

import mujoco
import mujoco.viewer
import numpy as np
import os

from utils import Viewer
from perception import LidarSensor
from interface import InputManager

class VehicleEnv:

    # Vehicle Environment 초기화
    def __init__(self,**kwargs):
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
        # 경로 설정
        self.vehicle_active_path = kwargs.get("vehicle_active_path", "../models/vehicle/vehicle_active.xml")
        self.vehicle_static_path = kwargs.get("vehicle_static_path", "../models/vehicle/vehicle_static.xml")
        self.actuator_path = kwargs.get("actuator_path", "../models/vehicle/actuator.xml")
        self.scene_path = kwargs.get("scene_path", "../models/scene/base_scene.xml")

        # Vehicle Model 구성
        xml = self._compose_model()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        #-------------------------------------------------
        self.input_manager = InputManager() # Input Manager 초기화
        self.viewer = Viewer(self.model, self.data) # Viewer 초기화
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
        with open(active_path, "r", encoding="utf-8") as f:
            active_xml = f.read()
        with open(static_path, "r", encoding="utf-8") as f:
            static_xml = f.read()
        with open(actuator_path, "r", encoding="utf-8") as f:
            actuator_xml = f.read()
        with open(scene_path, "r", encoding="utf-8") as f:
            scene_xml = f.read()

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
        self.data.ctrl[:] = np.array(action)
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0  # placeholder
        done = self.done
        info = {}
        return obs, reward, done, info

    # Vehicle Observation 반환 
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    # Vehicle Rendering 
    def render(self):
        self.viewer.render()

    # Vehicle Environment 종료 
    def close(self):
        self.viewer.close()

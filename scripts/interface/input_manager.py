# scripts/interface/input_manager.py

import numpy as np
# from .keyboard import KeyboardController
# from .joystick import JoystickController
# from .wheel import WheelController

class InputManager:
    def __init__(self, mode="keyboard", **kwargs):
        """
        Args:
            mode (str): 입력 모드 ("keyboard", "joystick", "wheel")
        """
        self.mode = mode
        self.config = kwargs

        if self.mode == "keyboard":
            from pynput import keyboard
            self.controller = KeyboardController()
        elif self.mode == "joystick":
            import pygame
            self.controller = JoystickController(**kwargs)
        elif self.mode == "wheel":
            import pygame
            self.controller = WheelController(**kwargs)
        else:
            raise ValueError(f"Unknown input mode: {mode}")

    def get_action(self):
        """
        현재 입력 장치에서 액션을 읽어 반환
        Returns:
            np.ndarray: action vector
        """
        return self.controller.get_action()


# ----------------------------
# 세부 컨트롤러 (간단 뼈대-> 개별 파일로 분리 예정)
# ----------------------------

class KeyboardController:
    def __init__(self):
        # 키 상태 저장
        self.keys = set()

    def get_action(self):
        # 예: [throttle, brake, steer, handbrake]
        # 임시 placeholder (나중에 키 이벤트 등록 필요)
        return np.array([0.0, 0.0, 0.0, 0.0])


class JoystickController:
    def __init__(self, device_id=0):
        # pygame joystick 초기화 예정
        self.device_id = device_id

    def get_action(self):
        return np.array([0.0, 0.0, 0.0, 0.0])


class WheelController:
    def __init__(self, device_id=0):
        # 레이싱휠 초기화 예정
        self.device_id = device_id

    def get_action(self):
        return np.array([0.0, 0.0, 0.0, 0.0])

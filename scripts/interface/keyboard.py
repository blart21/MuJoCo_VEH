# scripts/interface/keyboard.py

import keyboard
import numpy as np

class KeyboardController:
    def __init__(self, **kwargs):
        """
        Keyboard 모듈 기반 컨트롤러
        - MuJoCo 기본 뷰어와 충돌 없음
        """
        self.throttle = 0.0
        self.reverse = 0.0
        self.steer = 0.0
        self.brake = 0.0

    def get_action(self):
        # 키 상태 읽기
        self.throttle = 1.0 if keyboard.is_pressed("w") else 0.0
        self.reverse  = 1.0 if keyboard.is_pressed("s") else 0.0
        self.steer    = (-1.0 if keyboard.is_pressed("a") else 0.0) \
                        + (1.0 if keyboard.is_pressed("d") else 0.0)
        self.brake    = 1.0 if keyboard.is_pressed("space") or keyboard.is_pressed("down") else 0.0

        action= {
            "throttle": self.throttle,
            "reverse": self.reverse,
            "steer": np.clip(self.steer, -1.0, 1.0),
            "brake": self.brake,
        }
        
        return action

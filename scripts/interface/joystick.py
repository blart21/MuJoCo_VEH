# scripts/interface/joystick.py

import numpy as np
class JoystickController:
    def __init__(self, device_id=0):
        # pygame joystick 초기화 예정
        self.device_id = device_id

    def get_action(self):
        return np.array([0.0, 0.0, 0.0, 0.0])
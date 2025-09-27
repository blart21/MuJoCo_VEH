# scripts/interface/input_manager.py
from .keyboard import KeyboardController
from .joystick import JoystickController
from .wheel import WheelController

class InputManager:
    def __init__(self, window, **kwargs):
        self.mode = kwargs.get("mode", "keyboard")

        if self.mode == "keyboard":
            self.controller = KeyboardController(**kwargs)
        elif self.mode == "joystick":
            self.controller = JoystickController(**kwargs)
        elif self.mode == "wheel":
            self.controller = WheelController(**kwargs)
        else:
            raise ValueError(f"Unknown input mode: {self.mode}")

    def get_action(self):
        return self.controller.get_action()

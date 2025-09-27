# scripts/main.py

from vehicle import VehicleEnv
from util.viewer import Viewer
from interface.input_manager import InputManager

def main():
    env = VehicleEnv()
    obs = env.reset()

    viewer = Viewer(env.model, env.data)
    input_mgr = InputManager(window=viewer.window, mode="keyboard")

    done = False
    while not done:
        # dict 형태로 입력 받기
        action = input_mgr.get_action()

        # high-level dict를 그대로 env에 넘김
        obs, reward, done, info = env.step(action)

        if not viewer.render():
            break

    viewer.close()

if __name__ == "__main__":
    main()

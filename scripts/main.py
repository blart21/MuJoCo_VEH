from vehicle import VehicleEnv

def main():
    env = VehicleEnv()
    obs = env.reset()
    done = False

    while not done:
        action = [0.0] * env.model.nu  # actuator 개수만큼 0 입력
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    main()

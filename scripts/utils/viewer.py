# scripts/utils/viewer.py

# scripts/utils/viewer.py

import mujoco
import mujoco.viewer


class Viewer:
    def __init__(self, model, data):
        """
        MuJoCo Viewer 래퍼
        Args:
            model (mujoco.MjModel): MuJoCo 모델
            data (mujoco.MjData): MuJoCo 데이터
        """
        self.model = model
        self.data = data
        self.viewer = None  # 필요할 때만 띄움

    def render(self):
        # 첫 호출 시 뷰어 실행
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 이미 띄워져 있으면 업데이트
        if self.viewer.is_running():
            self.viewer.sync()
        else:
            self.close()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

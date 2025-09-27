# scripts/perception/lidar.py

import numpy as np
import mujoco

class LidarSensor:
    def __init__(self, model, data, **kwargs):
        """
        단일 LiDAR 센서 클래스

        Args:
            model (mujoco.MjModel): MuJoCo 모델
            data (mujoco.MjData): MuJoCo 데이터
            site_name (str): LiDAR가 장착된 site 이름 (default: "lidar")
            num_rays (int): 발사할 레이 수 (default: 30)
            max_dist (float): 최대 탐지 거리 (default: 5.0)
            angle_range (tuple): 측정 각도 범위 (default: (30, 150))
        """
        self.model , self.data = model, data
        self.site_id = model.site(kwargs.get("site_name", "lidar")).id
        self.num_rays = kwargs.get("num_rays", 30)
        self.max_dist = kwargs.get("max_dist", 5.0)
        self.angle_range = kwargs.get("angle_range", (30, 150))

    def read(self):
        origin = self.data.site_xpos[self.site_id].copy()
        rot_mat = self.data.site_xmat[self.site_id].reshape(3, 3) # Local -> World

        forward = np.array([1, 0, 0])   # theta = 0
        up = np.array([0, 0, -1])       # theta = 90
        angles = np.radians(np.linspace(self.angle_range[0], self.angle_range[1], self.num_rays)) # 측정 각도

        scan = np.full(self.num_rays, self.max_dist, dtype=np.float32)
        geomgroup = None
        flg_static = 1
        bodyexclude = self.model.body("chassis").id
        geomid = np.array([-1], dtype=np.int32)

        for i, theta in enumerate(angles):
            dir_local = np.cos(theta) * forward + np.sin(theta) * up
            dir_world = rot_mat @ dir_local
            dist = mujoco.mj_ray(
                self.model, self.data, origin, dir_world,
                geomgroup, flg_static, bodyexclude, geomid
            )
            scan[i] = dist if geomid[0] != -1 else self.max_dist

        return scan 

        
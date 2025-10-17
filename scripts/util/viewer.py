# scripts/util/viewer.py
from __future__ import annotations
import math
import numpy as np
import glfw
import mujoco


# ---------- helpers ----------
def _name2id(model, kind: str, name: str) -> int:
    if kind == "site":
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def _fallback_target(model) -> tuple[str, int, str]:
    # 우선순위: lidar_* site → 대표 바디(chassis, vehicle_active 등) → 루트바디
    for i in range(model.nsite):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if nm.startswith("lidar"):
            return "site", i, nm
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if nm in ("chassis", "vehicle_active", "car", "car_body"):
            return "body", i, nm
    return "body", 0, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0) or "root"


class Viewer:
    """
    순수 추적 전용 커스텀 뷰어
    - 조명은 ‘적당히 어두운 중간 톤’으로만 세팅
    - 매 프레임 타깃 이름 재해결 + 폴백
    - 카메라는 (lookat, azimuth, elevation, distance)만 갱신
    """

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data

        # --- GLFW & GL context ---
        if not glfw.init():
            raise RuntimeError("GLFW init 실패")
        self.window = glfw.create_window(
            kwargs.get("width", 1280),
            kwargs.get("height", 720),
            kwargs.get("title", "MuJoCo Viewer (Follow Only)"),
            None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window 생성 실패")
        glfw.make_context_current(self.window)

        # --- MuJoCo structs ---
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption();  mujoco.mjv_defaultOption(self.opt)
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # === 렌더 플래그 (밝기 과함 방지 프리셋) ===
        for flag_name, value in [
            ("mjRND_LIGHT", True),         # 라이트 사용
            ("mjRND_LIGHTING", True),      # 이름 다를 수 있어 둘 다 켜줌
            ("mjRND_HEADLIGHT", False),    # 카메라 헤드라이트 끔
            ("mjRND_SHADOW", True),        # 그림자 켜서 대비감 확보
            ("mjRND_FOG", True),           # Fog ON: 배경 톤 제어
        ]:
            flag = getattr(mujoco.mjtRndFlag, flag_name, None)
            if flag is not None:
                try:
                    self.ctx.flags[flag] = value
                except Exception:
                    pass

        # === 전역 조명 톤다운 (global은 예약어 → getattr로 접근) ===
        try:
            vis_global = getattr(self.model.vis, "global", None) or getattr(self.model.vis, "global_", None)
            if vis_global is not None:
                vis_global.ambient[:]  = [0.25, 0.25, 0.25, 1.0]  # 전체 밝기 낮춤
                vis_global.diffuse[:]  = [0.65, 0.65, 0.65, 1.0]
                vis_global.specular[:] = [0.10, 0.10, 0.10, 1.0]
        except Exception:
            pass

        # === 배경을 짙은 회색으로: Fog 범위/색 ===
        try:
            self.model.vis.map.fogstart = 40.0     # 카메라에서 40m 이후부터 서서히
            self.model.vis.map.fogend   = 120.0    # 120m에서 완전히 배경색
            fog = self.model.vis.map.fogrgba
            fog[:4] = [0.16, 0.16, 0.16, 1.0]      # 배경 진회색
        except Exception:
            pass

        # --- mouse state ---
        self.last_x = self.last_y = 0.0
        self.button_left = self.button_right = self.button_middle = False

        # --- overlay ---
        self._overlay_lines: list[str] = []

        # --- follow state ---
        # dict: {kind, name, id, dist, elev, az_mode, fixed_az, yaw_off, off_world, off(np3), smooth, printed}
        self._follow: dict | None = None
        self._ema = {"lookat": None, "az": None, "el": None, "dist": None}

        # callbacks
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # ---------- overlay ----------
    def queue_overlay(self, *lines: str):
        self._overlay_lines = [str(s) for s in lines if s is not None][:2]

    # ---------- follow API ----------
    def enable_follow(
        self,
        *,
        mode: str = "site",         # "site" | "body"
        target_name: str = "lidar_front",
        distance: float = 12.0,
        elevation: float = -15.0,
        # azimuth 모드:
        #   "fixed"  : 월드 고정 방위각(차가 돌아도 카메라 방향 고정)
        #   "heading": 차량 x축(yaw)에 yaw_offset_deg를 더함(차와 함께 회전)
        azimuth_mode: str = "fixed",
        fixed_azimuth_deg: float = 90.0,
        yaw_offset_deg: float = 180.0,
        # lookat 위치 오프셋
        lookat_offset=(0.0, 0.0, 0.7),
        offset_in_world: bool = True,   # True면 월드 프레임, False면 타깃 로컬 프레임
        smooth: float = 0.25
    ):
        kind = "site" if mode == "site" else "body"
        tid = _name2id(self.model, kind, target_name)
        self._follow = dict(
            kind=kind, name=str(target_name), id=int(tid),
            dist=float(distance), elev=float(elevation),
            az_mode="fixed" if azimuth_mode != "heading" else "heading",
            fixed_az=float(fixed_azimuth_deg),
            yaw_off=float(yaw_offset_deg),
            off_world=bool(offset_in_world),
            off=np.array(lookat_offset, float),
            smooth=float(np.clip(smooth, 0.0, 1.0)),
            printed=False
        )
        self._ema = {"lookat": None, "az": None, "el": None, "dist": None}

        if tid < 0:
            k2, i2, n2 = _fallback_target(self.model)
            self._follow.update(kind=k2, id=i2, name=n2)
            print(f"[Viewer] follow target '{target_name}' not found → fallback to {k2}('{n2}')", flush=True)

    # ---------- callbacks ----------
    def _on_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:   self.button_left   = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:self.button_right  = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:self.button_middle= (action == glfw.PRESS)

    def _on_cursor(self, window, xpos, ypos):
        dx, dy = xpos - self.last_x, ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos
        if not (self.button_left or self.button_right or self.button_middle):
            return
        if self.button_left:   action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        elif self.button_right:action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif self.button_middle: action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else: action = None
        if action is not None:
            mujoco.mjv_moveCamera(self.model, action, dx/100.0, dy/100.0, self.scn, self.cam)

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05*yoffset, self.scn, self.cam)

    # ---------- render ----------
    def render(self) -> bool:
        if glfw.window_should_close(self.window):
            return False

        glfw.poll_events()

        # --- follow update ---
        f = self._follow
        if f is not None:
            # 1) id 재해결
            if f["id"] < 0:
                f["id"] = _name2id(self.model, f["kind"], f["name"])
                if f["id"] < 0:
                    k2, i2, n2 = _fallback_target(self.model)
                    f.update(kind=k2, id=i2, name=n2)
                    if not f["printed"]:
                        print(f"[Viewer] follow re-resolve failed → fallback to {k2}('{n2}')", flush=True)
                        f["printed"] = True

            # 2) pose 읽기
            pos = R = None
            if f["id"] >= 0:
                try:
                    if f["kind"] == "site":
                        pos = np.array(self.data.site_xpos[f["id"]], float)
                        R   = np.array(self.data.site_xmat[f["id"]], float).reshape(3,3)
                    else:
                        pos = np.array(self.data.xpos[f["id"]], float)
                        R   = np.array(self.data.xmat[f["id"]], float).reshape(3,3)
                except Exception:
                    pos = R = None

            # 3) 카메라 파라미터 계산/적용
            if pos is not None and R is not None:
                lookat_target = pos + (f["off"] if f["off_world"] else (R @ f["off"]))
                if f["az_mode"] == "fixed":
                    az_des = f["fixed_az"]
                else:
                    xaxis = R[:, 0]
                    yaw_world = math.degrees(math.atan2(xaxis[1], xaxis[0]))
                    az_des = yaw_world + f["yaw_off"]
                el_des = f["elev"]; dist_des = f["dist"]

                a = f["smooth"]
                def ema(prev, cur): return cur if prev is None else (a*cur + (1-a)*prev)
                self._ema["lookat"] = lookat_target if self._ema["lookat"] is None else a*lookat_target + (1-a)*self._ema["lookat"]
                self._ema["az"]  = ema(self._ema["az"],  az_des)
                self._ema["el"]  = ema(self._ema["el"],  el_des)
                self._ema["dist"]= ema(self._ema["dist"],dist_des)

                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.cam.lookat[:]  = self._ema["lookat"]
                self.cam.azimuth    = float(self._ema["az"])
                self.cam.elevation  = float(self._ema["el"])
                self.cam.distance   = float(self._ema["dist"])

        # --- scene & draw ---
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        w, h = glfw.get_framebuffer_size(self.window)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(vp, self.scn, self.ctx)

        # --- overlay ---
        if self._overlay_lines:
            l1 = self._overlay_lines[0] if len(self._overlay_lines) >= 1 else ""
            l2 = self._overlay_lines[1] if len(self._overlay_lines) >= 2 else ""
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               vp, l1, l2, self.ctx)
            self._overlay_lines = []

        glfw.swap_buffers(self.window)
        return True

    def close(self):
        try:
            if self.ctx is not None:
                mujoco.mjr_freeContext(self.ctx)
        except Exception:
            pass
        glfw.destroy_window(self.window)
        glfw.terminate()

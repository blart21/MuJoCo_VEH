# scripts/util/viewer.py
import glfw
import mujoco

class Viewer:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data

        if not glfw.init():
            raise RuntimeError("GLFW init 실패")

        self.window = glfw.create_window(
            kwargs.get("width", 1280),
            kwargs.get("height", 720),
            kwargs.get("title", "MuJoCo Custom Viewer"),
            None, None
        )
        glfw.make_context_current(self.window)

        # MuJoCo 렌더링 구조체
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)

        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # 마우스 상태 저장
        self.last_x, self.last_y = 0, 0
        self.button_left = False
        self.button_right = False
        self.button_middle = False

        # GLFW 콜백 등록
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

    # ---------------- 콜백 ---------------- #
    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.button_left = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.button_right = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.button_middle = (action == glfw.PRESS)

    def _cursor_pos_callback(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos

        if not (self.button_left or self.button_right or self.button_middle):
            return

        if self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        elif self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif self.button_middle:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            action = None

        if action is not None:
            mujoco.mjv_moveCamera(self.model, action, dx/100, dy/100, self.scn, self.cam)

    def _scroll_callback(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, self.scn, self.cam
        )

    # ---------------- 렌더링 ---------------- #
    def render(self):
        if glfw.window_should_close(self.window):
            return False

        glfw.poll_events()

        mujoco.mjv_updateScene(
            self.model, self.data, self.opt,
            None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        mujoco.mjr_render(viewport, self.scn, self.ctx)
        glfw.swap_buffers(self.window)
        return True

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()

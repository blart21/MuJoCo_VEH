# scripts/main.py
# ------------------------------------------------------------
# 목적:
#   - VehicleEnv 생성/리셋 → 입력 처리 → 스텝 → HUD 출력 → 커스텀 뷰어 렌더
#   - 카메라는 3인칭 측면 고정 방위각(차량 회전과 무관)으로 추적
# 구성:
#   VehicleEnv  : 모델/데이터 로딩 + 제동(AEB/EBrake) 적용
#   Viewer      : 커스텀 추적 카메라 & 오버레이 출력
#   InputManager: 키보드 입력 → action dict
#   hud_strings : 속도/레이더 거리 등 HUD 문자열 생성
# ------------------------------------------------------------

from __future__ import annotations
from vehicle.vehicleEnv import VehicleEnv
from util.viewer import Viewer
from interface.input_manager import InputManager
from overlay import hud_strings


def main():
    # ---------- 환경 초기화 ----------
    env = VehicleEnv()
    env.reset()

    # ---------- 뷰어 생성 ----------
    viewer = Viewer(env.model, env.data, title="MuJoCo Custom Viewer")

    # ---------- 카메라 추적 세팅 ----------
    # mode="site"        : site 기준 추적(차량 모델의 특정 지점)
    # target_name        : 기본은 "lidar_front". 만약 모델에 없으면 뷰어 내부에서 자동 폴백(lidar_* → chassis 등)
    # distance           : 타깃에서의 카메라 거리
    # elevation          : 음수면 약간 내려다보는 시점
    # azimuth_mode       : "fixed" → 월드 고정 방위각(차량 yaw와 무관), "heading" → 차량 전방(x축) 기준
    # fixed_azimuth_deg  : azimuth_mode="fixed"일 때 카메라 방위각(예: +x=0°, +y=90°)
    # lookat_offset      : 타깃 기준 lookat 오프셋(미세 상향/중앙 보정용)
    # offset_in_world    : True면 lookat_offset을 월드 좌표로, False면 타깃 로컬 좌표로 적용
    # smooth             : 카메라 파라미터 EMA(0~1) (값 클수록 현재 프레임을 더 따름)
    viewer.enable_follow(
        mode="site",
        target_name="lidar_front",
        distance=12.0,
        elevation=-15.0,
        azimuth_mode="fixed",
        fixed_azimuth_deg=0.0,
        lookat_offset=(0.0, 0.0, 0.7),
        offset_in_world=True,
        smooth=0.25,
    )

    # (디버그) 현재 장면의 라이트 개수 출력
    print("[Debug] lights:", env.model.nlight)

    # ---------- 입력 매니저 ----------
    input_mgr = InputManager(window=viewer.window, mode="keyboard")

    # ---------- 루프 ----------
    while True:
        # 1) 입력 → 액션
        action = input_mgr.get_action()

        # 2) 환경 스텝 (제동/AEB 등 내부 적용)
        obs, reward, done, info = env.step(action)

        # 3) HUD 문자열 큐잉 (렌더 전에 세팅)
        l1, l2 = hud_strings(
            env.model, env.data,
            aeb_info=info.get("aeb", {}), 
            speed_site_candidates=("lidar_front", "lidar_high", "lidar_low")
        )
        viewer.queue_overlay(l1, l2)

        # 4) 렌더 & 종료 조건
        if not viewer.render() or done:
            break

    # ---------- 종료 ----------
    viewer.close()


if __name__ == "__main__":
    main()

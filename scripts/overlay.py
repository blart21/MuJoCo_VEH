# scripts/overlay.py
from __future__ import annotations
import numpy as np
import mujoco
from .perception.aeb_lidar import get_aeb_scan_with_tilt


# ============================================================
# 전방 속도 계산(견고 버전)
#   - 우선순위: site → body → geom
#   - 어떤 경로가 사용됐는지 source 문자열로 함께 반환
#   - HUD 용도이므로 후진은 0으로 클램프(표시 안정성)
# ============================================================

def _try_get_site_forward_speed(model, data, site_name: str) -> tuple[float | None, str]:
    """
    지정 site의 월드 선형속도(전방 +x 성분)를 계산.
    성공 시 (v_fwd_mps, source), 실패 시 (None, reason)을 반환.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        return None, f"no_site:{site_name}"

    v6 = np.zeros(6, dtype=float)
    try:
        # v6 = [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z] (in world frame)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, v6, 0)
        v_world = v6[3:6]  # 선형속도
        src = f"siteV: {site_name}"
    except Exception:
        # 구버전 폴백: site가 붙은 바디의 선형속도 사용
        bid = int(model.site_bodyid[sid])
        if hasattr(data, "cvel") and data.cvel.shape[0] > bid:
            v_world = np.array(data.cvel[bid][3:6], dtype=float)
            src = f"bodyV[site]: {site_name}"
        else:
            return None, "no_cvel"

    # 로컬 변환: v_local = R^T * v_world → 로컬 x 성분
    R_sw = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)  # site->world
    v_local = R_sw.T @ v_world
    v_fwd = float(v_local[0])
    return v_fwd, src


def _try_get_body_forward_speed(model, data, body_name: str | None) -> tuple[float | None, str]:
    """
    지정 바디(없으면 루트 바디)의 월드 선형속도에서 바디 +x축 성분을 계산.
    """
    if body_name:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None, f"no_body:{body_name}"
    else:
        bid = 0  # 루트 바디

    if hasattr(data, "cvel") and data.cvel.shape[0] > bid:
        v_world = np.array(data.cvel[bid][3:6], dtype=float)
        R_bw = np.array(data.xmat[bid], dtype=float).reshape(3, 3)  # body->world
        v_local = R_bw.T @ v_world
        v_fwd = float(v_local[0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        return v_fwd, f"bodyV: {name}"
    return None, "no_cvel_body"


def _try_get_chassis_by_geom(model, data, geom_name: str = "chassis_geom") -> tuple[float | None, str]:
    """
    지정 지오메트리가 속한 바디를 찾아, 그 바디의 +x축 성분 속도를 계산.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return None, f"no_geom:{geom_name}"
    bid = int(model.geom_bodyid[gid])
    if hasattr(data, "cvel") and data.cvel.shape[0] > bid:
        v_world = np.array(data.cvel[bid][3:6], dtype=float)
        R_bw = np.array(data.xmat[bid], dtype=float).reshape(3, 3)  # body->world
        v_local = R_bw.T @ v_world
        v_fwd = float(v_local[0])
        return v_fwd, f"bodyV[geom]: {geom_name}"
    return None, "no_cvel_geom"


def get_forward_speed_robust(
    model,
    data,
    *,
    speed_site_candidates: tuple[str, ...] = ("lidar", "lidar_high", "lidar_low"),
    body_candidates: tuple[str | None, ...] = ("chassis", None),  # None = 루트 바디
    geom_candidates: tuple[str, ...] = ("chassis_geom",),
) -> tuple[float, float, str]:
    """
    가능한 모든 경로(site → body → geom)로 차량 전방속도를 구한다.
    Returns
    -------
    (v_mps_clamped, v_kmh_clamped, source_str)
    """
    # 1) site 우선
    for s in speed_site_candidates:
        v, src = _try_get_site_forward_speed(model, data, s)
        if v is not None:
            return v, 3.6 * v, src

    # 2) body
    for b in body_candidates:
        v, src = _try_get_body_forward_speed(model, data, b)
        if v is not None:
            return v, 3.6 * v, src

    # 3) geom
    for g in geom_candidates:
        v, src = _try_get_chassis_by_geom(model, data, g)
        if v is not None:
            return v, 3.6 * v, src

    # 실패 시
    return 0.0, 0.0, "speed:???"



# ============================================================
# 레이더(라이다) 거리 표시
#   - 중앙 레이(center_idx)의 거리만 HUD에 간단 표기
#   - get_aeb_scan_with_tilt를 그대로 재사용
# ============================================================

def _center_dist_str(
    model,
    data,
    site_name: str,
    *,
    tilt_deg: float = -1.0,
    max_dist: float = 80.0,
    fov_deg: float = 90.0,
    num_rays: int = 61,
) -> str:
    """지정 site 기준 중앙 레이 거리 문자열(예: '12.3m'). 실패 시 '???'."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        return "???"
    try:
        thetas, dists, gids, center_idx, _ = get_aeb_scan_with_tilt(
            model, data,
            site_name=site_name,
            num_rays=num_rays, fov_deg=fov_deg, max_dist=max_dist,
            self_clearance=0.12, tilt_deg=tilt_deg
        )
        if center_idx is None or center_idx >= len(dists):
            return "???"
        return f"{float(dists[center_idx]):4.1f}m"
    except Exception:
        return "???"



# ============================================================
# HUD 문자열 빌더
# ============================================================

def hud_strings(
    model,
    data,
    *,
    aeb_info: dict,
    speed_site_candidates: tuple[str, ...] = ("lidar", "lidar_high", "lidar_low"),
    **kwargs,
) -> tuple[str, str]:
    """
    뷰어 오버레이용 텍스트 두 줄을 만들어 반환.
    """
    # 1) 속도 정보
    v_mps, v_kmh, v_src = get_forward_speed_robust(
        model, data,
        speed_site_candidates=speed_site_candidates,
        body_candidates=("chassis", None),
        geom_candidates=("chassis_geom",),
    )
    line1 = f"Time: {data.time:6.2f}s | Speed: {v_mps:4.2f} m/s  ({v_kmh:4.1f} km/h)  [{v_src}]"

    # 2) AEB 상태, 거리, TTC
    dist = aeb_info.get("dmin_ema", float('inf'))
    ttc = aeb_info.get("ttc", float('inf'))
    state = aeb_info.get("aeb_state", "N/A")

    dist_str = f"{dist:4.1f}m" if dist < 1000 else "---"
    ttc_str = f"{ttc:4.2f}s" if ttc < 1000 else "---"

    line2 = f"AEB: {state:<9} | Dist: {dist_str} | TTC: {ttc_str}"

    return line1, line2
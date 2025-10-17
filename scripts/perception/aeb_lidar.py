# scripts/perception/aeb_lidar.py
# ------------------------------------------------------------
# AEB 레이더/라이다 스캔 유틸
#  - 지정한 site에서 수평 FOV로 N개의 광선을 쏴서 가장 가까운 hit 거리/지오메트리 id를 구함
#  - 로컬 전방축 선택(x, y, -x, -y) + 소폭 pitch(tilt) 지원
#  - 자차/초근접 노이즈 제거(self_clearance)
#  - MuJoCo 버전별 mj_ray 호환 처리(_mj_ray_compat)
#  - 상위에서 로그 찍기 쉽도록 중앙 레이 인덱스(center_idx)와 발사 원점(origin)까지 함께 반환
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import mujoco


# ---------- 간단 회전 행렬 유틸 ----------
def _rotz(a: float) -> np.ndarray:
    """z축 기준 회전(요aw)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def _roty(a: float) -> np.ndarray:
    """y축 기준 회전(피치)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=float)


# ---------- MuJoCo mj_ray 버전 호환 래퍼 ----------
def _mj_ray_compat(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    p_world: np.ndarray,
    v_world: np.ndarray,
    *,
    bodyexclude: int = -1,
    flg_static: int = 0,
    geomgroup: np.ndarray | None = None,
) -> tuple[int, float]:
    """
    mj_ray 시그니처가 버전마다 다른 이슈를 흡수하는 래퍼.
    - 최신: mj_ray(model, data, p, v, geomgroup[6], flg_static, bodyexclude, geomid[1])
    - 구버전: mj_ray(model, data, p, v) -> (gid, dist)
    """
    p = np.asarray(p_world, float).reshape(3)
    v = np.asarray(v_world, float).reshape(3)
    gg = np.ones(6, np.uint8) if geomgroup is None else np.asarray(geomgroup, np.uint8).reshape(6)
    try:
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, p, v, gg, int(flg_static), int(bodyexclude), geomid)
        return int(geomid[0]), float(dist)
    except TypeError:
        gid, dist = mujoco.mj_ray(model, data, p, v)
        return int(gid), float(dist)


# ---------- 로컬 전방축 선택 ----------
def _axis_vec(axis: str) -> np.ndarray:
    """
    로컬 전방축 문자열을 단위 벡터로 변환.
    허용값: 'x', '-x', 'y', '-y' (기본값 'x').
    """
    a = axis.lower()
    if a == 'x':   return np.array([ 1, 0, 0], float)
    if a == '-x':  return np.array([-1, 0, 0], float)
    if a == 'y':   return np.array([ 0, 1, 0], float)
    if a == '-y':  return np.array([ 0,-1, 0], float)
    # fallback
    return np.array([1, 0, 0], float)


# ---------- 메인: AEB 스캔 with tilt ----------
def get_aeb_scan_with_tilt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_name: str = "lidar",
    num_rays: int = 51,
    fov_deg: float = 90.0,
    max_dist: float = 80.0,
    self_clearance: float = 0.15,
    tilt_deg: float = 0.0,          # 라이다 틸트(피치, +면 위로, -면 아래로)
    forward_axis: str = "x",        # 'x'|'y'|'-x'|'-y' (site의 로컬 전방축)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    지정 site에서 수평 FOV를 num_rays로 균등 분할해 레이캐스트.
    Returns:
        thetas     (rad) : 각 레이의 요(좌우) 각도 배열(-FOV/2 ~ +FOV/2)
        dists      (m)   : 각 레이의 히트 거리(미히트는 max_dist로 채움)
        gids       (id)  : 각 레이가 맞춘 geom id(미히트는 -1)
        center_idx (int) : 중앙 레이 인덱스(디버그용)
        origin     (m,3) : 발사 원점(site world pos)
    Raises:
        KeyError: site_name이 유효하지 않을 때
    """
    # 1) site pose
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise KeyError(f"Invalid site '{site_name}'")

    origin = np.array(data.site_xpos[sid], dtype=float)             # 월드 기준 원점
    R_sw   = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)  # site->world 회전

    # 2) 레이 각도/전방축
    fov_rad = np.deg2rad(fov_deg)
    thetas  = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, num_rays)
    tilt    = np.deg2rad(tilt_deg)          # 피치(아래로 내리려면 음수)
    fwd_local = _axis_vec(forward_axis)     # 로컬 전방축

    # 3) 결과 버퍼
    dists = np.full(num_rays, max_dist, dtype=float)
    gids  = np.full(num_rays, -1, dtype=int)

    # 4) 자차 자체 히트 제거용 bodyexclude (site가 붙은 body)
    bodyexclude = int(model.site_bodyid[sid]) if hasattr(model, "site_bodyid") else -1

    # 5) 각 레이 캐스팅
    for i, yaw in enumerate(thetas):
        # 로컬에서 yaw(좌우) → tilt(피치) 적용 후, 월드 방향으로 변환
        dir_local = _roty(tilt) @ (_rotz(yaw) @ fwd_local)
        dir_world = R_sw @ dir_local
        n = np.linalg.norm(dir_world)
        if n < 1e-12:
            continue
        dir_world /= n

        gid, dist = _mj_ray_compat(
            model, data, origin, dir_world,
            bodyexclude=bodyexclude, flg_static=1, geomgroup=None
        )

        # 초근접/자차 주변 노이즈 제거 → 미히트 취급
        if dist < self_clearance:
            gid, dist = -1, max_dist

        if gid >= 0:
            dists[i] = min(dist, max_dist)
            gids[i]  = gid
        else:
            dists[i] = max_dist
            gids[i]  = -1

    # 중앙 레이 인덱스(디버깅/로그용)
    center_idx = int(np.argmin(np.abs(thetas)))

    return thetas, dists, gids, center_idx, origin

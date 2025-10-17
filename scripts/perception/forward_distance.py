# scripts/perception/forward_distance.py
from __future__ import annotations
import numpy as np
import mujoco


def _axis_and_origin_from_site(model: mujoco.MjModel, data: mujoco.MjData, site_name: str):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise KeyError(f"Invalid site '{site_name}'")
    origin = np.array(data.site_xpos[sid], dtype=float)           # (3,)
    R = np.array(data.site_xmat[sid], dtype=float).reshape(3, 3)  # world<-local
    fwd = R[:, 0]                                                 # local x-axis in world
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    left = R[:, 1]; up = R[:, 2]                                  # 참고용
    return origin, fwd, left, up


def _geom_info(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise KeyError(f"Invalid geom '{geom_name}'")
    gpos = np.array(data.geom_xpos[gid], dtype=float)            # center (world)
    gR = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    gtype = int(model.geom_type[gid])
    gsize = np.array(model.geom_size[gid], dtype=float)          # meaning depends on type
    return gid, gpos, gR, gtype, gsize


def _oriented_box_half_extent_along(dir_world: np.ndarray, R: np.ndarray, halfsizes: np.ndarray) -> float:
    """
    Oriented box extent along a direction: sum_i |dir · axis_i| * halfsize_i
    where axis_i = R[:, i]
    """
    ax = np.abs(np.dot(dir_world, R[:, 0])) * halfsizes[0]
    ay = np.abs(np.dot(dir_world, R[:, 1])) * halfsizes[1]
    az = np.abs(np.dot(dir_world, R[:, 2])) * halfsizes[2]
    return float(ax + ay + az)


def distance_along_forward_to_geom_face(model: mujoco.MjModel, data: mujoco.MjData,
                                        from_site: str, to_geom: str,
                                        own_front_geom: str | None = "chassis_geom") -> dict:
    """
    차량의 from_site 전방축(로컬 x) 기준으로, 목표 지오메트리(to_geom)의 '앞쪽 면'까지의
    1D 거리(전방축 투영)를 근사 계산한다. (라이다 미사용)

    Returns:
      {
        'axis_distance_center':  float,   # from_site→target center 의 전방축 투영 거리 [m]
        'axis_distance_to_face': float,   # target 앞면까지의 거리(중심거리 - target half-extent) [m]
        'axis_distance_bumper_to_face': float or None,  # own_front_geom 고려 시: (from_site에서 own 앞면까지 보정 후)
        'lateral_offset':        float,   # 전방축에 수직한 측방 오프셋(좌우) [m]
        'target_center_distance': float,  # 실제 유클리드 중심간 거리 [m]
      }
    """
    origin, fwd, left, up = _axis_and_origin_from_site(model, data, from_site)
    gid, gpos, gR, gtype, gsize = _geom_info(model, data, to_geom)

    r = gpos - origin
    axis_distance_center = float(np.dot(r, fwd))                # 전방축 1D 투영
    lateral_offset = float(np.dot(r, left))                     # 좌우(+좌, -우)
    target_center_distance = float(np.linalg.norm(r))

    # target 앞면까지 (지오메 타입별 근사)
    axis_distance_to_face = axis_distance_center
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        # box: gsize = [hx, hy, hz] (half lengths)
        ext = _oriented_box_half_extent_along(fwd, gR, gsize[:3])
        axis_distance_to_face = axis_distance_center - ext
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        # sphere: gsize[0] = radius
        axis_distance_to_face = axis_distance_center - float(gsize[0])
    elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        # 대략: 반지름으로 근사(높이방향 정렬 가정이 약함)
        radius = float(gsize[0])
        axis_distance_to_face = axis_distance_center - radius
    else:
        # mesh 등: 중심까지의 축거리만 반환 (필요시 AABB로 확장 가능)
        axis_distance_to_face = axis_distance_center

    # own_front_geom(예: 'chassis_geom')의 "앞면"까지 보정 → 범퍼 기준 거리 제공
    axis_distance_bumper_to_face = None
    if own_front_geom:
        try:
            _, spos, sR, stype, ssize = _geom_info(model, data, own_front_geom)
            if stype == mujoco.mjtGeom.mjGEOM_BOX:
                self_ext = _oriented_box_half_extent_along(fwd, sR, ssize[:3])
                # from_site→자차 앞면까지는 'self_ext - (site가 그 중심에서 얼마나 앞/뒤인지)'가 정확하지만
                # 여기서는 site가 차체 중심 근처라는 전제 하에 self_ext를 그대로 사용
                axis_distance_bumper_to_face = axis_distance_to_face - self_ext
            elif stype == mujoco.mjtGeom.mjGEOM_SPHERE:
                axis_distance_bumper_to_face = axis_distance_to_face - float(ssize[0])
            else:
                axis_distance_bumper_to_face = axis_distance_to_face
        except Exception:
            axis_distance_bumper_to_face = None

    return dict(
        axis_distance_center=axis_distance_center,
        axis_distance_to_face=axis_distance_to_face,
        axis_distance_bumper_to_face=axis_distance_bumper_to_face,
        lateral_offset=lateral_offset,
        target_center_distance=target_center_distance,
    )

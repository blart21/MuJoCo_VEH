# scripts/vehicle/aeb.py
# ------------------------------------------------------------
# 목적: AEB 판단 및 제동 적용
# 요약
#  - 라이다 스캔에서 '바닥' + '자차 차체'만 제외(타차는 이름 같아도 감지)
#  - dmin: 유효 hit 중 최소 거리 → EMA 평활
#  - TTC(Time to Collision) 기반 3-상태 머신(Standby, Warning, Braking)으로 제어
#  - AEB 활성 시: EBrake(frictionloss) + 휠 모터 역토크 병행
#  - per-site 임계/거리 오버라이드 지원(AEBRadarMulti에서 사용)
# ------------------------------------------------------------

from __future__ import annotations
from enum import IntEnum
import numpy as np
import mujoco
from ..perception.aeb_lidar import get_aeb_scan_with_tilt


# AEB 상태를 정의하는 Enum 클래스
class AEBState(IntEnum):
    STANDBY = 0
    WARNING = 1
    BRAKING = 2


# ============================== 싱글 레이더 ==============================
class AEBRadar:
    """
    단일 라이다 사이트(site_name)를 이용한 AEB 판단 및 제동 주입기.
    """

    def __init__(
        self,
        *,
        site_name: str = "lidar",
        tilt_deg: float = -2.0,
        ema_alpha: float = 0.25,
        self_clearance: float = 0.20,
        motor_names: tuple[str, ...] = ("fl_motor", "fr_motor", "rl_motor", "rr_motor"),
        motor_brake_K: float = 600.0,
        clamp_ctrl: float = 10000.0,
        zero_drive_when_aeb: bool = True,
        static_brake_torque: float = 1800.0,
        static_brake_vmin: float = 0.2,
        verbose: bool = False,
        max_dist_override: float | None = None,
        **kwargs,
    ):
        self.site = site_name
        self.tilt_deg = float(tilt_deg)
        self.alpha = float(ema_alpha)
        self.self_clearance = float(self_clearance)

        self._dmin_ema: float | None = None
        self._was_active: bool = False
        self._aeb_state: AEBState = AEBState.STANDBY

        self._motor_names = tuple(motor_names)
        self._motor_brake_K = float(motor_brake_K)
        self._clamp_ctrl = float(clamp_ctrl)
        self._zero_drive_when_aeb = bool(zero_drive_when_aeb)
        self._static_brake_torque = float(static_brake_torque)
        self._static_brake_vmin = float(static_brake_vmin)

        self._act_ids: list[int] = []
        self._resolved = False
        self.verbose = bool(verbose)

        self.EXCLUDE_GEOMS = {"floor", "chassis_geom"}
        self._own_body: int | None = None
        self._max_dist_override = max_dist_override

        # 역주행 방지 & 정지 유지 로직
        self.prevent_reverse: bool = bool(kwargs.get("prevent_reverse", True))
        self.stop_hold_vth: float = float(kwargs.get("stop_hold_vth", 0.15))   # 이 속도 이하로 떨어지면 '홀드' 진입
        self.release_vth: float   = float(kwargs.get("release_vth", 0.40))     # 이 속도 이상이면 '홀드' 해제
        self._hold_mode: bool = False

    @staticmethod
    def _kmh(v_mps: float) -> float:
        return 3.6 * v_mps

    def _resolve_actuators_once(self, model: mujoco.MjModel) -> None:
        if self._resolved:
            return
        self._act_ids.clear()
        for name in self._motor_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._act_ids.append(int(aid))

        if not self._act_ids:
            # fallback: 이름에 "_motor" 포함된 액추에이터 자동 탐색
            for a in range(model.nu):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
                if "_motor" in nm:
                    self._act_ids.append(int(a))

        if not self._act_ids and model.nu > 0:
            # 최후: 앞의 몇 개를 사용
            self._act_ids = list(range(min(4, model.nu)))

        self._resolved = True
        if self.verbose:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self._act_ids]
            print(f"[AEB] resolved wheel actuators: {names}", flush=True)

    def _vehicle_forward_speed(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        # site body의 속도를 바디 로컬 +x로 투영
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
        if sid < 0:
            return 0.0
        bid = int(model.site_bodyid[sid])
        R_wb = data.xmat[bid].reshape(3, 3).T
        v_world = (np.array(data.cvel[bid][3:6], float) if hasattr(data, "cvel")
                   else np.array(data.qvel[:3], float))
        v_local_linear = R_wb @ v_world[0:3]
        return float(v_local_linear[0])

    def _adaptive_cfg(self, v_mps: float) -> dict:
        """
        라이다 스캔 설정을 반환합니다.
        """
        # 라이다 최대 감지 거리를 100m로 설정
        cfg = dict(max_dist=100.0, fov_deg=90.0, num_rays=71)

        # (참고) 만약 차량의 속도에 따라 설정을 바꾸고 싶다면 아래처럼 할 수 있습니다.
        # v_kmh = self._kmh(v_mps)
        # if v_kmh > 80.0:
        #     cfg['fov_deg'] = 60.0 # 예: 고속에서는 전방 집중을 위해 화각(FOV)을 좁힘

        # vehicleEnv.py에서 per-site 오버라이드 값이 설정된 경우, 그 값을 우선 적용
        if self._max_dist_override is not None:
            cfg["max_dist"] = float(self._max_dist_override)

        return cfg

    def _is_excluded(self, model: mujoco.MjModel, gid: int) -> bool:
        if gid < 0:
            return True
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name == "floor":
            return True
        if name == "chassis_geom":
            # 자신의 차체만 제외(다른 차량의 같은 이름 지오메트리는 감지 대상)
            if self._own_body is None:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
                self._own_body = int(model.site_bodyid[sid]) if sid >= 0 else -1
            return int(model.geom_bodyid[gid]) == self._own_body
        return False

    def update(self, t: float, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, dict]:
        v_fwd = max(0.0, self._vehicle_forward_speed(model, data))
        cfg = self._adaptive_cfg(v_fwd)

        _, dists, gids, _, _ = get_aeb_scan_with_tilt(
            model, data,
            site_name=self.site,
            num_rays=cfg["num_rays"],
            fov_deg=cfg["fov_deg"],
            max_dist=cfg["max_dist"],
            self_clearance=self.self_clearance,
            tilt_deg=self.tilt_deg,
        )

        valid_dists = [d for d, g in zip(dists, gids) if not self._is_excluded(model, int(g))]
        if not valid_dists:
            valid_dists = [cfg["max_dist"]]

        dmin_raw = float(np.min(valid_dists))
        self._dmin_ema = dmin_raw if self._dmin_ema is None else self.alpha * dmin_raw + (1.0 - self.alpha) * self._dmin_ema

        ttc = (self._dmin_ema / max(1e-2, v_fwd)) if v_fwd > 0.1 else float("inf")

        # 상태 머신
        if self._aeb_state == AEBState.STANDBY:
            if ttc <= 3.0:
                self._aeb_state = AEBState.WARNING
                if self.verbose:
                    print(f"[{t:.2f}s] AEB State -> WARNING (TTC: {ttc:.2f}s)")
        elif self._aeb_state == AEBState.WARNING:
            if ttc <= 2.5:
                self._aeb_state = AEBState.BRAKING
                if self.verbose:
                    print(f"[{t:.2f}s] AEB State -> BRAKING (TTC: {ttc:.2f}s)")
            elif ttc > 3.0:
                self._aeb_state = AEBState.STANDBY
                if self.verbose:
                    print(f"[{t:.2f}s] AEB State -> STANDBY (TTC: {ttc:.2f}s)")
        elif self._aeb_state == AEBState.BRAKING:
            if ttc > 2.5:
                self._aeb_state = AEBState.WARNING
                if self.verbose:
                    print(f"[{t:.2f}s] AEB State -> WARNING (TTC: {ttc:.2f}s)")

        active_now = (self._aeb_state == AEBState.BRAKING)

        info = {
            "dmin_raw": dmin_raw,
            "dmin_ema": float(self._dmin_ema),
            "v_forward_mps": v_fwd,
            "ttc": ttc,
            "trigger": active_now,
            "aeb_state": self._aeb_state.name,
            "scan_cfg": cfg,
        }
        return active_now, info

    def _apply_motor_brake(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._resolve_actuators_once(model)
        if int(data.ncon) == 0:
            return

        v_fwd = self._vehicle_forward_speed(model, data)

        # ✅ 홀드 모드 상태 업데이트(히스테리시스)
        if self.prevent_reverse:
            if self._hold_mode:
                if v_fwd > self.release_vth:
                    self._hold_mode = False
            else:
                if v_fwd < self.stop_hold_vth:
                    self._hold_mode = True

        for aid in self._act_ids:
            if not (0 <= aid < model.nu):
                continue
            if self._zero_drive_when_aeb:
                data.ctrl[aid] = 0.0

            dof = int(model.actuator_trnid[aid][0])
            if not (0 <= dof < model.nv):
                continue

            w = float(data.qvel[dof])

            # --- 안전 파라미터(필요시 미세조정) ---
            v_lock_eps = 0.07    # 차량 전방속도가 이하면 '거의 정지'로 간주
            w_stop_eps = 0.06    # 휠 각속도가 이하면 0으로 스냅
            tau_cap_hi = 4000.0  # 고속구간에서 최대 브레이크 토크 캡
            tau_cap_lo = 800.0   # 정지 근처에서의 최대 토크 캡(너무 크면 역주행 유발)
            K_visc_base = self._motor_brake_K  # 점성 계수(기존 값 재사용)

            # 1) 정지 스냅: 거의 정지면 각속도/가속 0으로 고정해 역회전 싹둑
            if abs(v_fwd) < v_lock_eps and abs(w) < w_stop_eps:
                data.qvel[dof] = 0.0
                if hasattr(data, "qacc"):
                    data.qacc[dof] = 0.0
                if hasattr(data, "qfrc_applied"):
                    data.qfrc_applied[dof] = 0.0

            # 2) 속도에 따른 토크 캡: 정지 근처일수록 토크 상한을 낮춰 ‘밀어넘김’ 방지
            #    0~v_lock_eps 범위에서 선형 램프업 (lo → hi)
            if abs(v_fwd) < v_lock_eps:
                ramp = abs(v_fwd) / max(1e-6, v_lock_eps)
                tau_cap = tau_cap_lo + (tau_cap_hi - tau_cap_lo) * ramp
            else:
                tau_cap = tau_cap_hi

            # 3) 기본은 소산적 점성(항상 u*w <= 0가 되도록)
            u = -K_visc_base * w

            # 4) 정지 토크(고정 마찰 유사)의 ‘조건부’ 추가:
            #    - 진행방향 속도가 충분히 있을 때만
            #    - 바퀴 회전 방향이 ‘진행방향과 같은’ 경우에만 (역방향이면 0)
            #      → 역주행으로 밀지 않음 (one-way clutch 효과)
            same_dir = (np.sign(w) == np.sign(v_fwd)) or (abs(w) < 1e-6) or (abs(v_fwd) < 1e-6)
            if v_fwd > max(self._static_brake_vmin, self.stop_hold_vth) and same_dir:
                u += -self._static_brake_torque * np.sign(v_fwd)

            # 5) 항상 ‘소산성’ 보장: u*w > 0 이면(모터가 에너지 공급) 즉시 점성만 남기기
            if u * w > 0:
                u = -K_visc_base * w

            # 6) 역회전 방지(홀드 모드)와 결합:
            if self._hold_mode:
                # 홀드에서는 점성만 허용 + 토크 캡 더 보수적으로
                u = -K_visc_base * w
                tau_cap = min(tau_cap, tau_cap_lo)

            # 7) 최종 클램프
            u = float(np.clip(u, -min(tau_cap, self._clamp_ctrl), min(tau_cap, self._clamp_ctrl)))
            data.ctrl[aid] += u

    def apply(
        self, ebrake, t: float, model: mujoco.MjModel, data: mujoco.MjData,
        dt: float, brake_level: float = 0.95,
    ) -> dict:
        active, info = self.update(t, model, data)
        if active and not self._was_active:
            if self.verbose:
                print("[AEB] 작동중 (engaged)", flush=True)
        elif not active and self._was_active:
            if self.verbose:
                print("[AEB] 해제됨 (released)", flush=True)
        self._was_active = active

        if active and int(data.ncon) > 0:
            ebrake.apply_brake(brake_level, dt)
            self._apply_motor_brake(model, data)

        return info


# ============================== 멀티 레이더 ==============================
class AEBRadarMulti:
    def __init__(self, **kwargs):
        # [수정] 불필요하고 오류를 유발하는 import 구문을 제거하고 직접 클래스를 할당합니다.
        _Single = AEBRadar
        self._radars: list[_Single] = []
        self.verbose = kwargs.get("verbose", True)
        self._was_active = False

        # ---- [추가] 독립 Stop-Latch 파라미터 ----
        self.stop_enable: bool = bool(kwargs.get("stop_enable", True))
        # 정지 진입 조건: "거리 ≤ stop_dist_m AND 속도 ≤ stop_speed_kmh"
        self.stop_dist_m: float = float(kwargs.get("stop_dist_m", 2.0))
        self.stop_speed_kmh: float = float(kwargs.get("stop_speed_kmh", 5.0))
        # 정지 해제(출발) 조건: "거리 ≥ release_dist_m AND 속도 ≥ release_speed_kmh"
        self.release_dist_m: float = float(kwargs.get("stop_release_dist_m", 2.5))
        self.release_speed_kmh: float = float(kwargs.get("stop_release_speed_kmh", 7.0))
        # 내부 래치 상태
        self._stop_latch: bool = False

        site_names = kwargs.get("site_names", ("lidar_high", "lidar_low"))
        per_site_cfg = kwargs.get("per_site_cfg", {})
        base_cfg = kwargs.copy()

        for sn in site_names:
            site_specific_cfg = base_cfg.copy()
            site_overrides = per_site_cfg.get(sn, {})
            site_specific_cfg.update(site_overrides)
            site_specific_cfg["site_name"] = sn
            self._radars.append(_Single(**site_specific_cfg))

    @staticmethod
    def _fuse_active_info(infos: list[dict]) -> dict:
        if not infos:
            return {}
        best = min(infos, key=lambda x: x.get("dmin_ema", float('inf')))
        fused = dict(best)
        fused["ttc"] = min(i.get("ttc", float('inf')) for i in infos)
        fused["trigger"] = any(i.get("trigger", False) for i in infos)
        fused["dmin_raw_min"] = min(i.get("dmin_raw", float('inf')) for i in infos)
        return fused

    def _force_stop_hold(self, model, data, dt):
        """
        Stop-Latch가 걸린 상태에서 '완전 정지 유지' 동작을 수행.
        - EBrake 최대(외부에서 apply 호출)
        - 바퀴 모터 입력 0
        - 정지 근처 각속도는 0으로 스냅(역회전 방지)
        """
        rep = self._radars[0] if self._radars else None
        if rep is None:
            return
        rep._resolve_actuators_once(model)
        for aid in rep._act_ids:
            if 0 <= aid < model.nu:
                data.ctrl[aid] = 0.0
                dof = int(model.actuator_trnid[aid][0])
                if 0 <= dof < model.nv:
                    if abs(data.qvel[dof]) < 0.6:
                        data.qvel[dof] = 0.0
                    if hasattr(data, "qacc"):
                        data.qacc[dof] = 0.0

    def apply(self, ebrake, t, model, data, dt, brake_level: float = 0.95) -> dict:
        # 1) 모든 레이더 업데이트(아직 제동은 안 넣음)
        all_infos = []
        fused_active = False
        for rad in self._radars:
            active, info = rad.update(t, model, data)
            all_infos.append(info)
            if active:
                fused_active = True

        fused_info = self._fuse_active_info(all_infos)

        # 2) 독립 Stop-Latch 판단 (AEB 브레이킹/역토크와 별개 로직)
        if self.stop_enable:
            d_ema = float(fused_info.get("dmin_ema", float("inf")))
            d_raw_min = float(fused_info.get("dmin_raw_min", d_ema))
            v_mps = float(fused_info.get("v_forward_mps", 0.0))
            v_kmh = 3.6 * v_mps

            # 진입 판단: 보수적으로 raw 또는 ema 중 더 작은 값 사용
            d_for_entry = min(d_ema, d_raw_min)

            # 진입 조건: 거리 ≤ stop_dist_m AND 속도 ≤ stop_speed_kmh
            if (not self._stop_latch) and (d_for_entry <= self.stop_dist_m) and (v_kmh <= self.stop_speed_kmh):
                self._stop_latch = True
                if self.verbose:
                    print(f"[AEB* STOP] latch ON (d={d_for_entry:.2f} m, v={v_kmh:.2f} km/h)", flush=True)

            # 해제 조건: 거리 ≥ release_dist_m AND 속도 ≥ release_speed_kmh
            if self._stop_latch and (d_ema >= self.release_dist_m) and (v_kmh >= self.release_speed_kmh):
                self._stop_latch = False
                if self.verbose:
                    print(f"[AEB* STOP] latch OFF (d={d_ema:.2f} m, v={v_kmh:.2f} km/h)", flush=True)

        # 3) Stop-Latch 동작
        if self._stop_latch:
            ebrake.apply_brake(1.0, dt)      # 마찰 제동 최대로
            self._force_stop_hold(model, data, dt)
            fused_info["forced_stop"] = True
            fused_info["forced_reason"] = "stop_latch(close&slow)"
            return fused_info

        # 4) 여기부터는 기존 ‘AEB 브레이킹 상태’일 때의 제동
        if fused_active and int(data.ncon) > 0:
            ebrake.apply_brake(brake_level, dt)
            for rad in self._radars:
                rad._apply_motor_brake(model, data)

        # 로그 유지
        if fused_active and not self._was_active:
            if self.verbose:
                print("[AEB*] 작동중 (engaged, fused)", flush=True)
        elif not fused_active and self._was_active:
            if self.verbose:
                print("[AEB*] 해제됨 (released, fused)", flush=True)

        self._was_active = fused_active
        return fused_info

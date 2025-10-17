# scripts/vehicle/aeb.py
# ------------------------------------------------------------
# 목적: AEB 판단 및 제동 적용
# 요약
#  - 라이다 스캔에서 '바닥' + '자차 차체'만 제외(타차는 이름 같아도 감지)
#  - dmin: 중앙 섹터의 P10(percentile 10) → EMA 평활
#  - 속도대역별 임계/범위(저·중·고속) + 전 구간에서 dmin_on/off 상향
#  - 히스테리시스(ON/OFF 분리) + 연속 프레임 조건으로 채터링 억제
#  - AEB 활성 시: EBrake(frictionloss) + 휠 모터 역토크 병행
#  - 액추에이터 경로 무효 시: 조인트 DOF(qfrc_applied) 직접 토크(폴백)
#  - per-site 임계/거리 오버라이드 지원(AEBRadarMulti에서 사용)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import mujoco
from perception.aeb_lidar import get_aeb_scan_with_tilt


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
        hold_time: float = 0.6,
        self_clearance: float = 0.20,
        # 히스테리시스/채터링 억제
        min_active_time: float = 0.35,
        cooldown_after_release: float = 0.20,
        consec_on_frames: int = 2,
        consec_off_frames: int = 6,
        # 모터 제어(오버라이드)
        motor_names: tuple[str, ...] = ("fl_motor", "fr_motor", "rl_motor", "rr_motor"),
        motor_brake_K: float = 600.0,
        clamp_ctrl: float = 10000.0,
        zero_drive_when_aeb: bool = True,
        static_brake_torque: float = 1800.0,
        static_brake_vmin: float = 0.2,
        # 로그
        verbose: bool = False,
        # 사이트별 임계/거리 오버라이드(선택)
        dmin_on_override: float | None = None,
        dmin_off_override: float | None = None,
        max_dist_override: float | None = None,
    ):
        # --- 센서/판단 파라미터 ---
        self.site = site_name
        self.tilt_deg = float(tilt_deg)
        self.alpha = float(ema_alpha)
        self.hold_time = float(hold_time)
        self.self_clearance = float(self_clearance)

        # --- 내부 상태 ---
        self._dmin_ema: float | None = None
        self._trigger_until: float = 0.0
        self._was_active: bool = False

        # 히스테리시스/채터링
        self._min_active_time = float(min_active_time)
        self._cooldown_after_release = float(cooldown_after_release)
        self._consec_on_need = int(max(1, consec_on_frames))
        self._consec_off_need = int(max(1, consec_off_frames))
        self._consec_on = 0
        self._consec_off = 0
        self._last_engage_t = -1e9
        self._last_release_t = -1e9

        # 모터 제어
        self._motor_names = tuple(motor_names)
        self._motor_brake_K = float(motor_brake_K)
        self._clamp_ctrl = float(clamp_ctrl)
        self._zero_drive_when_aeb = bool(zero_drive_when_aeb)
        self._static_brake_torque = float(static_brake_torque)
        self._static_brake_vmin = float(static_brake_vmin)
        self._act_ids: list[int] = []
        self._resolved = False

        # 로그
        self.verbose = bool(verbose)

        # 제외 이름(최소): 'floor' 무시, 'chassis_geom'은 자차 body일 때만 무시
        self.EXCLUDE_GEOMS = {"floor", "chassis_geom"}
        self._own_body: int | None = None  # 최초 1회 계산

        # per-site 오버라이드 저장
        self._dmin_on_override = dmin_on_override
        self._dmin_off_override = dmin_off_override
        self._max_dist_override = max_dist_override

    # ---------------------------- 유틸 ----------------------------
    @staticmethod
    def _kmh(v_mps: float) -> float:
        return 3.6 * v_mps

    def _resolve_actuators_once(self, model: mujoco.MjModel) -> None:
        """휠 모터 액추에이터 id를 1회만 해석."""
        if self._resolved:
            return
        self._act_ids.clear()

        # 1) 이름 매칭
        for name in self._motor_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._act_ids.append(int(aid))
        # 2) 폴백: '_motor' 포함
        if not self._act_ids:
            for a in range(model.nu):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
                if "_motor" in nm:
                    self._act_ids.append(int(a))
        # 3) 최종 폴백
        if not self._act_ids and model.nu > 0:
            self._act_ids = list(range(min(4, model.nu)))

        self._resolved = True
        if self.verbose:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self._act_ids]
            print(f"[AEB] resolved wheel actuators: {names}", flush=True)

    def _vehicle_forward_speed(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """
        라이다 site가 붙은 바디 기준 전방(x) 성분 속도 [m/s].
        """
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
        if sid < 0:
            return 0.0
        bid = int(model.site_bodyid[sid])
        R_bw = data.xmat[bid].reshape(3, 3)   # body←world
        R_wb = R_bw.T
        v_world = (np.array(data.cvel[bid][:3], float)
                   if hasattr(data, "cvel") else np.array(data.qvel[:3], float))
        v_local = R_wb @ v_world
        return float(v_local[0])

    def _adaptive_cfg(self, v_mps: float) -> dict:
        """
        속도대역별 FOV/거리/임계값(전 구간 dmin_on/off 상향) + per-site 오버라이드.
        """
        v = self._kmh(v_mps)
        cfg = dict(  # 기본 상향(전역)
            max_dist=80.0, fov_deg=90.0, center_deg=30.0, num_rays=61,
            ttc_on=2.0, ttc_off=3.0, dmin_on=10.0, dmin_off=12.0
        )
        if v < 5.0:  # 저속: 더 멀리서 작동
            cfg.update(max_dist=80.0, fov_deg=95.0, center_deg=34.0, num_rays=71,
                       ttc_on=3.0, ttc_off=4.0, dmin_on=12.0, dmin_off=14.0)
        elif v < 20.0:  # 중속
            cfg.update(max_dist=80.0, fov_deg=75.0, center_deg=28.0, num_rays=61,
                       ttc_on=2.2, ttc_off=3.0, dmin_on=11.0, dmin_off=13.0)
        else:  # 고속
            cfg.update(max_dist=140.0, fov_deg=60.0, center_deg=22.0, num_rays=71,
                       ttc_on=2.6, ttc_off=3.4, dmin_on=10.0, dmin_off=12.0)

        # per-site 오버라이드
        if self._max_dist_override is not None:
            cfg["max_dist"] = float(self._max_dist_override)
        if self._dmin_on_override is not None:
            cfg["dmin_on"] = float(self._dmin_on_override)
        if self._dmin_off_override is not None:
            cfg["dmin_off"] = float(self._dmin_off_override)
        return cfg

    def _is_excluded(self, model: mujoco.MjModel, gid: int) -> bool:
        """
        제외 규칙:
          - 'floor'는 항상 제외
          - 'chassis_geom'은 자차 body일 때만 제외(타차는 감지)
        """
        if gid < 0:
            return False
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name == "floor":
            return True
        if name == "chassis_geom":
            if self._own_body is None:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
                self._own_body = int(model.site_bodyid[sid]) if sid >= 0 else -1
            return int(model.geom_bodyid[gid]) == self._own_body
        return False

    # ---------------------------- 판단 ----------------------------
    def update(self, t: float, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, dict]:
        """
        현재 프레임의 AEB 활성 여부와 디버그 정보를 계산.
        Returns:
            (active: bool, info: dict)
        """
        v_fwd = max(0.0, self._vehicle_forward_speed(model, data))
        cfg = self._adaptive_cfg(v_fwd)

        # 스캔
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
        if sid < 0:
            raise KeyError(f"Invalid site '{self.site}'")

        thetas, dists, gids, center_idx, _origin = get_aeb_scan_with_tilt(
            model, data,
            site_name=self.site,
            num_rays=cfg["num_rays"],
            fov_deg=cfg["fov_deg"],
            max_dist=cfg["max_dist"],
            self_clearance=self.self_clearance,
            tilt_deg=self.tilt_deg,
        )

        # 바닥/자차 제외(자차 body만)
        keep = []
        for i, gid in enumerate(gids):
            if gid < 0:
                keep.append(i); continue
            if self._is_excluded(model, int(gid)):
                continue
            keep.append(i)

        if keep:
            thetas = thetas[keep]
            dists  = dists[keep]
            gids   = np.array(gids)[keep]
        else:
            thetas = np.array([0.0], float)
            dists  = np.array([cfg["max_dist"]], float)
            gids   = np.array([-1], int)

        # 중앙 섹터 dmin(P10)
        half_c = np.deg2rad(cfg["center_deg"] * 0.5)
        mask = (thetas >= -half_c) & (thetas <= half_c)
        sel = dists[mask] if np.any(mask) else dists
        dmin_raw = float(np.percentile(sel, 10)) if sel.size >= 5 else float(np.min(sel))

        # EMA
        self._dmin_ema = dmin_raw if self._dmin_ema is None else self.alpha * dmin_raw + (1.0 - self.alpha) * self._dmin_ema

        # TTC
        ttc = (self._dmin_ema / max(1e-2, v_fwd)) if v_fwd > 0.1 else float("inf")

        # 히스테리시스/연속 프레임
        want_on  = (self._dmin_ema <= cfg["dmin_on"]) or (ttc <= cfg["ttc_on"])
        want_off = (self._dmin_ema >= cfg["dmin_off"]) and (ttc >= cfg["ttc_off"])

        # 해제 직후 쿨다운
        if (t - self._last_release_t) < self._cooldown_after_release:
            want_on = False

        active_now = (t <= self._trigger_until)
        if not active_now:
            self._consec_on = self._consec_on + 1 if want_on else 0
            self._consec_off = 0
            if self._consec_on >= self._consec_on_need:
                self._last_engage_t = t
                sticky = max(self._min_active_time, 0.0) + max(self.hold_time, 0.0)
                self._trigger_until = t + sticky
                active_now = True
        else:
            long_enough = (t - self._last_engage_t) >= self._min_active_time
            if want_off and long_enough:
                self._consec_off += 1
            else:
                self._consec_off = 0
            if self._consec_off >= self._consec_off_need:
                self._last_release_t = t
                self._trigger_until = t
                active_now = False
                self._consec_on = 0
                self._consec_off = 0

        # 디버그(중앙 레이)
        if self.verbose and center_idx is not None and 0 <= center_idx < len(gids):
            gid_c = int(gids[center_idx])
            name_c = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid_c) if gid_c >= 0 else None
            dist_c = float(dists[min(center_idx, len(dists) - 1)])
            print(f"[AEB/Radar] center-hit name={name_c} dist={dist_c:.2f}m tilt={self.tilt_deg}deg", flush=True)

        info = {
            "dmin_raw": dmin_raw,
            "dmin_ema": float(self._dmin_ema),
            "v_forward_mps": v_fwd,
            "v_forward_kmh": self._kmh(v_fwd),
            "ttc": ttc,
            "trigger": active_now,
            "tilt_deg": self.tilt_deg,
            "scan_cfg": cfg,
            "consec_on": self._consec_on,
            "consec_off": self._consec_off,
        }
        return active_now, info

    # ---------------------------- 제동 주입 ----------------------------
    def _apply_motor_brake(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """
        우선 액추에이터(ctrl)에 역토크 주입,
        실패 시 조인트 DOF(qfrc_applied)로 폴백.
        """
        self._resolve_actuators_once(model)
        grounded = (int(data.ncon) > 0)

        # 전방속도(정적 제동 판단)
        v_fwd = 0.0
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site)
            if sid >= 0:
                bid = int(model.site_bodyid[sid])
                R_bw = data.xmat[bid].reshape(3, 3); R_wb = R_bw.T
                v_world = (np.array(data.cvel[bid][:3], float)
                           if hasattr(data, "cvel") else np.array(data.qvel[:3], float))
                v_fwd = float((R_wb @ v_world)[0])
        except Exception:
            pass

        # 1) 액추에이터 경로
        actuator_effective = False
        for aid in self._act_ids:
            if not (0 <= aid < model.nu):
                continue
            if self._zero_drive_when_aeb:
                data.ctrl[aid] = 0.0
            dof = int(model.actuator_trnid[aid][0])
            if not (0 <= dof < model.nv) or not grounded:
                continue
            w = float(data.qvel[dof])
            u = -self._motor_brake_K * w
            if abs(w) < 0.1 and abs(v_fwd) >= self._static_brake_vmin:
                u += -self._static_brake_torque * np.sign(v_fwd if v_fwd != 0 else 1.0)
            u = float(np.clip(u, -self._clamp_ctrl, self._clamp_ctrl))
            data.ctrl[aid] += u
            actuator_effective = True

        # 2) 폴백: 조인트 DOF에 직접 토크
        if not actuator_effective:
            for jn in ("fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    continue
                dof = int(model.jnt_dofadr[jid])
                if not (0 <= dof < model.nv) or not grounded:
                    continue
                w = float(data.qvel[dof])
                u = -self._motor_brake_K * w
                if abs(w) < 0.1 and abs(v_fwd) >= self._static_brake_vmin:
                    u += -self._static_brake_torque * np.sign(v_fwd if v_fwd != 0 else 1.0)
                u = float(np.clip(u, -self._clamp_ctrl, self._clamp_ctrl))
                data.qfrc_applied[dof] += u

    # ---------------------------- 공개 API ----------------------------
    def apply(
        self,
        ebrake,
        t: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dt: float,
        brake_level: float = 0.95,
    ) -> dict:
        active, info = self.update(t, model, data)

        if active and not self._was_active:
            print("[AEB] 작동중 (engaged)", flush=True)
        elif (not active) and self._was_active:
            print("[AEB] 해제됨 (released)", flush=True)
        self._was_active = active

        # 접지된 경우에만 실제 제동 적용
        if active and int(data.ncon) > 0:
            ebrake.apply_brake(brake_level, dt)
            self._apply_motor_brake(model, data)

        return info


# ============================== 멀티 레이더 ==============================
class AEBRadarMulti:
    """
    여러 site를 병렬로 스캔 → 가장 보수적인 판단(dmin_ema 최소 / TTC 최소)을 융합.
    per_site_cfg로 사이트별 dmin_on/off, max_dist 오버라이드 가능.
    """

    def __init__(
        self,
        *,
        site_names: tuple[str, ...] = ("lidar_high", "lidar_low"),
        tilt_deg: float = -1.0,
        ema_alpha: float = 0.30,
        hold_time: float = 0.6,
        self_clearance: float = 0.12,
        min_active_time: float = 0.35,
        cooldown_after_release: float = 0.20,
        consec_on_frames: int = 2,
        consec_off_frames: int = 6,
        motor_names: tuple[str, ...] = ("fl_motor", "fr_motor", "rl_motor", "rr_motor"),
        motor_brake_K: float = 1200.0,
        clamp_ctrl: float = 10000.0,
        zero_drive_when_aeb: bool = True,
        static_brake_torque: float = 3000.0,
        static_brake_vmin: float = 0.2,
        verbose: bool = True,
        per_site_cfg: dict[str, dict] | None = None,
    ):
        # 내부에서 기존 AEBRadar 재사용
        from vehicle.aeb import AEBRadar as _Single

        self._radars: list[_Single] = []
        self.verbose = bool(verbose)
        self._was_active = False

        per_site_cfg = per_site_cfg or {}
        for sn in site_names:
            cfg = per_site_cfg.get(sn, {})
            self._radars.append(_Single(
                site_name=sn,
                tilt_deg=tilt_deg,
                ema_alpha=ema_alpha,
                hold_time=hold_time,
                self_clearance=self_clearance,
                min_active_time=min_active_time,
                cooldown_after_release=cooldown_after_release,
                consec_on_frames=consec_on_frames,
                consec_off_frames=consec_off_frames,
                motor_names=tuple(motor_names),
                motor_brake_K=motor_brake_K,
                clamp_ctrl=clamp_ctrl,
                zero_drive_when_aeb=zero_drive_when_aeb,
                static_brake_torque=static_brake_torque,
                static_brake_vmin=static_brake_vmin,
                verbose=verbose,
                # per-site 오버라이드
                dmin_on_override = cfg.get("dmin_on_override"),
                dmin_off_override= cfg.get("dmin_off_override"),
                max_dist_override= cfg.get("max_dist_override"),
            ))

    @staticmethod
    def _fuse_active_info(infos: list[dict]) -> dict:
        """
        - dmin_ema: 최소값(가장 위험)
        - ttc: 최소값
        - trigger: OR
        """
        best = min(infos, key=lambda x: x["dmin_ema"])
        fused = dict(best)
        fused["ttc"] = min(i["ttc"] for i in infos)
        fused["trigger"] = any(i["trigger"] for i in infos)
        fused["scan_cfg"] = best.get("scan_cfg", {})
        return fused

    def update(self, t: float, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, dict]:
        actives_infos = [rad.update(t, model, data) for rad in self._radars]
        actives = [ai[0] for ai in actives_infos]
        infos   = [ai[1] for ai in actives_infos]
        fused_active = any(actives)
        fused_info = self._fuse_active_info(infos)

        if fused_active and not self._was_active:
            print("[AEB*] 작동중 (engaged, fused)", flush=True)
        elif (not fused_active) and self._was_active:
            print("[AEB*] 해제됨 (released, fused)", flush=True)
        self._was_active = fused_active

        return fused_active, fused_info

    def apply(self, ebrake, t, model, data, dt, brake_level: float = 0.95) -> dict:
        """
        각 레이더가 개별로 업데이트 및 제동 적용(활성된 것들만).
        friction + 역토크가 합쳐져 강하게 작동.
        """
        fused_active = False
        fused_info: dict | None = None
        for rad in self._radars:
            active, info = rad.update(t, model, data)
            if active and int(data.ncon) > 0:
                ebrake.apply_brake(brake_level, dt)
                rad._apply_motor_brake(model, data)
            fused_active = fused_active or active
            if fused_info is None or info["dmin_ema"] < fused_info["dmin_ema"]:
                fused_info = info
        return fused_info if fused_info is not None else {}

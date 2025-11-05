# scripts/vehicle/vehicleEnv.py
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import mujoco

# 🔧 상대 임포트 (방법 A: python -m scripts.main 로 실행)
from ..perception import LidarSensor
from .control import compose_control
from .ebrake import EBrake
from .aeb import AEBRadarMulti
from .torque_sync_tcs import TorqueSyncTCS


# ---------- 유틸 ----------
def _read_text(p: Path) -> str:
    with p.open("r", encoding="utf-8") as f:
        return f.read()

def _remove_xml_header_and_mujoco(xml: str) -> str:
    """ XML 선언/DOCTYPE 제거 + 모든 <mujoco> 여닫기 제거 """
    xml = re.sub(r'<\?xml[^>]*\?>', '', xml, flags=re.I)
    xml = re.sub(r'<!DOCTYPE[^>]*>', '', xml, flags=re.I | re.S)
    xml = re.sub(r'<\s*/\s*mujoco\s*>', '', xml, flags=re.I)         # </mujoco>
    xml = re.sub(r'<\s*mujoco\b[^>]*>', '', xml, flags=re.I)          # <mujoco ...>
    return xml.strip()

def _take_block(xml: str, tag: str) -> str | None:
    """ <tag> ... </tag> 를 찾아 '안쪽 내용'만 반환 (없으면 None) """
    m = re.search(rf'<{tag}\b[^>]*>(.*?)</{tag}>', xml, flags=re.I | re.S)
    return m.group(1).strip() if m else None

def _take_element(xml: str, tag: str) -> str | None:
    """ <tag>...</tag> '전체 요소' 통째로 반환 (없으면 None) """
    m = re.search(rf'<{tag}\b[^>]*>.*?</{tag}>', xml, flags=re.I | re.S)
    return m.group(0).strip() if m else None

def _strip_root_level_tags(xml: str, tags: tuple[str, ...]) -> str:
    """ 지정 태그들을 전체에서 모두 제거 (여는/닫는 포함) """
    for tg in tags:
        xml = re.sub(rf'<{tg}\b[^>]*>.*?</{tg}>', '', xml, flags=re.I | re.S)  # 통째 요소 제거
        xml = re.sub(rf'<{tg}\b[^>]*>', '', xml, flags=re.I)                   # 여는 태그
        xml = re.sub(rf'</{tg}\s*>', '', xml, flags=re.I)                      # 닫는 태그
    return xml


class VehicleEnv:
    """
    단일 차량 + AEB 시뮬레이션 환경 래퍼.

    - scene/base_scene.xml 에 vehicle_active / vehicle_static / actuator 를
      문자열 치환으로 인라인하여 하나의 XML로 구성한 뒤 MjModel 생성
    - EBrake(마찰손실) + AEB(AEBRadarMulti)의 제동을 병행
    - step(action) 에서 baseline control → 페달 브레이크 → AEB → (TCS 동기화) 순서로 적용
    """

    def __init__(self, **kwargs):
        # ---------- 경로 (파일 위치 기준, CWD 무관) ----------
        proj_root = Path(__file__).resolve().parents[2]
        self.vehicle_active_path = Path(kwargs.get(
            "vehicle_active_path",
            proj_root / "models" / "vehicle" / "vehicle_active.xml"
        ))
        self.vehicle_static_path = Path(kwargs.get(
            "vehicle_static_path",
            proj_root / "models" / "vehicle" / "vehicle_static.xml"
        ))
        self.actuator_path = Path(kwargs.get(
            "actuator_path",
            proj_root / "models" / "vehicle" / "actuator.xml"
        ))
        self.scene_path = Path(kwargs.get(
            "scene_path",
            proj_root / "models" / "scene" / "base_scene.xml"
        ))

        # ---------- 모델 로딩/초기화 ----------
        xml = self._compose_model()                          # 문자열로 병합
        self.model = mujoco.MjModel.from_xml_string(xml)     # include/상대경로 문제 회피
        self.data  = mujoco.MjData(self.model)

        # 좌/우 토크 동기화 + 간단 TCS
        self.tcs = TorqueSyncTCS(self.model)

        # ---------- E-Brake(마찰손실) ----------
        self.ebrake = EBrake(
            model=self.model,
            data=self.data,
            frictionloss_max=2500.0,         # 제동 효과 강도
            tau_actuator=0.05,               # 응답 지연(작을수록 빠름)
            wheel_joint_names=["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"],
        )

        # ---------- 라이다 래퍼 (모델엔 'lidar'가 없고 ['lidar_high','lidar_low']만 있음) ----------
        try:
            self.lidar = LidarSensor(self.model, self.data, site_name="lidar_low")
        except Exception:
            self.lidar = LidarSensor(self.model, self.data, site_name="lidar_high")

        # ---------- AEB(상·하 듀얼 라이다) ----------
        self.aeb = AEBRadarMulti(
            site_names=("lidar_high", "lidar_low"),
            tilt_deg=0.0,
            ema_alpha=0.30,
            self_clearance=0.12,
            motor_brake_K=2000.0,
            clamp_ctrl=10000.0,
            zero_drive_when_aeb=True,
            static_brake_torque=5000.0,
            static_brake_vmin=0.05,
            verbose=False,
            per_site_cfg={
                "lidar_low": {                 # ▼ 저/낮은 물체 대응 강화
                    "dmin_on_override": 12.0,  # 켜짐 임계 거리
                    "dmin_off_override": 14.0, # 꺼짐 임계 거리(히스테리시스)
                    "max_dist_override": 90.0, # 최대 레이 길이
                }
            },
        )

        # (디버그) 휠 액추에이터 목록
        self._wheel_act_ids = []
        for name in ("fl_motor", "fr_motor", "rl_motor", "rr_motor"):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._wheel_act_ids.append(aid)

        print(
            "[VehicleEnv] model ready:",
            {"active": str(self.vehicle_active_path),
             "static": str(self.vehicle_static_path),
             "actuator": str(self.actuator_path),
             "scene": str(self.scene_path)},
            "\n[AEB] sites=('lidar_high','lidar_low'), wheel actuators:",
            [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self._wheel_act_ids],
            flush=True,
        )

        self.done = False

    # ---------- 모델 합성 (원래 구조 유지) ----------
    def _compose_model(self) -> str:
        """
        base_scene.xml 내부 플레이스홀더를 실제 vehicle/actuator XML로 치환하여
        단일 XML 문자열을 반환.

        규칙:
        - VEHICLE_ACTIVE / VEHICLE_STATIC 자리에는 '오직 <body>…</body>들'만 삽입
          (즉, 서브 XML의 <worldbody> 안쪽 내용만 추출하여 넣음)
        - ACTUATOR 자리에는 '<actuator>…</actuator>' 요소 통째로 삽입
        - 서브 XML에 남아 있는 <compiler>/<option>/<asset>/<sensor>/<contact> 등
          '루트 수준 태그'는 모두 제거
        - 플레이스홀더가 여러번 있어도 1회만 치환하고 나머지는 삭제
        """
        # 경로 로드
        scene_xml    = _read_text(self.scene_path)
        active_xml   = _read_text(self.vehicle_active_path)
        static_xml   = _read_text(self.vehicle_static_path)
        actuator_xml = _read_text(self.actuator_path)

        # 서브 XML: 헤더/루트 제거
        active_xml   = _remove_xml_header_and_mujoco(active_xml)
        static_xml   = _remove_xml_header_and_mujoco(static_xml)
        actuator_xml = _remove_xml_header_and_mujoco(actuator_xml)

        # 1) ACTIVE/STATIC → worldbody 내부 내용만 꺼냄(= <body>… 들)
        active_world_inner = _take_block(active_xml, "worldbody")
        static_world_inner = _take_block(static_xml, "worldbody")

        # 혹시 worldbody가 없으면 전체에서 body들만 긁어 모음(폴백)
        if not active_world_inner:
            bodies = re.findall(r'<body\b[^>]*>.*?</body>', active_xml, flags=re.I | re.S)
            active_world_inner = "\n".join(bodies)
        if not static_world_inner:
            bodies = re.findall(r'<body\b[^>]*>.*?</body>', static_xml, flags=re.I | re.S)
            static_world_inner = "\n".join(bodies)

        # ACTIVE/STATIC에서 루트 수준 태그(compiler/option/asset/sensor/contact/actuator)는 제거
        drop_tags = ("compiler", "option", "asset", "sensor", "contact", "tendon", "keyframe", "actuator")
        active_xml_clean = _strip_root_level_tags(active_world_inner or "", drop_tags)
        static_xml_clean = _strip_root_level_tags(static_world_inner or "", drop_tags)

        # 2) ACTUATOR → <actuator> 요소 통째로 추출
        actuator_elem = _take_element(actuator_xml, "actuator")
        if not actuator_elem:
            actuator_elem = ""  # 안전 폴백

        merged = scene_xml

        # 플레이스홀더 패턴 (공백/대소문자 유연 처리)
        pat_active   = re.compile(r'<!--\s*VEHICLE[_\s]+ACTIVE\s+INCLUDE\s*-->', re.I)
        pat_static   = re.compile(r'<!--\s*VEHICLE[_\s]+STATIC\s+INCLUDE\s*-->', re.I)
        pat_actuator = re.compile(r'<!--\s*ACTUATOR\s+INCLUDE\s*-->',            re.I)

        # 3) 각 플레이스홀더는 첫 1회만 치환
        merged, _ = pat_active.subn(active_xml_clean, merged, count=1)
        merged, _ = pat_static.subn(static_xml_clean, merged, count=1)
        merged, _ = pat_actuator.subn(actuator_elem,  merged, count=1)

        # 4) 남은 플레이스홀더는 전부 제거
        merged = pat_active.sub('', merged)
        merged = pat_static.sub('', merged)
        merged = pat_actuator.sub('', merged)

        return merged

    # ---------- 리셋 ----------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.done = False
        return self._get_obs()

    # ---------- 스텝 ----------
    def step(self, action: dict):
        """
        Args
        ----
        action: dict
            {"throttle": float, "reverse": float, "steer": float, "brake": float}
        """
        dt = float(self.model.opt.timestep)

        # 1) 기본 control 구성(엔진/조향/서스펜션)
        suspension = [0.0, 0.0, 0.0, 0.0]
        ctrl = compose_control(action, suspension)

        # 2) baseline ctrl 먼저 적용 (이후 AEB가 덮어씀)
        self.data.ctrl[:] = ctrl

        # 3) 운전자 브레이크(마찰손실 기반)
        self.ebrake.apply_brake(action.get("brake", 0.0), dt)

        # 4) AEB (활성 시 frictionloss + 휠 역토크 병행)
        info_aeb = self.aeb.apply(
            self.ebrake, t=self.data.time, model=self.model, data=self.data, dt=dt, brake_level=0.95
        )

        # 5) 좌/우 토크 동기화 + 간단 TCS (최종 ctrl에 적용)
        self.tcs.apply(self.data, self.data.ctrl)

        # 6) 물리 스텝
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done, info = 0.0, self.done, {"aeb": info_aeb}
        return obs, reward, done, info

    # ---------- 관측값 ----------
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

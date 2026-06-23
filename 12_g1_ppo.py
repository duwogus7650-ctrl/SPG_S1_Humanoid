"""
12. Unitree G1 실제 로봇 — PPO 강화학습 (고품질 렌더)
=====================================================================

11번(MuJoCo 표준 휴머노이드)을, 실제 **Unitree G1 휴머노이드 로봇 모델**
(MuJoCo Menagerie, 관절36/액추에이터29, 실제 메시)로 바꾸고,
**MuJoCo 고품질 렌더러**(메시·조명·그림자·반사)로 진짜 로봇처럼 보여준다.
학습은 11번과 동일하게 실제 로봇사 방식 **PPO**(Stable-Baselines3) + 병렬환경.

  · 모델 = Unitree G1 (robot_descriptions로 자동 다운로드)
  · 물리 = MuJoCo / 학습 = SB3 PPO + VecNormalize + 위치제어
  · 학습방식 = DeepMimic(모션 모방) — 파라메트릭 걷기 레퍼런스를 따라가도록 보상
  · 렌더 = mujoco.Renderer 오프스크린 → pygame 합성 + UI 오버레이
  · 구조 = [학습 스레드] PPO 연속 학습 / [메인] 정책 스냅샷 numpy추론·렌더

조작: ↑↓ 학습집중(렌더 fps↓) · Q/E 카메라 회전 · R 리셋 · ESC
실행: pip install mujoco gymnasium stable_baselines3 torch robot_descriptions imageio pygame-ce
      → python 12_g1_ppo.py

기대치: 순수 보상설계는 '질질 끌기/발작 떨림'에 빠지기 쉬워, 걷기 레퍼런스
모션을 모방(DeepMimic)하도록 바꿨다. 정책은 레퍼런스에 대한 잔차(보정)만
학습하므로 처음부터 주기적인 발 떼기가 나오고, 점차 균형을 잡아 또렷하고
매끄럽게 걷는다. GPU가 있으면 렌더가 부드럽다(소프트웨어 GL이면 ↑로 fps↓).
"""

import os
import math
import time
import threading
import warnings
warnings.filterwarnings("ignore")
# 단, NaN/overflow 등 수치이상은 보이게 둔다(조용한 발산 방지).
warnings.filterwarnings("default", category=RuntimeWarning)
import numpy as np
import pygame
import mujoco
import gymnasium as gym
from gymnasium import spaces
from robot_descriptions import g1_mj_description
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import torch.nn as nn

# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1240, 780
ARENA_H = 510
RENDER_W, RENDER_H = 900, 372         # 오프스크린 렌더 해상도(아레나로 확대)

N_ENVS = 6
NET_ARCH = [256, 256]
FALL_Z = 0.55                         # 골반 높이 이 아래면 종료
ACTION_SCALE = float(os.environ.get("ACTION_SCALE", "0.25"))  # Unitree legged_gym G1 값
FALL_TILT = 0.5                       # 투영중력 z 이 위(>-0.5)면 과도하게 기울어짐 → 종료
FOOT_CONTACT_H = 0.07                 # 발 링크 z 이 아래면 접지로 간주
STAND_FOOT_Z = 0.033                  # 똑바로 섰을 때 발 링크(ankle_roll) z(m)

SCENE = os.path.join(os.path.dirname(g1_mj_description.MJCF_PATH), "scene.xml")
_M0 = mujoco.MjModel.from_xml_path(SCENE)
NU = _M0.nu
FRAME_SKIP = max(1, round(0.02 / _M0.opt.timestep))   # ~50Hz 제어
# --- Unitree 방식(legged_gym G1): 레퍼런스 모션 없이 속도추종+보상셰이핑 ---
ACT_DIM = 12                          # 다리 12관절만 제어(팔/허리 고정) — Unitree
OBS_DIM = 47                          # angvel3+grav3+cmd3+dofpos12+dofvel12+lastact12+clock2
LEG_QPOS = list(range(7, 19))         # 다리 12관절 qpos 인덱스
LEG_QVEL = list(range(6, 18))         # 다리 12관절 dof(qvel) 인덱스
LEG_ACT = list(range(0, 12))          # 다리 12 액추에이터 인덱스(ctrl)
# Unitree 기본 관절각(action=0 기준) [hip_p,hip_r,hip_y,knee,ank_p,ank_r]×2
DEFAULT_LEG = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], np.float64)
HIP_YR_Q = [8, 9, 14, 15]             # hip_roll/yaw qpos 인덱스(다리벌어짐 방지)
GAIT_CLOCK_HZ = 1.0 / 0.8            # 보행시계 주기 0.8s
OBS_CMD_SCALE = np.array([2.0, 2.0, 0.25])
# Unitree legged_gym G1 보상 스케일
R_TRACK_LIN, R_TRACK_ANG = 1.0, 0.5
R_LIN_Z, R_ANG_XY, R_ORI, R_HEIGHT = -2.0, -0.05, -1.0, -10.0
R_DOF_ACC, R_DOF_VEL, R_ARATE, R_DOF_LIM = -2.5e-7, -1e-3, -0.01, -5.0
R_ALIVE, R_HIP, R_CONTACT_NOVEL, R_SWING_H, R_CONTACT = 0.15, -1.0, -0.2, -20.0, 0.18
BASE_HEIGHT_TARGET, FEET_SWING_TARGET, TRACK_SIGMA = 0.78, 0.08, 0.25

# 기본자세(키프레임0) — 레퍼런스 모션의 베이스
_d0 = mujoco.MjData(_M0)
mujoco.mj_resetDataKeyframe(_M0, _d0, 0)
HOME_QPOS = _d0.qpos.copy()
HOME_J = HOME_QPOS[7:].copy()          # 29개 관절 기본각

# ---- 보행 레퍼런스: 발 궤적 계획 + 수치 IK(협응된 자연 보행) -----------------
# 기존 '독립 사인파' 레퍼런스는 다리 협응이 없어 벌어지고 꼬이는 비자연 보행으로
# 수렴했다(§18~19). 대신 계획된 발 궤적(스탠스=접지로 뒤로, 스윙=낮은 호로 들어 앞으로)을
# 실제 G1 모델의 수치 IK로 풀어 '협응된 자연 보행' 관절궤적을 만든다(09 검증 방식의 G1 이식).
# 기구학 검증: 좌우 발이 번갈아 ~0.05m 들리는 정상 교대보행 + 직립자세(렌더 확인).
# qpos[7:] 인덱스: L hip_p=0 hip_r=1 knee=3 ank_p=4 / R hip_p=6 hip_r=7 knee=9 ank_p=10
#                  L shoulder_p=15 / R shoulder_p=22
GAIT_PERIOD = float(os.environ.get("GAIT_PERIOD", "1.4"))   # 한 사이클(두 걸음) 시간(s)
_REF = dict(PELVIS_H=0.78, FOOT_Z0=0.05, STANCE_Y=0.13, STEP_LEN=0.26,
            STEP_H=0.055, DUTY=0.62, SWAY=0.025, N_PHASE=60, A_SHO=0.22)
_LEG_DOF = {"left": dict(hip_p=6, hip_r=7, knee=9, ank_p=10),
            "right": dict(hip_p=12, hip_r=13, knee=15, ank_p=16)}


def _foot_target(phase, leg):
    """위상에서 발 목표 위치(골반 (0,0,PELVIS_H) 직립고정 월드좌표). x=전방 y=좌 z=상."""
    r = _REF
    p = phase if leg == "left" else (phase + 0.5) % 1.0
    sign = 1.0 if leg == "left" else -1.0
    if p < r["DUTY"]:                                  # 스탠스: 접지로 뒤로
        s = p / r["DUTY"]; x = r["STEP_LEN"] * (0.5 - s); z = r["FOOT_Z0"]
    else:                                              # 스윙: 들어 앞으로
        u = (p - r["DUTY"]) / (1.0 - r["DUTY"])
        x = r["STEP_LEN"] * (u - 0.5); z = r["FOOT_Z0"] + r["STEP_H"] * math.sin(math.pi * u)
    y = sign * r["STANCE_Y"] - r["SWAY"] * math.sin(2.0 * math.pi * phase) * sign
    return np.array([x, y, z])


def _build_reference_table():
    """발 궤적+감쇠최소자승 IK로 N_PHASE x 29 관절각 테이블 생성(import 시 1회, ~3s).
    무릎은 굴곡(+)으로만 굽도록 강제 + 전 관절 한계 클램프 → IK가 과신전(뒤로 꺾임)
    가지로 빠지는 것 방지(G1 무릎 한계 [-0.087, 2.88], 과신전 불가)."""
    m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
    feet = {lg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, lg + "_ankle_roll_link")
            for lg in ("left", "right")}

    def jrange(name):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])
    lim = {}  # (leg, 관절) → (lo, hi) 클램프 한계
    for leg in ("left", "right"):
        lim[(leg, "hip_p")] = jrange(leg + "_hip_pitch_joint")
        lim[(leg, "hip_r")] = jrange(leg + "_hip_roll_joint")
        klo, khi = jrange(leg + "_knee_joint")
        lim[(leg, "knee")] = (max(0.05, klo), khi)        # 무릎 굴곡(+) 강제

    N = _REF["N_PHASE"]; table = np.zeros((N, 29)); jacp = np.zeros((3, m.nv))
    for i in range(N):
        ph = i / N
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[0:3] = [0.0, 0.0, _REF["PELVIS_H"]]; d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for leg in ("left", "right"):
            lp = _LEG_DOF[leg]; names = ["hip_p", "hip_r", "knee"]; dofs = [lp[n] for n in names]
            d.qpos[lp["knee"] + 1] = 0.3                  # 굴곡 가지에서 시작(과신전 방지)
            tgt = _foot_target(ph, leg)
            for _ in range(120):
                mujoco.mj_forward(m, d)
                err = tgt - d.xpos[feet[leg]]
                if np.linalg.norm(err) < 1e-4:
                    break
                mujoco.mj_jacBody(m, d, jacp, None, feet[leg])
                J = jacp[:, dofs]
                dq = J.T @ np.linalg.solve(J @ J.T + 2e-3 * np.eye(3), err)
                for k, nm in enumerate(names):
                    qa = dofs[k] + 1; lo, hi = lim[(leg, nm)]
                    d.qpos[qa] = min(hi, max(lo, d.qpos[qa] + dq[k]))   # 관절 한계 클램프
            d.qpos[lp["ank_p"] + 1] = -(d.qpos[lp["hip_p"] + 1] + d.qpos[lp["knee"] + 1])  # 발 평평
        mujoco.mj_forward(m, d)
        q = d.qpos[7:].copy()
        thL = 2.0 * math.pi * ph                       # 팔: 같은쪽 다리와 반대 스윙
        q[15] = HOME_J[15] + _REF["A_SHO"] * math.cos(thL)
        q[22] = HOME_J[22] + _REF["A_SHO"] * math.cos(thL + math.pi)
        table[i] = q
    return table


# IK 레퍼런스는 이제 학습에 안 쓰임(Unitree 방식). run.py 레퍼런스 뷰어에서만 필요하므로
# 첫 호출 시에만 빌드(지연 로딩) → 학습/평가 import 비용 0.
_REF_TABLE = None


def ref_pose(phase):
    """위상 phase∈[0,1)의 29개 관절 목표각(IK 보행 레퍼런스, 선형보간). run.py 뷰어용."""
    global _REF_TABLE
    if _REF_TABLE is None:
        _REF_TABLE = _build_reference_table()
    N = len(_REF_TABLE)
    x = (phase % 1.0) * N
    i0 = int(x) % N; i1 = (i0 + 1) % N; f = x - math.floor(x)
    return (1.0 - f) * _REF_TABLE[i0] + f * _REF_TABLE[i1]


# 명령 속도 범위(reset 시 샘플) — 처음엔 전진 위주로 학습
CMD_LIN_X = (float(os.environ.get("CMD_VX_LO", "0.3")), float(os.environ.get("CMD_VX_HI", "0.8")))

# 색 (UI) — SPG S1 Humanoid 브랜드: 딥네이비 + 시그니처 앰버
C_BG = (8, 18, 34)                                  # 딥네이비 배경
C_PANEL = (11, 23, 42); C_PANEL_LINE = (40, 64, 100)
C_TEXT = (238, 242, 248); C_DIM = (150, 168, 195); C_NEON = (255, 176, 0)  # 시그니처 앰버
C_ACCENT = (255, 176, 0)
C_CURVE_BEST = (255, 176, 0); C_CURVE_AVG = (120, 165, 230)


# ---------------------------------------------------------------------------
# SPG S1 외피(시각 전용·비충돌·무질량) — (바디, 타입, size, pos, 재질).
#  물리 불변: contype/conaffinity=0(충돌X), 바디는 명시 inertial 보유(질량 불변).
# 몸통은 원래 G1 형상 그대로. 머리만 SPG 헬멧+바이저로 변경.
#  좌표는 torso_link 로컬 프레임(실측): head_link 메시 z=0.281~0.487(중심 0.384),
#  폭 ±0.078, 앞면 x=+0.078 — 헬멧 돔은 폭을 줄여 슬림하게 밀착, 바이저는 눈높이 가는 슬릿.
_SKIN = [
    ("torso_link", "ellipsoid", (0.076, 0.074, 0.104), (0.016, 0.0, 0.382), "spg_shell"),  # 헬멧 돔(슬림·머리 밀착)
    ("torso_link", "ellipsoid", (0.038, 0.054, 0.010), (0.062, 0.0, 0.392), "spg_amber"),  # 가는 앰버 바이저 슬릿
    ("torso_link", "box",       (0.010, 0.0075, 0.030), (0.082, 0.0, 0.250), "spg_core"),  # SPG 가슴 코어 마크(글로잉 앰버)
]
_GEOM_T = {"box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


def build_model(plate=True):
    """SPG S1 외피(다크메탈 셸 + 앰버 액센트)를 비충돌 장식으로 입혀 컴파일.
    plate=False면 순정 G1(외피 없음). 외피는 시각 전용이라 물리/학습에 영향 없음."""
    if not plate:
        return mujoco.MjModel.from_xml_path(SCENE)
    spec = mujoco.MjSpec.from_file(SCENE)

    def addmat(name, rgba, emission, spec_v, shin, refl=0.3):
        mt = spec.add_material(); mt.name = name
        mt.rgba = rgba; mt.emission = emission
        mt.specular = spec_v; mt.shininess = shin; mt.reflectance = refl
    addmat("spg_shell", [0.055, 0.075, 0.135, 1.0], 0.0, 0.15, 0.30, 0.05)  # 매트 다크네이비(응집)
    addmat("spg_amber", [1.0, 0.69, 0.0, 1.0], 0.35, 0.5, 0.4, 0.3)     # 시그니처 앰버(#FFB000, 가는 슬릿)
    addmat("spg_core",  [1.0, 0.78, 0.25, 1.0], 0.95, 0.6, 0.4, 0.3)    # 발광 코어
    # 몸통 recolor(다크네이비)·UNITREE 로고 제거는 restyle()에서 모델 레벨로 일괄 처리한다
    # (모든 뷰어가 build_model 후 restyle를 호출 → 단일 소스). build_model은 외피 geom만 추가.
    added = 0
    for i, (body, typ, size, pos, mat) in enumerate(_SKIN):
        try:
            b = spec.body(body)
        except Exception:
            b = None
        if b is None:                     # mujoco 3.9: 없는 바디명은 None 반환(예외 아님)
            continue
        g = b.add_geom()
        g.name = "spgskin_%d" % i
        g.type = _GEOM_T[typ]; g.size = list(size); g.pos = list(pos)
        g.material = mat; g.contype = 0; g.conaffinity = 0; g.group = 2
        g.density = 0.0                    # 무질량 보장(inertiafromgeom=true여도 물리 불변)
        added += 1
    if added != len(_SKIN):               # 바디명이 바뀌면 외피가 조용히 사라지므로 시끄럽게 경고
        print("[SPG][warn] 외피 geom %d/%d만 부착 — 바디명 변경?" % (added, len(_SKIN)), flush=True)
    return spec.compile()


class G1Env(gym.Env):
    """Unitree legged_gym 방식: 레퍼런스 모션 없이 속도명령 추종 + 보상셰이핑으로 보행 학습.
    · 행동 = 다리 12관절 목표각(target = action*0.25 + 기본각). 팔/허리는 기본자세 고정.
    · 관찰 = base각속도·투영중력·속도명령·다리각/속도·직전행동·보행시계 (47).
    · 보상 = 속도추종 + 직립/높이 + 발스윙높이/접지패턴/미끄럼방지/다리벌어짐방지 + 부드러움.
    · 종료 = 골반 낮음 or 과도한 기울기. (레퍼런스/RSI 없음 → 레퍼런스 계열 버그 원천 제거)
    """
    def __init__(self, plate=True):     # SPG S1 외피 기본 적용(시각 전용·물리 불변)
        self.m = build_model(plate)
        self.d = mujoco.MjData(self.m)
        self.lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.hi = self.m.actuator_ctrlrange[:, 1].copy()
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.home = self.d.qpos.copy()
        self.home_j = self.home[7:].copy()
        self.base_ctrl = self.home_j.astype(np.float64).copy()   # 팔/허리 기본자세 유지
        self.base_ctrl[:12] = DEFAULT_LEG                        # 다리 = Unitree 기본각
        self.leg_lo = self.lo[LEG_ACT].copy(); self.leg_hi = self.hi[LEG_ACT].copy()
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.dt = self.m.opt.timestep * FRAME_SKIP
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.phase = 0.0
        self.cmd = np.array([0.5, 0.0, 0.0])             # 속도명령 [vx, vy, wz]
        self.prev_a = np.zeros(ACT_DIM, np.float32)
        self.prev_dofvel = np.zeros(ACT_DIM)
        self.prev_foot_xy = np.zeros((2, 2))
        self.t = 0

    def _obs(self):
        R = self.d.xmat[1].reshape(3, 3)
        ang_vel = self.d.qvel[3:6] * 0.25
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        dofpos = self.d.qpos[LEG_QPOS] - DEFAULT_LEG
        dofvel = self.d.qvel[LEG_QVEL] * 0.05
        ph = 2.0 * math.pi * self.phase
        clock = np.array([math.sin(ph), math.cos(ph)])
        return np.concatenate([ang_vel, grav, self.cmd * OBS_CMD_SCALE,
                               dofpos, dofvel, self.prev_a, clock])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[7:] = self.home_j
        self.d.qpos[LEG_QPOS] = DEFAULT_LEG + self.np_random.uniform(-0.03, 0.03, 12)
        self.d.qpos[:2] = 0.0
        self.d.qpos[2] = 0.80
        self.d.qpos[3:7] = self.home[3:7]
        mujoco.mj_forward(self.m, self.d)
        foot_z = self.d.xpos[self.feet][:, 2].min()
        self.d.qpos[2] += (STAND_FOOT_Z - foot_z)          # 발바닥이 지면에 닿게
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.cmd = np.array([self.np_random.uniform(*CMD_LIN_X), 0.0, 0.0])
        self.phase = float(self.np_random.uniform(0.0, 1.0))
        self.prev_a[:] = 0.0
        self.prev_dofvel = self.d.qvel[LEG_QVEL].copy()
        self.prev_foot_xy = self.d.xpos[self.feet][:, :2].copy()
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        a = np.clip(a, -1, 1).astype(np.float32)
        ctrl = self.base_ctrl.copy()
        ctrl[LEG_ACT] = np.clip(a * ACTION_SCALE + DEFAULT_LEG, self.leg_lo, self.leg_hi)
        self.d.ctrl[:] = np.clip(ctrl, self.lo, self.hi)
        prev_dofvel = self.d.qvel[LEG_QVEL].copy()
        prev_foot_xy = self.d.xpos[self.feet][:, :2].copy()
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.m, self.d)
        self.phase = (self.phase + self.dt * GAIT_CLOCK_HZ) % 1.0

        R = self.d.xmat[1].reshape(3, 3)
        v_body = R.T @ self.d.qvel[0:3]                  # base 선속도(몸통좌표)
        w_body = self.d.qvel[3:6]                        # base 각속도(몸통좌표)
        grav = R.T @ np.array([0.0, 0.0, -1.0])          # 투영중력(직립=[0,0,-1])
        z = self.d.qpos[2]

        r = R_TRACK_LIN * math.exp(-((self.cmd[0]-v_body[0])**2 + (self.cmd[1]-v_body[1])**2) / TRACK_SIGMA)
        r += R_TRACK_ANG * math.exp(-((self.cmd[2]-w_body[2])**2) / TRACK_SIGMA)
        r += R_LIN_Z * float(v_body[2])**2
        r += R_ANG_XY * float(w_body[0]**2 + w_body[1]**2)
        r += R_ORI * float(grav[0]**2 + grav[1]**2)
        r += R_HEIGHT * (z - BASE_HEIGHT_TARGET)**2
        dofvel = self.d.qvel[LEG_QVEL]
        r += R_DOF_VEL * float(np.sum(dofvel**2))
        r += R_DOF_ACC * float(np.sum(((dofvel - prev_dofvel) / self.dt)**2))
        r += R_ARATE * float(np.sum((a - self.prev_a)**2))
        r += R_HIP * float(np.sum(self.d.qpos[HIP_YR_Q]**2))   # 다리벌어짐 방지
        r += R_ALIVE
        # 보행시계로 기대 접지 판정 → 접지패턴/발스윙높이/미끄럼방지
        foot_z = self.d.xpos[self.feet][:, 2]
        contact = foot_z < FOOT_CONTACT_H
        exp_stance = [self.phase < 0.5, self.phase >= 0.5]
        for i in range(2):
            if contact[i] == exp_stance[i]:
                r += R_CONTACT                            # 접지패턴 일치 보상
            if not exp_stance[i]:                          # 스윙: 발 들기 목표
                r += R_SWING_H * (foot_z[i] - FEET_SWING_TARGET)**2
            if contact[i]:                                # 접지 중 수평이동(미끄럼) 페널티
                fv = (self.d.xpos[self.feet[i]][:2] - prev_foot_xy[i]) / self.dt
                r += R_CONTACT_NOVEL * float(np.sum(fv**2))
        self.prev_a = a.copy()
        self.t += 1
        terminated = bool(z < 0.5 or grav[2] > -FALL_TILT)
        truncated = self.t >= 1500
        return self._obs(), float(r), terminated, truncated, {}


def make_env():
    return Monitor(G1Env())


# ---------------------------------------------------------------------------
SHARED = {"W0": None, "b0": None, "W2": None, "b2": None, "Wa": None, "ba": None,
          "mean": None, "var": None, "steps": 0, "rew": [], "ep": float("nan"),
          "stop": False, "ready": False, "error": None}
LOCK = threading.Lock()


class Snapshot(BaseCallback):
    def _on_step(self):
        return not SHARED["stop"]

    def _on_rollout_end(self):
        p = self.model.policy
        pn = p.mlp_extractor.policy_net
        # 추론 경로(policy_action)는 pn[0],pn[2]만 풀어 '2-은닉층 tanh MLP'을 가정한다.
        # SB3 버전/구조가 바뀌어 레이어 배치가 달라지면 '크래시 없이 틀린 추론'(녹화가
        # 실제 학습 정책과 불일치)이 되므로, 가정이 깨지면 여기서 즉시 멈춘다.
        assert NET_ARCH == [256, 256], \
            f"policy_action은 net_arch [256,256] 가정인데 {NET_ARCH}"
        assert len(pn) >= 3 and isinstance(pn[0], nn.Linear) \
            and isinstance(pn[2], nn.Linear), \
            f"정책망 구조가 추론 경로(pn[0],pn[2]) 가정과 다름: {pn}"
        env = self.training_env
        while not hasattr(env, "obs_rms") and hasattr(env, "venv"):
            env = env.venv
        rms = env.obs_rms
        ep = float("nan")
        if len(self.model.ep_info_buffer) > 0:
            ep = safe_mean([e["r"] for e in self.model.ep_info_buffer])
        with LOCK:
            SHARED["W0"] = pn[0].weight.detach().cpu().numpy().copy()
            SHARED["b0"] = pn[0].bias.detach().cpu().numpy().copy()
            SHARED["W2"] = pn[2].weight.detach().cpu().numpy().copy()
            SHARED["b2"] = pn[2].bias.detach().cpu().numpy().copy()
            SHARED["Wa"] = p.action_net.weight.detach().cpu().numpy().copy()
            SHARED["ba"] = p.action_net.bias.detach().cpu().numpy().copy()
            SHARED["mean"] = rms.mean.copy(); SHARED["var"] = rms.var.copy()
            SHARED["steps"] = int(self.model.num_timesteps)
            if not math.isnan(ep):
                SHARED["ep"] = ep; SHARED["rew"].append(ep)
            SHARED["ready"] = True


def train_thread():
    venv = VecNormalize(DummyVecEnv([make_env for _ in range(N_ENVS)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4,
                clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
                # log_std_init=-1: 잔차 탐험 노이즈 축소(std 1.0→0.37).
                # 29관절이 매 스텝 풀스윙으로 흔들리면 노이즈만으로 넘어져
                # 균형 신호가 묻힌다(위치제어 휴머노이드 표준 설정).
                policy_kwargs=dict(net_arch=NET_ARCH, log_std_init=-1.0),
                device="cpu", verbose=0)
    model.learn(total_timesteps=50_000_000, callback=Snapshot())


def policy_action(obs):
    with LOCK:
        if not SHARED["ready"]:
            return np.zeros(ACT_DIM), np.zeros(NET_ARCH[0]), np.zeros(OBS_DIM)
        W0, b0 = SHARED["W0"], SHARED["b0"]; W2, b2 = SHARED["W2"], SHARED["b2"]
        Wa, ba = SHARED["Wa"], SHARED["ba"]; mean, var = SHARED["mean"], SHARED["var"]
    o = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
    h1 = np.tanh(W0 @ o + b0)
    h2 = np.tanh(W2 @ h1 + b2)
    return Wa @ h2 + ba, h1, o            # 행동, 은닉활성, 정규화 감각입력


# ---------------------------------------------------------------------------
# G1 리스킨(다크네이비) + UNITREE 로고 제거 + 카메라 투영(브랜딩 텍스트 배치)
def restyle(model):
    """SPG S1 다크네이비 바디 + UNITREE 로고 제거(외피 색은 build_model에서 지정).
    모든 뷰어가 build_model 후 호출하는 몸통 색 단일 소스. 색만 변경 → 물리 불변."""
    BODY = (0.105, 0.145, 0.225, 1.0); DARK = (0.040, 0.055, 0.090, 1.0)   # 다크네이비 스틸
    for name, rgba, sp, sh, rf in [("metal", BODY, 0.45, 0.55, 0.18),
                                   ("black", DARK, 0.30, 0.45, 0.10)]:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if i >= 0:
            model.mat_rgba[i] = rgba
            model.mat_specular[i] = sp; model.mat_shininess[i] = sh
            model.mat_reflectance[i] = rf
    lm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "logo_link")
    for g in range(model.ngeom):
        if model.geom_dataid[g] == lm:            # UNITREE 로고 메시 숨김(투명)
            model.geom_matid[g] = -1
            model.geom_rgba[g] = (0.0, 0.0, 0.0, 0.0)


def cam_pos(cam):
    az = math.radians(cam.azimuth); el = math.radians(cam.elevation)
    b = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    return np.array(cam.lookat) - cam.distance * b, b      # campos, 시선전방 b


def project_to_screen(p, cam, fovy_deg):
    campos, fwd = cam_pos(cam)
    right = np.cross(fwd, [0, 0, 1.0]); right /= (np.linalg.norm(right) or 1)
    up = np.cross(right, fwd)
    rel = np.array(p) - campos; xc = rel @ right; yc = rel @ up; zc = rel @ fwd
    if zc <= 0.05:
        return None
    f = 1.0 / math.tan(math.radians(fovy_deg)/2); asp = WIDTH / ARENA_H
    sx = (0.5 + 0.5*(xc/zc)*f/asp) * WIDTH
    sy = (0.5 - 0.5*(yc/zc)*f) * ARENA_H
    return sx, sy


# ---------------------------------------------------------------------------
def panel(screen, rect, alpha=235):
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(s, (*C_PANEL, alpha), s.get_rect(), border_radius=10)
    pygame.draw.rect(s, (*C_PANEL_LINE, 255), s.get_rect(), 1, border_radius=10)
    screen.blit(s, rect.topleft)


def draw_curve(screen, rect, hist, fonts):
    panel(screen, rect)
    screen.blit(fonts["hd"].render("실시간 학습 곡선 (에피소드 보상)", True, C_TEXT), (rect.x+12, rect.y+8))
    px, py = rect.x+44, rect.y+34; pw, ph = rect.width-58, rect.height-58
    pygame.draw.line(screen, C_PANEL_LINE, (px, py+ph), (px+pw, py+ph), 1)
    if len(hist) < 2:
        screen.blit(fonts["sm"].render("학습 데이터 수집 중...", True, C_DIM), (px+10, py+ph/2)); return
    h = hist[-300:]
    lo, hi = min(h), max(h); rng = (hi-lo) or 1.0; n = len(h)
    pts = [(px+pw*(i/max(1, n-1)), py+ph*(1-(v-lo)/rng)) for i, v in enumerate(h)]
    pygame.draw.lines(screen, C_CURVE_BEST, False, pts, 2)
    screen.blit(fonts["sm"].render("현재 %.0f" % h[-1], True, C_CURVE_BEST), (px+6, py))
    screen.blit(fonts["sm"].render("최고 %.0f" % max(h), True, C_CURVE_AVG), (px+6, py+16))


def _node(screen, pos, v):
    v = max(0.0, min(1.0, v))
    glow = int(60 + 160 * v)
    pygame.draw.circle(screen, (int(30+90*v), int(70+150*v), int(90+150*v)),
                       (int(pos[0]), int(pos[1])), 7 if v > 0.5 else 5)
    pygame.draw.circle(screen, (glow, 255, 235), (int(pos[0]), int(pos[1])), 3)


def draw_nn(screen, rect, in_act, h_act, out_act, fonts):
    panel(screen, rect)
    screen.blit(fonts["hd"].render("살아있는 신경망", True, C_TEXT), (rect.x+12, rect.y+8))
    with LOCK:
        W0 = SHARED["W0"]; Wa = SHARED["Wa"]
    x0, x1, x2 = rect.x+50, rect.centerx, rect.right-50
    top, bot = rect.y+38, rect.bottom-22
    NIN, HS = 16, 22
    OS = min(ACT_DIM, 22)
    isel = np.linspace(0, OBS_DIM-1, NIN).astype(int)
    hsel = np.linspace(0, NET_ARCH[0]-1, HS).astype(int)
    osel = np.linspace(0, ACT_DIM-1, OS).astype(int)

    def cy(c, i): return (top+bot)/2 if c == 1 else top+(bot-top)*i/(c-1)
    ip = [(x0, cy(NIN, i)) for i in range(NIN)]
    hp = [(x1, cy(HS, i)) for i in range(HS)]
    op = [(x2, cy(OS, i)) for i in range(OS)]
    if W0 is not None:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for ii, i in enumerate(isel):                                   # 감각→은닉
            sig = min(1.0, abs(in_act[i]) / 2.5)
            for jj, j in enumerate(hsel):
                w = W0[j, i]; a = min(1.0, abs(w) * 2.4)
                col = (255, 176, 80) if w >= 0 else (96, 160, 255)
                al = int(28 + 150 * a * (0.4 + 0.6 * sig))
                pygame.draw.line(s, (*col, al), (ip[ii][0]-rect.x, ip[ii][1]-rect.y),
                                 (hp[jj][0]-rect.x, hp[jj][1]-rect.y), 1)
        for jj, j in enumerate(hsel):                                   # 은닉→관절
            act = abs(h_act[j])
            for kk, k in enumerate(osel):
                w = Wa[k, j]; a = min(1.0, abs(w) * 2.4)
                col = (255, 176, 80) if w >= 0 else (96, 160, 255)
                al = int(28 + 160 * a * (0.4 + 0.6 * act))
                pygame.draw.line(s, (*col, al), (hp[jj][0]-rect.x, hp[jj][1]-rect.y),
                                 (op[kk][0]-rect.x, op[kk][1]-rect.y), 1)
        screen.blit(s, rect.topleft)
    for ii, i in enumerate(isel):                                       # 감각 노드(점등)
        _node(screen, ip[ii], abs(in_act[i]) / 2.5 if W0 is not None else 0.2)
    for jj, j in enumerate(hsel):
        _node(screen, hp[jj], (h_act[j]+1)/2 if W0 is not None else 0.2)
    for kk, k in enumerate(osel):
        _node(screen, op[kk], (math.tanh(out_act[k])+1)/2)
    fs = fonts["sm"]
    screen.blit(fs.render("감각", True, C_DIM), (x0-12, bot+6))
    screen.blit(fs.render("은닉", True, C_DIM), (x1-12, bot+6))
    screen.blit(fs.render("관절", True, C_DIM), (x2-12, bot+6))


# ---------------------------------------------------------------------------
def main():
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption("SPG S1 HUMANOID — PPO 강화학습 (SPG Robotics)")
    clock = pygame.time.Clock()
    FF = "malgungothic,applegothic,gulim,arial"
    fonts = {"num": pygame.font.SysFont("consolas,"+FF, 34, bold=True),
             "hd": pygame.font.SysFont(FF, 15, bold=True),
             "bd": pygame.font.SysFont(FF, 17, bold=True),
             "md": pygame.font.SysFont(FF, 14),
             "sm": pygame.font.SysFont(FF, 12),
             "brand": pygame.font.SysFont("arialblack,arial", 13, bold=True)}

    th = threading.Thread(target=train_thread, daemon=True); th.start()
    start_t = time.time()                 # 학습 시작 시각(경과 시간 표시용)

    disp = G1Env()
    disp.m.vis.global_.offwidth = max(RENDER_W, 640)
    disp.m.vis.global_.offheight = max(RENDER_H, 480)
    restyle(disp.m)                       # 다크네이비 리스킨 + UNITREE 로고 제거
    obs, _ = disp.reset()
    renderer = mujoco.Renderer(disp.m, RENDER_H, RENDER_W)
    cam = mujoco.MjvCamera()
    cam.distance = 2.3; cam.elevation = -9; cam.azimuth = 215    # 정면 3/4
    last_h = np.zeros(NET_ARCH[0]); last_out = np.zeros(ACT_DIM); last_in = np.zeros(OBS_DIM)
    fps_opts = [40, 24, 15, 10, 6]; fps_i = 0
    ep_steps = 0; best_ep = float("-inf")

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_UP:
                    fps_i = min(len(fps_opts)-1, fps_i+1)
                elif e.key == pygame.K_DOWN:
                    fps_i = max(0, fps_i-1)
                elif e.key == pygame.K_q:
                    cam.azimuth -= 6
                elif e.key == pygame.K_e:
                    cam.azimuth += 6
                elif e.key == pygame.K_r:
                    obs, _ = disp.reset(); ep_steps = 0

        # 디스플레이 1스텝 (현재 정책 결정론적)
        act, last_h, last_in = policy_action(obs); last_out = act
        obs, _, done, _, _ = disp.step(act); ep_steps += 1
        if done or ep_steps > 3000:
            obs, _ = disp.reset(); ep_steps = 0

        # MuJoCo 고품질 렌더 → pygame 합성
        pel = disp.d.xpos[1]
        cam.lookat[:] = [pel[0], pel[1], 0.65]
        renderer.update_scene(disp.d, cam)
        img = np.ascontiguousarray(renderer.render())
        surf = pygame.image.frombuffer(img.tobytes(), (RENDER_W, RENDER_H), "RGB")
        surf = pygame.transform.smoothscale(surf, (WIDTH, ARENA_H))
        screen.blit(surf, (0, 0))
        screen.fill(C_BG, (0, ARENA_H, WIDTH, HEIGHT-ARENA_H))

        with LOCK:
            steps = SHARED["steps"]; ep = SHARED["ep"]; rew = list(SHARED["rew"])
        if not math.isnan(ep):
            best_ep = max(best_ep, ep)

        bw = fonts["bd"].render("  SPG S1 HUMANOID · PPO 강화학습  ", True, (16, 18, 26))
        br = pygame.Rect(WIDTH//2-bw.get_width()//2-6, 14, bw.get_width()+12, 36)
        bs = pygame.Surface((br.width, br.height), pygame.SRCALPHA)
        pygame.draw.rect(bs, (*C_ACCENT, 240), bs.get_rect(), border_radius=10)
        screen.blit(bs, br.topleft); screen.blit(bw, (br.x+6, br.y+8))

        el = int(time.time() - start_t); hh, mm, ss = el//3600, (el % 3600)//60, el % 60
        p1 = pygame.Rect(14, 14, 282, 118); panel(screen, p1)
        screen.blit(fonts["md"].render("학습 스텝 (timesteps)", True, C_DIM), (28, 22))
        screen.blit(fonts["num"].render("{:,}".format(steps), True, C_NEON), (28, 40))
        screen.blit(fonts["md"].render("학습 시간  %d:%02d:%02d" % (hh, mm, ss), True, C_ACCENT), (28, 82))
        screen.blit(fonts["sm"].render("PPO · 병렬환경 %d개 · 관절 %d" % (N_ENVS, NU), True, C_DIM), (28, 102))
        p2 = pygame.Rect(WIDTH-258, 60, 244, 70); panel(screen, p2)
        screen.blit(fonts["md"].render("최고 에피소드 보상", True, C_DIM), (WIDTH-244, 68))
        screen.blit(fonts["bd"].render("%.0f" % (best_ep if best_ep > -1e8 else 0), True, C_ACCENT),
                    (WIDTH-244, 90))
        focus = "학습집중 %dfps" % fps_opts[fps_i] if fps_i > 0 else "일반 40fps"
        info = fonts["sm"].render("골반 %.2fm · 전진 %.1fm · [%s]  (↑↓ 조절)"
                                  % (disp.d.qpos[2], disp.d.qpos[0], focus), True, (220, 228, 240))
        screen.blit(info, (20, ARENA_H-22))

        y0 = ARENA_H+8; ph = HEIGHT-y0-30
        draw_nn(screen, pygame.Rect(14, y0, 600, ph), last_in, last_h, last_out, fonts)
        draw_curve(screen, pygame.Rect(628, y0, WIDTH-628-14, ph), rew, fonts)
        screen.blit(fonts["sm"].render("SPG S1 HUMANOID · SPG Robotics · MuJoCo · SB3 PPO · Unitree 방식 · 백그라운드 학습",
                    True, C_DIM), (16, HEIGHT-20))
        screen.blit(fonts["sm"].render("↑↓ 학습집중  Q/E 회전  R 리셋  ESC", True, C_DIM),
                    (WIDTH-280, HEIGHT-20))
        pygame.display.flip()
        clock.tick(fps_opts[fps_i])

    with LOCK:
        SHARED["stop"] = True
    th.join(timeout=3.0)
    pygame.quit()


if __name__ == "__main__":
    main()

"""
unitree_walk_spg.py — Unitree 사전학습 G1 정책을 우리 폴더에서 바로 재생 (standalone + SPG UI)
=====================================================================
legged_gym 설치/클론 없이, 이 파일 하나로 Unitree의 검증된 G1 보행 정책(motion.pt)을
재생한다. 첫 실행 시 Unitree의 g1_12dof 모델(메시) + motion.pt를 unitree_g1/ 에 자동
다운로드하고, deploy_mujoco 로직(obs 47 + PD 토크)을 그대로 쓰되 화면만 SPG S1
Humanoid(딥네이비+앰버)로 합성한다.

  python unitree_walk_spg.py                 # 창에서 재생 (첫 실행 시 자동 다운로드)
  python unitree_walk_spg.py --record w.mp4 --seconds 20
  python unitree_walk_spg.py --selftest      # motion.pt 없이 모델/렌더만 점검(개발용)

조작: ↑↓ 전진속도 명령 · Q/E 카메라 회전 · ESC 종료
※ motion.pt 로드/추론은 본인 PC에서 일어난다(외부 정책 실행).
"""

import os
import sys
import math
import argparse
import urllib.request
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
ASSET = HERE / "unitree_g1"
sys.path.insert(0, str(HERE))
import spg_skin                      # SPG S1 외피(헬멧·바이저·다크네이비·로고제거) 공유 헬퍼
REPO = "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main"
MODEL_DIR = REPO + "/resources/robots/g1_description"
MOTION_URL = REPO + "/deploy/pre_train/g1/motion.pt"
MESHES = [
    "pelvis", "pelvis_contour_link", "left_hip_pitch_link", "left_hip_roll_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
    "right_ankle_pitch_link", "right_ankle_roll_link", "torso_link_23dof_rev_1_0", "logo_link",
    "head_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_rubber_hand", "right_shoulder_pitch_link",
    "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_rubber_hand",
]
# Unitree deploy/deploy_mujoco/configs/g1.yaml 값
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float32)
DEFAULT_ANGLES = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0], np.float32)
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], np.float32)
NUM_ACT, NUM_OBS, SIM_DT, DECIM, PERIOD = 12, 47, 0.002, 10, 0.8

# SPG S1 브랜드
C_BG = (8, 18, 34); C_PANEL = (11, 23, 42); C_LINE = (40, 64, 100)
C_TEXT = (238, 242, 248); C_DIM = (150, 168, 195); C_ACCENT = (255, 176, 0)
W, H, ARENA_H, RW, RH = 1240, 780, 560, 960, 432

SCENE_XML = """<mujoco model="g1_spg_scene">
  <include file="g1_12dof.xml"/>
  <statistic center="0 0 0.8" extent="1.6"/>
  <visual>
    <global offwidth="1024" offheight="640" azimuth="215" elevation="-8"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.4" specular="0.2 0.2 0.2"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.10 0.14 0.20" rgb2="0.06 0.09 0.14"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" condim="3"/>
  </worldbody>
</mujoco>
"""


def ensure_assets(need_motion):
    ASSET.mkdir(exist_ok=True); (ASSET / "meshes").mkdir(exist_ok=True)
    def fetch(url, dst):
        if not dst.exists():
            print(f"[dl] {dst.name}", flush=True); urllib.request.urlretrieve(url, str(dst))
    fetch(MODEL_DIR + "/g1_12dof.xml", ASSET / "g1_12dof.xml")
    for nm in MESHES:
        fetch(MODEL_DIR + f"/meshes/{nm}.STL", ASSET / "meshes" / f"{nm}.STL")
    (ASSET / "scene.xml").write_text(SCENE_XML, encoding="utf-8")
    motion = ASSET / "motion.pt"
    if need_motion:
        fetch(MOTION_URL, motion)
    return ASSET / "scene.xml", motion


def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2*(-qz*qx+qw*qy), -2*(qz*qy+qw*qx), 1-2*(qw*qw+qz*qz)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None); ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    scene_path, motion_path = ensure_assets(need_motion=not args.selftest)
    m = spg_skin.build(str(scene_path)); d = mujoco.MjData(m)   # SPG 외피 적용(시각 전용·물리 불변)
    m.opt.timestep = SIM_DT
    # 초기 자세: 다리 기본각 + 발을 바닥에
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[7:7+NUM_ACT] = DEFAULT_ANGLES
    d.qpos[2] = 0.80; mujoco.mj_forward(m, d)

    if args.selftest:
        policy = None
        print("[selftest] motion.pt 없이 더미정책으로 모델/렌더 점검", flush=True)
    else:
        import torch
        policy = torch.jit.load(str(motion_path)); policy.eval()    # ← 사용자 PC에서 외부 정책 로드

    renderer = mujoco.Renderer(m, RH, RW)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 2.6, -8, 215
    pygame.init(); pygame.font.init()
    FF = "malgungothic,applegothic,gulim,arial"
    fonts = {"bd": pygame.font.SysFont(FF, 18, bold=True), "num": pygame.font.SysFont("consolas," + FF, 30, bold=True),
             "md": pygame.font.SysFont(FF, 14), "sm": pygame.font.SysFont(FF, 12)}
    if args.record:
        import imageio
        screen = pygame.Surface((W, H))
        writer = imageio.get_writer(args.record, fps=50, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        total = int(args.seconds * 50)
    else:
        screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("SPG S1 HUMANOID — Unitree 사전학습 정책 (standalone)")
        total = None
    clock = pygame.time.Clock()
    action = np.zeros(NUM_ACT, np.float32); cmd = np.array([0.5, 0, 0], np.float32)
    counter = 0; frame = 0; running = True

    while running:
        if not args.record:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_UP:
                    cmd[0] = min(1.2, cmd[0] + 0.1)
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_DOWN:
                    cmd[0] = max(-0.4, cmd[0] - 0.1)
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    cam.azimuth -= 6
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_e:
                    cam.azimuth += 6
        # Unitree deploy_mujoco 그대로: PD 토크 + 정책 추론
        target = action * ACTION_SCALE + DEFAULT_ANGLES
        d.ctrl[:] = (target - d.qpos[7:7+NUM_ACT]) * KPS - d.qvel[6:6+NUM_ACT] * KDS
        mujoco.mj_step(m, d); counter += 1
        if counter % DECIM == 0:
            qj = (d.qpos[7:7+NUM_ACT] - DEFAULT_ANGLES) * 1.0
            dqj = d.qvel[6:6+NUM_ACT] * DOF_VEL_SCALE
            grav = get_gravity_orientation(d.qpos[3:7]); omega = d.qvel[3:6] * ANG_VEL_SCALE
            ph = (counter * SIM_DT) % PERIOD / PERIOD
            obs = np.concatenate([omega, grav, cmd * CMD_SCALE, qj, dqj, action,
                                  [math.sin(2*math.pi*ph), math.cos(2*math.pi*ph)]]).astype(np.float32)
            if policy is not None:
                import torch
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            # selftest: action=0 유지(기본자세)

        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.6]
        renderer.update_scene(d, cam); img = np.ascontiguousarray(renderer.render())
        screen.blit(pygame.transform.smoothscale(
            pygame.image.frombuffer(img.tobytes(), (RW, RH), "RGB"), (W, ARENA_H)), (0, 0))
        screen.fill(C_BG, (0, ARENA_H, W, H - ARENA_H))
        bw = fonts["bd"].render("  SPG S1 HUMANOID · Unitree 사전학습 정책 재생  ", True, (16, 18, 26))
        br = pygame.Rect(W // 2 - bw.get_width() // 2 - 6, 14, bw.get_width() + 12, 38)
        bs = pygame.Surface((br.width, br.height), pygame.SRCALPHA)
        pygame.draw.rect(bs, (*C_ACCENT, 240), bs.get_rect(), border_radius=10)
        screen.blit(bs, br.topleft); screen.blit(bw, (br.x + 6, br.y + 9))
        pr = pygame.Rect(14, ARENA_H + 14, 320, 96)
        ps = pygame.Surface((pr.width, pr.height), pygame.SRCALPHA)
        pygame.draw.rect(ps, (*C_PANEL, 235), ps.get_rect(), border_radius=10)
        pygame.draw.rect(ps, (*C_LINE, 255), ps.get_rect(), 1, border_radius=10)
        screen.blit(ps, pr.topleft)
        screen.blit(fonts["md"].render("전진 명령 (↑↓)", True, C_DIM), (28, ARENA_H + 22))
        screen.blit(fonts["num"].render("%.2f m/s" % cmd[0], True, C_ACCENT), (28, ARENA_H + 40))
        screen.blit(fonts["sm"].render("골반 %.2fm · 전진 %.1fm · Unitree legged_gym G1 (12-dof)%s"
                    % (d.qpos[2], d.qpos[0], "  [SELFTEST]" if args.selftest else ""), True, C_DIM),
                    (28, ARENA_H + 82))
        screen.blit(fonts["sm"].render("SPG S1 HUMANOID · powered by Unitree pretrained policy",
                    True, C_DIM), (16, H - 20))
        if args.record:
            writer.append_data(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))); frame += 1
            if frame >= total:
                running = False
        else:
            pygame.display.flip(); clock.tick(50)
    if args.record:
        writer.close(); print(f"[spg] 녹화 저장: {args.record} ({frame} 프레임)")
    pygame.quit()


if __name__ == "__main__":
    main()

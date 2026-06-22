"""
deploy_mujoco_spg.py — Unitree G1 사전학습 정책 재생 (Unitree deploy_mujoco "그대로" + SPG S1 Humanoid UI)
=====================================================================
Unitree의 deploy/deploy_mujoco/deploy_mujoco.py 로직을 **그대로** 사용한다(설정/모델/
PD제어/관측/정책 모두 Unitree 것). 바뀐 것은 화면뿐 — plain mujoco.viewer 대신
SPG S1 Humanoid 브랜드(딥네이비 #0b1f3a + 시그니처 앰버 #ffb000)로 합성한다.

■ 사용법 (Unitree 모델/정책을 그대로 쓰므로 그들 repo 안에서 실행)
  1) git clone https://github.com/unitreerobotics/unitree_rl_gym
  2) cd unitree_rl_gym && pip install -e .   (+ pip install pygame-ce imageio "imageio[ffmpeg]")
  3) 이 파일을 deploy/deploy_mujoco/ 에 복사
  4) cd deploy/deploy_mujoco
     python deploy_mujoco_spg.py g1.yaml                 # 창에서 재생
     python deploy_mujoco_spg.py g1.yaml --record w.mp4  # 창 없이 mp4

조작: ESC 종료 · Q/E 카메라 회전.
※ motion.pt 로드/추론은 사용자 PC에서 일어난다(외부 정책 실행).
"""

import time
import argparse
import numpy as np
import mujoco
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR

# --- SPG S1 Humanoid 브랜드 팔레트 ---
C_BG = (8, 18, 34); C_PANEL = (11, 23, 42); C_LINE = (40, 64, 100)
C_TEXT = (238, 242, 248); C_DIM = (150, 168, 195); C_ACCENT = (255, 176, 0)
W, H, ARENA_H = 1240, 780, 560
RW, RH = 960, 432


def get_gravity_orientation(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([2*(-qz*qx + qw*qy), -2*(qz*qy + qw*qx), 1 - 2*(qw*qw + qz*qz)])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file", type=str)
    ap.add_argument("--record", default=None); ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    import os
    if args.record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    # ----- Unitree deploy_mujoco.py 그대로: 설정 로드 -----
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    simulation_dt = cfg["simulation_dt"]; control_decimation = cfg["control_decimation"]
    kps = np.array(cfg["kps"], dtype=np.float32); kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    ang_vel_scale = cfg["ang_vel_scale"]; dof_pos_scale = cfg["dof_pos_scale"]
    dof_vel_scale = cfg["dof_vel_scale"]; action_scale = cfg["action_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    num_actions = cfg["num_actions"]; num_obs = cfg["num_obs"]
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy(); obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path); d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    m.vis.global_.offwidth = max(RW, 640); m.vis.global_.offheight = max(RH, 480)
    policy = torch.jit.load(policy_path)          # ← 사용자 PC에서 외부 정책 로드

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
        pygame.display.set_caption("SPG S1 HUMANOID — Unitree 사전학습 정책 (deploy)")
        total = None
    clock = pygame.time.Clock(); frame = 0; running = True

    while running:
        if not args.record:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    cam.azimuth -= 6
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_e:
                    cam.azimuth += 6
        # ----- Unitree deploy_mujoco.py 그대로: PD 토크 + 정책 추론 -----
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if counter % control_decimation == 0:
            qj = (d.qpos[7:] - default_angles) * dof_pos_scale
            dqj = d.qvel[6:] * dof_vel_scale
            gravity = get_gravity_orientation(d.qpos[3:7]); omega = d.qvel[3:6] * ang_vel_scale
            period = 0.8; ph = (counter * simulation_dt) % period / period
            obs[:3] = omega; obs[3:6] = gravity; obs[6:9] = cmd * cmd_scale
            obs[9:9+num_actions] = qj; obs[9+num_actions:9+2*num_actions] = dqj
            obs[9+2*num_actions:9+3*num_actions] = action
            obs[9+3*num_actions:9+3*num_actions+2] = np.array([np.sin(2*np.pi*ph), np.cos(2*np.pi*ph)])
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        # ----- SPG S1 Humanoid 합성(화면만 변경) -----
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.6]
        renderer.update_scene(d, cam); img = np.ascontiguousarray(renderer.render())
        surf = pygame.image.frombuffer(img.tobytes(), (RW, RH), "RGB")
        screen.blit(pygame.transform.smoothscale(surf, (W, ARENA_H)), (0, 0))
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
        screen.blit(fonts["md"].render("전진 명령 / 속도", True, C_DIM), (28, ARENA_H + 22))
        screen.blit(fonts["num"].render("%.2f m/s" % cmd[0], True, C_ACCENT), (28, ARENA_H + 40))
        screen.blit(fonts["sm"].render("골반 %.2fm · 전진 %.1fm · Unitree legged_gym G1 (12-dof)"
                    % (d.qpos[2], d.qpos[0]), True, C_DIM), (28, ARENA_H + 82))
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

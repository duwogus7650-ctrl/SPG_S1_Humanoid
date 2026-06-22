"""
SPG S1 (Unitree G1) 정책 재생 뷰어 — 오프라인
=====================================================================
저장된 체크포인트(ppo_model.zip + vecnormalize.pkl)를 불러와, 학습된 정책으로
G1이 걷는 모습을 12번과 같은 고품질 렌더 + '살아있는 신경망' viz로 보여준다.
학습 없이 '결과'만 재생하므로 가볍고, 클라우드(GitHub Actions)에서 학습한
체크포인트를 받아 로컬에서 확인하기에 좋다.

사용:
  python play.py                         # ./checkpoints 의 정책을 창으로 재생
  python play.py path/to/ckpt_dir        # 다른 폴더의 정책 재생
  python play.py ckpt --record out.mp4 --seconds 15   # 창 없이 mp4로 저장(공유/검증용)

체크포인트 폴더에는 ppo_model.zip 과 vecnormalize.pkl 이 함께 있어야 한다
(12_g1_ppo.py/tools/record_training.py 학습 산출물, 또는 GitHub 아티팩트).

조작(창 모드): R 리셋 · Q/E 카메라 회전 · ↑↓ fps · ESC 종료
"""

import os
import sys
import math
import time
import argparse
import importlib.util
from pathlib import Path


def load_g1():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("g1sim", str(here / "12_g1_ppo.py"))
    g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1)
    return g1


def populate_shared(g1, model, venv):
    """로드한 정책 가중치/정규화 통계를 g1.SHARED에 채운다.
    → g1.policy_action(추론)과 g1.draw_nn(신경망 viz)을 그대로 재사용할 수 있다."""
    import torch.nn as nn
    p = model.policy
    pn = p.mlp_extractor.policy_net
    assert g1.NET_ARCH == [256, 256], f"net_arch [256,256] 가정과 다름: {g1.NET_ARCH}"
    assert len(pn) >= 3 and isinstance(pn[0], nn.Linear) and isinstance(pn[2], nn.Linear), \
        f"정책망 구조가 추론 경로(pn[0],pn[2]) 가정과 다름: {pn}"
    rms = venv.obs_rms
    with g1.LOCK:
        g1.SHARED["W0"] = pn[0].weight.detach().cpu().numpy().copy()
        g1.SHARED["b0"] = pn[0].bias.detach().cpu().numpy().copy()
        g1.SHARED["W2"] = pn[2].weight.detach().cpu().numpy().copy()
        g1.SHARED["b2"] = pn[2].bias.detach().cpu().numpy().copy()
        g1.SHARED["Wa"] = p.action_net.weight.detach().cpu().numpy().copy()
        g1.SHARED["ba"] = p.action_net.bias.detach().cpu().numpy().copy()
        g1.SHARED["mean"] = rms.mean.copy()
        g1.SHARED["var"] = rms.var.copy()
        g1.SHARED["steps"] = int(getattr(model, "num_timesteps", 0))
        g1.SHARED["ready"] = True


def main():
    ap = argparse.ArgumentParser(description="SPG S1 정책 재생 뷰어")
    ap.add_argument("ckpt", nargs="?", default="checkpoints",
                    help="체크포인트 폴더(ppo_model.zip + vecnormalize.pkl). 기본 checkpoints")
    ap.add_argument("--record", default=None, help="mp4 경로(주면 창 없이 헤드리스 녹화)")
    ap.add_argument("--seconds", type=float, default=20.0, help="녹화 길이(초). --record 시")
    args = ap.parse_args()
    record = args.record is not None

    if record:                                  # 창 없이 렌더(검증/공유용)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    g1 = load_g1()
    import numpy as np
    import mujoco
    import pygame
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ck = Path(args.ckpt)
    mp, vp = ck / "ppo_model.zip", ck / "vecnormalize.pkl"
    if not mp.exists() or not vp.exists():
        print(f"[play] 체크포인트를 찾을 수 없습니다: {mp} / {vp}\n"
              f"       학습 산출물(ppo_model.zip + vecnormalize.pkl)이 있는 폴더를 지정하세요.\n"
              f"       예) python play.py checkpoints", file=sys.stderr)
        sys.exit(1)

    venv = VecNormalize.load(str(vp), DummyVecEnv([g1.make_env]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(mp), device="cpu")
    populate_shared(g1, model, venv)
    print(f"[play] 로드 완료: {ck}  (학습 스텝 {g1.SHARED['steps']:,})", flush=True)

    disp = g1.G1Env()
    disp.m.vis.global_.offwidth = max(g1.RENDER_W, 640)
    disp.m.vis.global_.offheight = max(g1.RENDER_H, 480)
    g1.restyle(disp.m)
    obs, _ = disp.reset()
    renderer = mujoco.Renderer(disp.m, g1.RENDER_H, g1.RENDER_W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.3, -9, 215

    pygame.init()
    pygame.font.init()
    FF = "malgungothic,applegothic,gulim,arial"
    fonts = {"num": pygame.font.SysFont("consolas," + FF, 34, bold=True),
             "hd": pygame.font.SysFont(FF, 15, bold=True),
             "bd": pygame.font.SysFont(FF, 17, bold=True),
             "md": pygame.font.SysFont(FF, 14),
             "sm": pygame.font.SysFont(FF, 12)}

    if record:
        import imageio
        screen = pygame.Surface((g1.WIDTH, g1.HEIGHT))
        writer = imageio.get_writer(args.record, fps=50, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        total_frames = int(args.seconds * 50)
    else:
        screen = pygame.display.set_mode((g1.WIDTH, g1.HEIGHT), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("SPG S1 HUMANOID — 학습된 정책 재생")
        total_frames = None
    clock = pygame.time.Clock()
    fps_opts = [50, 30, 20, 120]; fps_i = 0

    last_h = np.zeros(g1.NET_ARCH[0]); last_out = np.zeros(g1.ACT_DIM); last_in = np.zeros(g1.OBS_DIM)
    ep_steps = 0; falls = 0; frame = 0; running = True

    while running:
        if not record:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_r:
                        obs, _ = disp.reset(); ep_steps = 0
                    elif e.key == pygame.K_q:
                        cam.azimuth -= 6
                    elif e.key == pygame.K_e:
                        cam.azimuth += 6
                    elif e.key == pygame.K_UP:
                        fps_i = (fps_i + 1) % len(fps_opts)
                    elif e.key == pygame.K_DOWN:
                        fps_i = (fps_i - 1) % len(fps_opts)

        act, last_h, last_in = g1.policy_action(obs); last_out = act
        obs, _r, done, trunc, _ = disp.step(act); ep_steps += 1
        if done or trunc or ep_steps > 3000:
            if done:
                falls += 1
            obs, _ = disp.reset(); ep_steps = 0

        pel = disp.d.xpos[1]
        cam.lookat[:] = [pel[0], pel[1], 0.65]
        renderer.update_scene(disp.d, cam)
        img = np.ascontiguousarray(renderer.render())
        surf = pygame.image.frombuffer(img.tobytes(), (g1.RENDER_W, g1.RENDER_H), "RGB")
        surf = pygame.transform.smoothscale(surf, (g1.WIDTH, g1.ARENA_H))
        screen.blit(surf, (0, 0))
        screen.fill(g1.C_BG, (0, g1.ARENA_H, g1.WIDTH, g1.HEIGHT - g1.ARENA_H))

        # 상단 타이틀
        bw = fonts["bd"].render("  SPG S1 HUMANOID · 학습된 정책 재생  ", True, (16, 18, 26))
        br = pygame.Rect(g1.WIDTH // 2 - bw.get_width() // 2 - 6, 14, bw.get_width() + 12, 36)
        bs = pygame.Surface((br.width, br.height), pygame.SRCALPHA)
        pygame.draw.rect(bs, (*g1.C_ACCENT, 240), bs.get_rect(), border_radius=10)
        screen.blit(bs, br.topleft); screen.blit(bw, (br.x + 6, br.y + 8))

        # 좌상단 정보 패널
        p1 = pygame.Rect(14, 14, 290, 118); g1.panel(screen, p1)
        screen.blit(fonts["md"].render("학습 스텝 (timesteps)", True, g1.C_DIM), (28, 22))
        screen.blit(fonts["num"].render("{:,}".format(g1.SHARED["steps"]), True, g1.C_NEON), (28, 40))
        screen.blit(fonts["sm"].render("재생 모드 · 결정론적 정책", True, g1.C_ACCENT), (28, 84))
        screen.blit(fonts["sm"].render("골반 %.2fm · 전진 %.2fm · 낙상 %d회"
                    % (disp.d.qpos[2], disp.d.qpos[0], falls), True, (220, 228, 240)), (28, 104))

        # 살아있는 신경망 viz(원본 함수 재사용)
        y0 = g1.ARENA_H + 8; ph = g1.HEIGHT - y0 - 30
        g1.draw_nn(screen, pygame.Rect(14, y0, g1.WIDTH - 28, ph), last_in, last_h, last_out, fonts)
        if not record:
            screen.blit(fonts["sm"].render("R 리셋  Q/E 회전  ↑↓ fps  ESC 종료", True, g1.C_DIM),
                        (g1.WIDTH - 260, g1.HEIGHT - 20))

        if record:
            arr = pygame.surfarray.array3d(screen)        # (W,H,3)
            writer.append_data(np.transpose(arr, (1, 0, 2)))
            frame += 1
            if frame >= total_frames:
                running = False
        else:
            pygame.display.flip()
            clock.tick(fps_opts[fps_i])

    if record:
        writer.close()
        print(f"[play] 녹화 저장: {args.record} ({frame} 프레임)", flush=True)
    pygame.quit()


if __name__ == "__main__":
    main()

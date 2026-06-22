"""
run.py — SPG S1 (Unitree G1) 오프라인 실행 런처
=====================================================================
설치 후 한 줄로 시뮬레이션을 본다. 학습된 체크포인트가 없어도 '자연 보행 레퍼런스'를
즉시 재생하므로, pip install 직후 바로 또박또박 걷는 모습을 볼 수 있다.

  python run.py                  # 자연 보행 레퍼런스 재생(즉시, 학습/체크포인트 불필요)
  python run.py reference        # 위와 동일
  python run.py train            # 빈 정책부터 PPO 학습하며 실시간 관찰
  python run.py play [폴더]      # 학습된 정책 재생(기본 ./checkpoints)
  python run.py reference --record walk.mp4 --seconds 15   # 창 없이 mp4 저장

설치:  pip install -r requirements.txt   (자세한 내용 실행방법.md)
조작(창):  Q/E 카메라 회전 · ESC 종료   (train/play는 R 리셋도)
"""

import os
import sys
import math
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_g1():
    spec = importlib.util.spec_from_file_location("g1sim", str(HERE / "12_g1_ppo.py"))
    g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1)
    return g1


def view_reference(record=None, seconds=20.0):
    """IK 자연 보행 레퍼런스를 기구학적으로(물리 없이) 전진 재생 — 목표 걸음새를 본다.
    학습/체크포인트 불필요. 골반을 무보행 속도로 전진시키며 관절=레퍼런스로 렌더."""
    if record:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    g1 = load_g1()
    import numpy as np
    import mujoco
    import pygame

    m = g1.build_model(False)
    m.vis.global_.offwidth = max(g1.RENDER_W, 640)
    m.vis.global_.offheight = max(g1.RENDER_H, 480)
    g1.restyle(m)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, g1.RENDER_H, g1.RENDER_W)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 2.4, -8, 215

    pygame.init(); pygame.font.init()
    FF = "malgungothic,applegothic,gulim,arial"
    fonts = {"bd": pygame.font.SysFont(FF, 17, bold=True),
             "md": pygame.font.SysFont(FF, 14), "sm": pygame.font.SysFont(FF, 12)}
    if record:
        import imageio
        screen = pygame.Surface((g1.WIDTH, g1.HEIGHT))
        writer = imageio.get_writer(record, fps=50, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        total = int(seconds * 50)
    else:
        screen = pygame.display.set_mode((g1.WIDTH, g1.HEIGHT), pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("SPG S1 HUMANOID — 자연 보행 레퍼런스 (목표 걸음새)")
        total = None
    clock = pygame.time.Clock()

    dt = m.opt.timestep * g1.FRAME_SKIP
    speed = 2.0 * g1._REF["STEP_LEN"] / g1.GAIT_PERIOD     # 무보행(no-slip 근사) 전진속도
    phase = 0.0; x = 0.0; frame = 0; running = True
    while running:
        if not record:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    cam.azimuth -= 6
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_e:
                    cam.azimuth += 6
        phase = (phase + dt / g1.GAIT_PERIOD) % 1.0
        x += speed * dt
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[0:3] = [x, 0.0, g1._REF["PELVIS_H"]]; d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = g1.ref_pose(phase)
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [x, 0.0, 0.6]
        renderer.update_scene(d, cam)
        img = np.ascontiguousarray(renderer.render())
        surf = pygame.image.frombuffer(img.tobytes(), (g1.RENDER_W, g1.RENDER_H), "RGB")
        surf = pygame.transform.smoothscale(surf, (g1.WIDTH, g1.ARENA_H))
        screen.blit(surf, (0, 0))
        screen.fill(g1.C_BG, (0, g1.ARENA_H, g1.WIDTH, g1.HEIGHT - g1.ARENA_H))
        bw = fonts["bd"].render("  SPG S1 HUMANOID · 자연 보행 레퍼런스(목표 걸음새)  ", True, (16, 18, 26))
        br = pygame.Rect(g1.WIDTH // 2 - bw.get_width() // 2 - 6, 14, bw.get_width() + 12, 36)
        bs = pygame.Surface((br.width, br.height), pygame.SRCALPHA)
        pygame.draw.rect(bs, (*g1.C_ACCENT, 240), bs.get_rect(), border_radius=10)
        screen.blit(bs, br.topleft); screen.blit(bw, (br.x + 6, br.y + 8))
        screen.blit(fonts["md"].render("발궤적+IK 협응 보행 · 전진 %.1fm" % x, True, g1.C_TEXT), (24, g1.ARENA_H + 16))
        if not record:
            screen.blit(fonts["sm"].render("Q/E 회전  ESC 종료", True, g1.C_DIM), (g1.WIDTH - 150, g1.HEIGHT - 22))
        if record:
            arr = pygame.surfarray.array3d(screen); writer.append_data(np.transpose(arr, (1, 0, 2)))
            frame += 1
            if frame >= total:
                running = False
        else:
            pygame.display.flip(); clock.tick(50)
    if record:
        writer.close(); print(f"[run] 녹화 저장: {record} ({frame} 프레임)")
    pygame.quit()


def main():
    mode = "reference"
    rest = sys.argv[1:]
    if rest and not rest[0].startswith("-"):
        mode = rest[0]; rest = rest[1:]

    if mode == "train":
        load_g1().main()
    elif mode == "play":
        spec = importlib.util.spec_from_file_location("playmod", str(HERE / "play.py"))
        play = importlib.util.module_from_spec(spec); spec.loader.exec_module(play)
        sys.argv = ["play.py"] + rest
        play.main()
    elif mode in ("reference", "ref"):
        ap = argparse.ArgumentParser()
        ap.add_argument("--record", default=None); ap.add_argument("--seconds", type=float, default=20.0)
        a = ap.parse_args(rest)
        view_reference(a.record, a.seconds)
    else:
        print(__doc__)
        sys.exit(0 if mode in ("-h", "--help", "help") else 2)


if __name__ == "__main__":
    main()

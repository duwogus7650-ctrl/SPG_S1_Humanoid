"""
헤드리스 학습 + 자동 녹화 러너 (SPG S1 / 12_g1_ppo.py 기반)
=====================================================================
화면(GUI)이 없는 서버/CI에서 12번 시뮬레이션을 그대로 돌리며,
**같은 정책이 끊김 없이 PPO로 학습하는 동안** 일정 간격(기본 1시간)으로
현재 정책을 15초씩 MuJoCo 오프스크린 렌더 → mp4로 저장한다.
체크포인트(PPO 가중치 + 정규화 통계)를 저장/복원하므로, GitHub Actions의
6시간 잡 한도를 넘겨 여러 세그먼트로 누적 학습(예: 12시간)을 이어갈 수 있다.

핵심 설계
  · 물리/환경/보상/정책/리스킨은 12_g1_ppo.py를 그대로 재사용(동일 시뮬레이션)
  · [백그라운드 스레드] PPO 연속 학습 → Snapshot 콜백이 가중치를 numpy로 공유
  · [메인 스레드] 그 numpy 스냅샷으로 추론·렌더(torch 동시접근 없음 = 안전)
  · 렌더 백엔드 = OSMesa(소프트웨어) → 디스플레이 불필요

환경변수(모두 선택)
  OUT_DIR        클립 저장 폴더            (기본 recordings)
  CKPT_DIR       체크포인트 폴더            (기본 checkpoints)
  DURATION_SEC   이 세그먼트 학습 시간(초)  (기본 19800 = 5h30m, 업로드 마진 확보)
  INTERVAL_SEC   녹화 간격(초)              (기본 3600 = 1시간)
  CLIP_SEC       클립 길이(초)              (기본 15)
  CLIP_FPS       클립 fps                   (기본 50 = 제어주파수, 실시간 재생)
  OFFSET_SEC     이전 세그먼트까지 누적 학습 시간(초) — 클립 시간 라벨용 (기본 0)
  SAVE_EVERY_SEC 체크포인트 저장 주기(초)   (기본 600)
"""

import os
# --- 디스플레이 없는 환경: 소프트웨어 GL + 단일 스레드(라이브러리 충돌 방지) ---
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
# pygame을 창 없이(오프스크린 Surface) 구동 — 원본 12번의 UI를 그대로 합성하기 위함
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import warnings
warnings.filterwarnings("ignore")

# --- torch / dynamo를 mujoco보다 "먼저" 로드·워밍한다 ----------------------
# (mujoco가 먼저 OpenMP를 올린 뒤 Adam이 torch._dynamo를 늦게 임포트하면
#  헤드리스에서 segfault가 난다. 여기서 미리 그 경로를 통과시켜 둔다.)
import torch
import torch._dynamo  # noqa: F401
import torch.optim
torch.optim.Adam([torch.zeros(1, requires_grad=True)])
torch.set_num_threads(1)

import sys
import json
import time
import threading
import importlib.util
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import mujoco
import imageio
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# --- 12_g1_ppo.py를 모듈로 로드해 그대로 재사용 (모듈명이 숫자로 시작해 importlib 사용) ---
_HERE = Path(__file__).resolve().parent
_SIM_PATH = _HERE.parent / "12_g1_ppo.py"
_spec = importlib.util.spec_from_file_location("g1sim", str(_SIM_PATH))
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)


# ---------------------------------------------------------------------------
def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


OUT_DIR = Path(_env("OUT_DIR", "recordings"))
CKPT_DIR = Path(_env("CKPT_DIR", "checkpoints"))
DURATION_SEC = _env("DURATION_SEC", 19800, int)
INTERVAL_SEC = _env("INTERVAL_SEC", 3600, int)
CLIP_SEC = _env("CLIP_SEC", 15, int)
CLIP_FPS = _env("CLIP_FPS", 50, int)
OFFSET_SEC = _env("OFFSET_SEC", 0, int)
SAVE_EVERY_SEC = _env("SAVE_EVERY_SEC", 600, int)
# 첫 롤아웃이 이 시간 내 준비되지 않으면(=학습 스레드가 조용히 멈춤) 중단한다.
READY_TIMEOUT_SEC = _env("READY_TIMEOUT_SEC", 900, int)

# 실시간 업로드(선택): 클립을 찍는 즉시 GitHub Release에 올려 학습 도중에도 본다.
GH_TOKEN = os.environ.get("GH_TOKEN")
GH_REPO = os.environ.get("GH_REPO")               # "owner/repo"
RUN_ID = os.environ.get("RUN_ID", "0")
LIVE_TAG = f"live-run{RUN_ID}"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CKPT_DIR / "ppo_model.zip"
VEC_PATH = CKPT_DIR / "vecnormalize.pkl"
META_PATH = CKPT_DIR / "meta.json"

# 학습 스레드와 공유하는 모델 핸들(최종 저장용)
_TRAIN = {"model": None, "venv": None}


def _find_cjk_font():
    """한글 라벨이 깨지지 않게 CJK 글리프가 있는 폰트를 찾는다(환경별 후보)."""
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",          # fonts-nanum (GH)
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # fonts-noto-cjk
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",             # 이 컨테이너
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):         # 최후(한글X)
        if os.path.exists(p):
            return p
    return None


def build_fonts():
    """원본 12_g1_ppo.py의 fonts 딕셔너리를 헤드리스에서 동일 키로 구성."""
    path = _find_cjk_font()

    def mk(size, bold=False):
        if path:
            f = pygame.font.Font(path, size)
            f.set_bold(bold)
        else:
            f = pygame.font.SysFont("arial", size, bold=bold)
        return f

    return {"num": mk(30, True), "hd": mk(15, True), "bd": mk(17, True),
            "md": mk(14), "sm": mk(12), "brand": mk(13, True)}


# ---------------------------------------------------------------------------
def build_or_load():
    """체크포인트가 있으면 이어서, 없으면 새 PPO 모델을 만든다."""
    if MODEL_PATH.exists() and VEC_PATH.exists():
        venv = VecNormalize.load(
            str(VEC_PATH), DummyVecEnv([g1.make_env for _ in range(g1.N_ENVS)]))
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(str(MODEL_PATH), env=venv, device="cpu")
        print(f"[train] 체크포인트에서 이어서 학습 (timesteps={model.num_timesteps:,})",
              flush=True)
    else:
        venv = VecNormalize(
            DummyVecEnv([g1.make_env for _ in range(g1.N_ENVS)]),
            norm_obs=True, norm_reward=True, clip_obs=10.0)
        # 탐험 하이퍼(LOG_STD/ENT_COEF)도 env로 받는다(기본값=현행, 동작 불변).
        # ablation 승자(예: v2/v4)의 탐험 설정을 본학습에 그대로 적용하기 위함.
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95,
                    ent_coef=float(os.environ.get("ENT_COEF", "0.0")), learning_rate=3e-4,
                    clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=g1.NET_ARCH,
                                       log_std_init=float(os.environ.get("LOG_STD", "-1.0"))),
                    device="cpu", verbose=0)
        print("[train] 새 PPO 모델로 학습 시작", flush=True)
    return model, venv


def save_checkpoint(model, venv):
    # 원자적 저장: 임시파일에 먼저 쓰고 os.replace로 교체한다. 저장 도중 프로세스가
    # 강제 종료돼도 기존 체크포인트가 잘리지 않아, 다음 세그먼트가 항상 온전한 파일을 읽는다.
    tmp_m = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
    tmp_v = VEC_PATH.with_name(VEC_PATH.name + ".tmp")
    model.save(str(tmp_m))
    venv.save(str(tmp_v))
    os.replace(tmp_m, MODEL_PATH)
    os.replace(tmp_v, VEC_PATH)


class PeriodicSave(BaseCallback):
    """일정 주기로 체크포인트 저장 + 정지 신호 감시."""
    def __init__(self, model, venv):
        super().__init__()
        self._m, self._v = model, venv
        self._last = time.time()

    def _on_step(self):
        if time.time() - self._last > SAVE_EVERY_SEC:
            save_checkpoint(self._m, self._v)
            self._last = time.time()
        return not g1.SHARED["stop"]


def train_worker():
    # 예외를 잡지 않으면 데몬 스레드가 조용히 죽고 → 'ready'가 영원히 안 켜져 →
    # 메인 루프가 잡 타임아웃까지 무한 대기한다(에러 로그도 없이). 반드시 신호를 남긴다.
    try:
        model, venv = build_or_load()
        _TRAIN["model"], _TRAIN["venv"] = model, venv
        # Snapshot: 매 롤아웃마다 가중치/정규화 통계를 numpy로 SHARED에 복사(렌더가 사용)
        model.learn(total_timesteps=1_000_000_000,
                    callback=[g1.Snapshot(), PeriodicSave(model, venv)],
                    reset_num_timesteps=False)
        save_checkpoint(model, venv)   # 정지 후 최종 저장
        print("[train] 학습 종료, 체크포인트 저장 완료", flush=True)
    except BaseException as e:          # noqa: BLE001 — 어떤 예외든 메인에 알려야 함
        import traceback
        g1.SHARED["error"] = repr(e)
        g1.SHARED["stop"] = True
        print("[train] !! 학습 스레드 예외 — 중단:\n" + traceback.format_exc(),
              flush=True)


# ---------------------------------------------------------------------------
def compose_frame(screen, disp, render_img, last_in, last_h, last_out,
                  cum_sec, best_ep, fonts):
    """원본 12_g1_ppo.py의 화면(로봇 아레나 + 신경망 + 학습곡선 + 패널)을
    창 없이 Surface에 그대로 합성하고, numpy 프레임(H,W,3)으로 돌려준다."""
    W, H, AH = g1.WIDTH, g1.HEIGHT, g1.ARENA_H

    # 상단: MuJoCo 고품질 렌더를 아레나 크기로 확대
    surf = pygame.image.frombuffer(render_img.tobytes(), (g1.RENDER_W, g1.RENDER_H), "RGB")
    surf = pygame.transform.smoothscale(surf, (W, AH))
    screen.blit(surf, (0, 0))
    screen.fill(g1.C_BG, (0, AH, W, H - AH))

    with g1.LOCK:
        steps = g1.SHARED["steps"]
        rew = list(g1.SHARED["rew"])

    # 상단 브랜딩 바
    bw = fonts["bd"].render("  SPG S1 HUMANOID · PPO 강화학습  ", True, (16, 18, 26))
    br = pygame.Rect(W // 2 - bw.get_width() // 2 - 6, 14, bw.get_width() + 12, 36)
    bs = pygame.Surface((br.width, br.height), pygame.SRCALPHA)
    pygame.draw.rect(bs, (*g1.C_ACCENT, 240), bs.get_rect(), border_radius=10)
    screen.blit(bs, br.topleft)
    screen.blit(bw, (br.x + 6, br.y + 8))

    # 좌상단 패널: 학습 스텝 + 누적 학습시간
    hh, mm, ss = cum_sec // 3600, (cum_sec % 3600) // 60, cum_sec % 60
    p1 = pygame.Rect(14, 14, 282, 118)
    g1.panel(screen, p1)
    screen.blit(fonts["md"].render("학습 스텝 (timesteps)", True, g1.C_DIM), (28, 22))
    screen.blit(fonts["num"].render("{:,}".format(steps), True, g1.C_NEON), (28, 40))
    screen.blit(fonts["md"].render("학습 시간  %d:%02d:%02d" % (hh, mm, ss), True, g1.C_ACCENT), (28, 82))
    screen.blit(fonts["sm"].render("PPO · 병렬환경 %d개 · 관절 %d" % (g1.N_ENVS, g1.NU), True, g1.C_DIM), (28, 102))

    # 우상단 패널: 최고 에피소드 보상
    p2 = pygame.Rect(W - 258, 60, 244, 70)
    g1.panel(screen, p2)
    screen.blit(fonts["md"].render("최고 에피소드 보상", True, g1.C_DIM), (W - 244, 68))
    screen.blit(fonts["bd"].render("%.0f" % (best_ep if best_ep > -1e8 else 0), True, g1.C_ACCENT), (W - 244, 90))

    info = fonts["sm"].render("골반 %.2fm · 전진 %.1fm · [헤드리스 자동 녹화]"
                              % (disp.d.qpos[2], disp.d.qpos[0]), True, (220, 228, 240))
    screen.blit(info, (20, AH - 22))

    # 하단: 살아있는 신경망 + 실시간 학습 곡선 (원본 함수 그대로 사용)
    y0 = AH + 8
    ph = H - y0 - 30
    g1.draw_nn(screen, pygame.Rect(14, y0, 600, ph), last_in, last_h, last_out, fonts)
    g1.draw_curve(screen, pygame.Rect(628, y0, W - 628 - 14, ph), rew, fonts)
    screen.blit(fonts["sm"].render(
        "SPG S1 HUMANOID · SPG Robotics · MuJoCo · SB3 PPO · Unitree 방식 · 백그라운드 학습",
        True, g1.C_DIM), (16, H - 20))

    arr = pygame.surfarray.array3d(screen)        # (W, H, 3)
    return np.transpose(arr, (1, 0, 2))           # (H, W, 3)


def record_clip(disp, renderer, cam, path, cum_sec, screen, fonts):
    """현재 정책(SHARED 스냅샷)으로 CLIP_SEC초 롤아웃을 풀 UI로 렌더해 mp4 저장."""
    obs, _ = disp.reset()
    n_frames = CLIP_SEC * CLIP_FPS
    best_ep = float("-inf")
    writer = imageio.get_writer(
        str(path), fps=CLIP_FPS, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for _ in range(n_frames):
            act, h_act, in_act = g1.policy_action(obs)   # 행동, 은닉활성, 정규화입력
            obs, _, done, _, _ = disp.step(act)
            if done:
                obs, _ = disp.reset()
            ep = g1.SHARED["ep"]
            if ep == ep and ep > best_ep:                # NaN 아니고 갱신
                best_ep = ep
            pel = disp.d.xpos[1]
            cam.lookat[:] = [pel[0], pel[1], 0.65]
            renderer.update_scene(disp.d, cam)
            img = np.ascontiguousarray(renderer.render())
            frame = compose_frame(screen, disp, img, in_act, h_act, act,
                                  cum_sec, best_ep, fonts)
            writer.append_data(frame)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# 실시간 클립 업로드 — 찍는 즉시 GitHub Release 에셋으로 올린다(학습 중 미리보기).
# 업로드 실패는 절대 학습/녹화를 막지 않도록 모두 무시한다.
_release = {"id": None, "init": False}


def _gh_api(method, url, data=None, ctype=None):
    full = url if url.startswith("http") else "https://api.github.com" + url
    req = urllib.request.Request(full, data=data, method=method)
    req.add_header("Authorization", f"Bearer {GH_TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if ctype:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def _ensure_release():
    """run별 라이브 릴리스를 (없으면 생성하고) 준비한다. 실패 시 None."""
    if _release["init"]:
        return _release["id"]
    _release["init"] = True
    if not (GH_TOKEN and GH_REPO):
        print("[live] GH_TOKEN/REPO 없음 → 실시간 업로드 비활성", flush=True)
        return None
    try:
        try:
            body = _gh_api("GET", f"/repos/{GH_REPO}/releases/tags/{LIVE_TAG}")
            _release["id"] = json.loads(body)["id"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            payload = json.dumps({
                "tag_name": LIVE_TAG,
                "name": f"실시간 학습 클립 (run {RUN_ID})",
                "body": "DeepMimic 학습 중 자동 업로드되는 라이브 클립(15초). "
                        "최신 파일일수록 더 학습된 정책입니다.",
                "prerelease": True}).encode()
            body = _gh_api("POST", f"/repos/{GH_REPO}/releases", data=payload,
                           ctype="application/json")
            _release["id"] = json.loads(body)["id"]
        print(f"[live] 릴리스 준비 완료 tag={LIVE_TAG} id={_release['id']}", flush=True)
    except Exception as ex:
        print(f"[live] 릴리스 준비 실패(무시): {ex}", flush=True)
        _release["id"] = None
    return _release["id"]


def publish_clip(path):
    rid = _ensure_release()
    if not rid:
        return
    try:
        data = Path(path).read_bytes()
        url = (f"https://uploads.github.com/repos/{GH_REPO}/releases/{rid}"
               f"/assets?name={path.name}")
        _gh_api("POST", url, data=data, ctype="video/mp4")
        print(f"[live] 업로드 완료 → {path.name} ({len(data)//1024}KB)", flush=True)
    except Exception as ex:
        print(f"[live] 업로드 실패(무시): {ex}", flush=True)


def main():
    print(f"[run] OUT={OUT_DIR}  CKPT={CKPT_DIR}  DURATION={DURATION_SEC}s  "
          f"INTERVAL={INTERVAL_SEC}s  CLIP={CLIP_SEC}s@{CLIP_FPS}fps  "
          f"OFFSET={OFFSET_SEC}s", flush=True)

    th = threading.Thread(target=train_worker, daemon=True)
    th.start()

    # 창 없이 그릴 pygame Surface + 폰트(원본 UI 합성용)
    pygame.init()
    pygame.font.init()
    screen = pygame.Surface((g1.WIDTH, g1.HEIGHT))
    fonts = build_fonts()

    # 디스플레이용 환경 + 오프스크린 렌더러(메인 스레드에서만 사용)
    disp = g1.G1Env()
    disp.m.vis.global_.offwidth = max(g1.RENDER_W, 640)
    disp.m.vis.global_.offheight = max(g1.RENDER_H, 480)
    g1.restyle(disp.m)
    renderer = mujoco.Renderer(disp.m, g1.RENDER_H, g1.RENDER_W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.3, -9, 215

    # 첫 롤아웃(가중치 준비)까지 대기 — 타임아웃을 둬 조용한 행(hang)을 방지한다.
    print("[run] 첫 학습 롤아웃 대기 중...", flush=True)
    _wait0 = time.time()
    while not g1.SHARED["ready"] and not g1.SHARED["stop"]:
        if time.time() - _wait0 > READY_TIMEOUT_SEC:
            g1.SHARED["stop"] = True
            print(f"[run] !! 첫 롤아웃이 {READY_TIMEOUT_SEC}s 내 준비되지 않음 — 중단",
                  flush=True)
            break
        time.sleep(2.0)
    if g1.SHARED["error"]:
        th.join(timeout=10.0)
        raise SystemExit(f"[run] 학습 스레드 실패로 중단: {g1.SHARED['error']}")
    if not g1.SHARED["ready"]:
        th.join(timeout=10.0)
        raise SystemExit("[run] 첫 롤아웃 준비 실패로 중단(체크포인트/환경 확인 필요)")

    seg_start = time.time()
    n_clips = 0

    def cum():  # 누적 학습 시간(초)
        return OFFSET_SEC + int(time.time() - seg_start)

    # 세그먼트 시작 직후 baseline 클립 1개(이어학습이면 '현재 수준' 스냅샷)
    p = OUT_DIR / f"clip_{cum()//3600:02d}h{(cum()%3600)//60:02d}m_steps{g1.SHARED['steps']}.mp4"
    print(f"[rec] baseline → {p.name}", flush=True)
    record_clip(disp, renderer, cam, p, cum(), screen, fonts)
    publish_clip(p)
    n_clips += 1

    next_at = INTERVAL_SEC
    while (time.time() - seg_start) < DURATION_SEC:
        time.sleep(2.0)
        if (time.time() - seg_start) >= next_at:
            c = cum()
            p = OUT_DIR / f"clip_{c//3600:02d}h{(c%3600)//60:02d}m_steps{g1.SHARED['steps']}.mp4"
            print(f"[rec] interval → {p.name}", flush=True)
            record_clip(disp, renderer, cam, p, c, screen, fonts)
            publish_clip(p)
            n_clips += 1
            next_at += INTERVAL_SEC

    # 종료: 학습 정지 → 스레드 마무리(최종 체크포인트 저장) 대기.
    # save_checkpoint는 원자적이므로, 스레드가 제때 못 끝내도 기존 체크포인트는 온전하다.
    g1.SHARED["stop"] = True
    th.join(timeout=180.0)
    train_ok = not th.is_alive()
    if not train_ok:
        print("[run] !! 학습 스레드가 180s 내 종료되지 않음 — 최종 저장이 누락됐을 수 있음"
              "(직전 주기 체크포인트는 온전).", flush=True)

    meta = {"cumulative_seconds": cum(),
            "timesteps": int(g1.SHARED["steps"]),
            "clips_this_segment": n_clips,
            "train_complete": train_ok,
            "error": g1.SHARED["error"]}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[run] 완료. 클립 {n_clips}개, 누적 {meta['cumulative_seconds']}s, "
          f"{meta['timesteps']:,} steps", flush=True)
    if g1.SHARED["error"]:
        raise SystemExit(f"[run] 학습 중 예외 발생: {g1.SHARED['error']}")


if __name__ == "__main__":
    main()

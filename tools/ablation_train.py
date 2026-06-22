"""
보행 ablation 학습기 — 변형(variant) 1개를 헤드리스로 학습 후 평가
=====================================================================
12번 학습 실패 근본원인(레퍼런스가 못 걷는 궤적 → "모방 vs 균형" 충돌, §18)을
풀기 위한 처방을 변형별로 검증한다. 12_g1_ppo.py의 보상가중치/ACTION_SCALE는
env로 덮어쓰고(기본값=현행), PPO 탐험(LOG_STD/ENT_COEF)도 env로 받는다.

녹화/렌더 없음(물리만) → GL 불필요, GitHub Actions 표준 러너에서 가볍게 병렬.
끝나면 결정론적 K에피소드 평가 → ablation_out/<variant>_result.json 저장(아티팩트).

env:
  VARIANT    변형 이름(파일/로그 라벨)          기본 "v"
  STEPS      학습 스텝                          기본 5_000_000
  EVAL_EPS   평가 에피소드 수                    기본 8
  LOG_STD    PPO log_std_init(탐험)             기본 -1.0
  ENT_COEF   PPO ent_coef(탐험)                 기본 0.0
  ACTION_SCALE / W_POSE / W_UP / W_ALIVE ...    12_g1_ppo.py가 직접 읽음(import 전 설정)
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # 창 없이 pygame import
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

# torch를 mujoco보다 먼저 워밍(헤드리스 OpenMP segfault 회피, record_training.py와 동일)
import torch
import torch._dynamo  # noqa: F401
import torch.optim
torch.optim.Adam([torch.zeros(1, requires_grad=True)])
torch.set_num_threads(1)

import json
import time
import importlib.util
from pathlib import Path

import numpy as np

VARIANT = os.environ.get("VARIANT", "v")
STEPS = int(os.environ.get("STEPS", "5000000"))
EVAL_EPS = int(os.environ.get("EVAL_EPS", "8"))
LOG_STD = float(os.environ.get("LOG_STD", "-1.0"))
ENT_COEF = float(os.environ.get("ENT_COEF", "0.0"))


def load_sim():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("g1sim", str(here.parent / "12_g1_ppo.py"))
    g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1)   # ACTION_SCALE/W_* 를 env에서 읽어 들임
    return g1


def main():
    g1 = load_sim()
    print(f"[{VARIANT}] ACTION_SCALE={g1.ACTION_SCALE} CMD_LIN_X={g1.CMD_LIN_X} "
          f"LOG_STD={LOG_STD} ENT_COEF={ENT_COEF} STEPS={STEPS:,}", flush=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = VecNormalize(DummyVecEnv([g1.make_env for _ in range(g1.N_ENVS)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=2048, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=ENT_COEF, learning_rate=3e-4,
                clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=g1.NET_ARCH, log_std_init=LOG_STD),
                device="cpu", verbose=0)
    t = time.time()
    model.learn(total_timesteps=STEPS)
    train_min = (time.time() - t) / 60.0

    out = Path("ablation_out"); out.mkdir(exist_ok=True)
    model.save(str(out / f"{VARIANT}_model.zip"))
    venv.save(str(out / f"{VARIANT}_vec.pkl"))

    # 결정론적 평가(학습 때와 동일한 obs 정규화, 보상정규화 off)
    venv.training = False
    venv.norm_reward = False
    env = g1.G1Env()
    vels, lens, falls, ups, dists, actrates = [], [], [], [], [], []
    flip_hits, flip_tot = 0, 0
    for i in range(EVAL_EPS):
        obs, _ = env.reset(seed=5000 + i)
        x0 = float(env.d.qpos[0]); us = 0.0; n = 0; term = False
        prev = np.zeros(env.action_space.shape, np.float32); pdiff = None; ar = []
        while True:
            nobs = venv.normalize_obs(np.asarray(obs, dtype=np.float64))
            a, _ = model.predict(nobs, deterministic=True)
            diff = a - prev                                # 행동 변화(떨림 측정)
            ar.append(float(np.square(diff).sum()))
            if pdiff is not None:
                flip_hits += int((np.sign(diff) != np.sign(pdiff)).sum()); flip_tot += diff.size
            pdiff = diff; prev = a.copy()
            obs, _r, te, tr, _ = env.step(a)
            us += float(env.d.xmat[1].reshape(3, 3)[2, 2]); n += 1
            if te or tr:
                term = bool(te); break
        d = float(env.d.qpos[0]) - x0
        vels.append(d / max(n * env.dt, 1e-9)); lens.append(n)
        falls.append(1.0 if term else 0.0); ups.append(us / max(n, 1)); dists.append(d)
        actrates.append(float(np.mean(ar)) if ar else 0.0)

    res = dict(variant=VARIANT, steps=STEPS, train_min=train_min,
               mean_fwd_vel=float(np.mean(vels)), mean_ep_len=float(np.mean(lens)),
               fall_rate=float(np.mean(falls)), mean_upright=float(np.mean(ups)),
               mean_fwd_dist=float(np.mean(dists)),
               mean_act_rate=float(np.mean(actrates)),       # 떨림: 스텝간 행동변화 제곱합(낮을수록 부드러움)
               signflip=float(flip_hits / max(flip_tot, 1)),  # 행동 부호반전 비율(낮을수록 부드러움)
               per_ep_len=[int(x) for x in lens],
               config=dict(ACTION_SCALE=g1.ACTION_SCALE, LOG_STD=LOG_STD, ENT_COEF=ENT_COEF))
    (out / f"{VARIANT}_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{VARIANT}] DONE train={train_min:.1f}min "
          f"ep_len={res['mean_ep_len']:.0f}/1500 ({res['mean_ep_len']*env.dt:.1f}s) "
          f"fall={res['fall_rate']:.2f} vel={res['mean_fwd_vel']:.3f} dist={res['mean_fwd_dist']:.2f}m "
          f"act_rate={res['mean_act_rate']:.2f} flip={res['signflip']:.2f}",
          flush=True)


if __name__ == "__main__":
    main()

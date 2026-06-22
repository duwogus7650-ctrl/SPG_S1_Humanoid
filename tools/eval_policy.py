"""
헤드리스 보행 정책 평가 — feedback-runner용 메트릭 산출
=====================================================================
저장된 체크포인트(ppo_model.zip + vecnormalize.pkl)를 로드해, 12_g1_ppo.py의
G1Env에서 **결정론적 롤아웃**을 K개 에피소드 돌리고, 보행 품질을 숫자로 뽑아
results.json으로 쓴다. 이 숫자를 walk_oracle.json(목표 스펙)과 비교하면
"정책이 실제로 V_REF로 직진 보행하며 안 넘어지는가"를 자동 검증할 수 있다.

산출 메트릭(walk_oracle.json과 키 일치):
  mean_fwd_vel   평균 전진속도(m/s)        목표 V_REF(0.8)
  mean_ep_len    평균 생존 스텝(최대 1500)  목표 1500(완주)
  fall_rate      넘어진 에피소드 비율(0~1)  목표 0
  mean_upright   평균 골반 직립도(0~1)      목표 1
참고용(오라클 없음, 리포트에만 표시): mean_ep_return, mean_fwd_dist

GUI/렌더 불필요(물리만 사용) → 디스플레이 없는 Windows/CI에서 그대로 동작.

사용:
  python tools/eval_policy.py --episodes 5 --out out --ckpt-dir checkpoints
환경변수 CKPT_DIR로도 체크포인트 폴더 지정 가능.
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np


def load_sim():
    """12_g1_ppo.py를 모듈로 로드(숫자 시작 파일명이라 importlib 사용)."""
    here = Path(__file__).resolve().parent
    sim_path = here.parent / "12_g1_ppo.py"
    spec = importlib.util.spec_from_file_location("g1sim", str(sim_path))
    g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1)
    return g1


def evaluate(episodes, ckpt_dir, seed0=1000):
    """체크포인트를 로드해 결정론적 롤아웃 → 메트릭 dict 반환."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    g1 = load_sim()
    ckpt = Path(ckpt_dir)
    model_path = ckpt / "ppo_model.zip"
    vec_path = ckpt / "vecnormalize.pkl"
    if not model_path.exists() or not vec_path.exists():
        # 체크포인트가 없으면 평가 불가 → exit 1(feedback-runner는 RUN FAILED로 표시).
        # record_training.py를 한 번이라도 돌려 체크포인트를 만든 뒤 평가하라는 신호.
        raise SystemExit(
            f"[eval] 체크포인트 없음: {model_path} / {vec_path}\n"
            f"       먼저 학습으로 체크포인트를 생성하세요 "
            f"(tools/record_training.py 또는 동등한 저장).")

    # VecNormalize: 학습 때와 '동일한' obs 정규화를 재현(평가 모드, 보상정규화 off).
    venv = VecNormalize.load(str(vec_path),
                             DummyVecEnv([g1.make_env]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(model_path), device="cpu")

    # 메트릭 수집용 환경(내가 직접 스텝을 제어해 물리량을 읽는다).
    env = g1.G1Env()
    dt = env.dt

    vels, lens, falls, ups, rets, dists = [], [], [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        x0 = float(env.d.qpos[0])
        ret = 0.0
        up_sum = 0.0
        steps = 0
        terminated = False
        while True:
            nobs = venv.normalize_obs(np.asarray(obs, dtype=np.float64))
            act, _ = model.predict(nobs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(act)
            ret += float(rew)
            up_sum += float(env.d.xmat[1].reshape(3, 3)[2, 2])  # 골반 z축 상향
            steps += 1
            if term or trunc:
                terminated = bool(term)   # term=넘어짐(not alive), trunc=완주(1500)
                break
        dist = float(env.d.qpos[0]) - x0
        vels.append(dist / max(steps * dt, 1e-9))
        lens.append(float(steps))
        falls.append(1.0 if terminated else 0.0)
        ups.append(up_sum / max(steps, 1))
        rets.append(ret)
        dists.append(dist)

    return {
        "mean_fwd_vel": float(np.mean(vels)),
        "mean_ep_len": float(np.mean(lens)),
        "fall_rate": float(np.mean(falls)),
        "mean_upright": float(np.mean(ups)),
        # 참고용(오라클 없음)
        "mean_ep_return": float(np.mean(rets)),
        "mean_fwd_dist": float(np.mean(dists)),
        "_episodes": int(episodes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default="out")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "checkpoints"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(args.episodes, args.ckpt_dir)

    results_path = out_dir / "eval_results.json"
    results_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval] {args.episodes} episodes → {results_path}")
    for k, v in metrics.items():
        print(f"        {k:16s} = {v:.4f}" if isinstance(v, float) else f"        {k:16s} = {v}")


if __name__ == "__main__":
    main()

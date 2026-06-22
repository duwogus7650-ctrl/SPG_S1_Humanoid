"""
보행 레퍼런스(ref_pose) 자동 최적화 — 열린루프 동역학 안정성 + 전진
=====================================================================
12번 학습 실패의 근본원인: 손합성 레퍼런스가 동역학적으로 못 걷는 궤적
(잔차=0 열린루프에서 ~1.7s 만에 전방 낙상). "넘어지는 동작을 모방하라"가
되어 PPO가 34M 스텝을 학습해도 ~2s 만에 넘어진다(실측).

이 스크립트는 손튜닝(§17이 크라우치/스탠스로 시도 → 악화) 대신, 게이트
상수를 **블랙박스 최적화**해 열린루프 생존시간과 전진거리를 동시에 키운다.
= "실현가능한 레퍼런스" 자동 탐색(경량 궤적최적화). 검증은 초~분 단위(학습 불필요).

목적함수(클수록 좋음):  score = mean_survival_s + W_FWD * mean_fwd_dist
  - 잔차=0(순수 레퍼런스)로 K개 시드 롤아웃 → 낙상까지 시간 + 전진거리 측정
  - 스윙진폭은 하한을 둬 '제자리 서기'로 퇴화 방지(진짜 보행 유지)

출력: reference_optimized.json (최적 상수 + before/after 메트릭). 적용은 사용자가
12_g1_ppo.py 상수에 반영하거나, 결과를 보고 결정.
"""

import os
os.environ.setdefault("PYTHONUTF8", "1")
import json
import importlib.util
from pathlib import Path

import numpy as np

CAP_S = 8.0          # 생존시간 상한(초) — 열린루프는 결국 넘어지므로 과적합 방지
W_FWD = 1.0          # 전진거리 가중(1m ~= 1s 생존)
K_SEEDS = 3          # 후보당 평가 시드 수
N_RANDOM = 220       # 랜덤탐색 샘플 수
SEED = 12345

# (이름, 하한, 상한, 현재값) — ref_pose/step이 모듈 전역으로 읽는 게이트 상수
PARAMS = [
    ("HIP0",       -0.35,  0.05, -0.10),
    ("KNEE0",       0.10,  0.55,  0.20),
    ("ANK0",       -0.35,  0.05, -0.10),
    ("HIP_ROLL0",   0.00,  0.16,  0.06),
    ("A_HIP",       0.20,  0.45,  0.35),   # 스윙 하한 0.2 → 보행 유지
    ("A_KNEE",      0.35,  0.70,  0.55),
    ("A_ANK",       0.05,  0.30,  0.15),
    ("GAIT_PERIOD", 1.0,   2.2,   1.4),
]


def load_sim():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("g1sim", str(here.parent / "12_g1_ppo.py"))
    g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1)
    return g1


def apply(g1, x):
    for (name, _lo, _hi, _cur), v in zip(PARAMS, x):
        setattr(g1, name, float(v))


def score(g1, env, x, k=K_SEEDS, cap_s=CAP_S):
    """잔차=0 열린루프로 생존시간/전진거리 측정 → (score, surv_s, fwd_m)."""
    apply(g1, x)
    cap = int(cap_s / env.dt)
    zero = np.zeros(env.action_space.shape, np.float32)
    survs, fwds = [], []
    for s in range(k):
        env.reset(seed=1000 + s)
        x0 = float(env.d.qpos[0])
        n = 0
        for _ in range(cap):
            _o, _r, term, trunc, _ = env.step(zero)
            n += 1
            if term:
                break
        survs.append(n * env.dt)
        fwds.append(float(env.d.qpos[0]) - x0)
    surv = float(np.mean(survs)); fwd = float(np.mean(fwds))
    return surv + W_FWD * fwd, surv, fwd


def main():
    g1 = load_sim()
    env = g1.G1Env()
    rng = np.random.default_rng(SEED)
    lo = np.array([p[1] for p in PARAMS]); hi = np.array([p[2] for p in PARAMS])
    cur = np.array([p[3] for p in PARAMS])

    base_score, base_surv, base_fwd = score(g1, env, cur, k=5)
    print(f"[base] survival={base_surv:.2f}s fwd={base_fwd:.2f}m score={base_score:.2f}", flush=True)

    best_x, best = cur.copy(), base_score
    best_surv, best_fwd = base_surv, base_fwd
    # 1) 랜덤 전역 탐색
    for i in range(N_RANDOM):
        x = lo + rng.random(len(PARAMS)) * (hi - lo)
        sc, sv, fw = score(g1, env, x)
        if sc > best:
            best, best_x, best_surv, best_fwd = sc, x.copy(), sv, fw
            print(f"[rand {i:3d}] NEW best score={sc:.2f} surv={sv:.2f}s fwd={fw:.2f}m", flush=True)

    # 2) 최적점 주변 국소 정제(좌표 하강, 점점 작은 스텝)
    for step in (0.10, 0.05, 0.02):
        improved = True
        while improved:
            improved = False
            for j in range(len(PARAMS)):
                for sgn in (+1, -1):
                    x = best_x.copy()
                    x[j] = np.clip(x[j] + sgn * step * (hi[j] - lo[j]), lo[j], hi[j])
                    sc, sv, fw = score(g1, env, x, k=5)
                    if sc > best + 1e-6:
                        best, best_x, best_surv, best_fwd = sc, x, sv, fw
                        improved = True
    print(f"[best] survival={best_surv:.2f}s fwd={best_fwd:.2f}m score={best:.2f}", flush=True)

    out = {
        "objective": {"cap_s": CAP_S, "w_fwd": W_FWD, "k_seeds": K_SEEDS},
        "baseline": {name: float(c) for (name, _l, _h, c) in PARAMS},
        "baseline_metrics": {"survival_s": base_surv, "fwd_m": base_fwd, "score": base_score},
        "optimized": {name: float(v) for (name, _l, _h, _c), v in zip(PARAMS, best_x)},
        "optimized_metrics": {"survival_s": best_surv, "fwd_m": best_fwd, "score": best},
    }
    Path("reference_optimized.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[done] wrote reference_optimized.json", flush=True)
    print(f"[summary] open-loop survival {base_surv:.2f}s -> {best_surv:.2f}s "
          f"({best_surv/max(base_surv,1e-9):.1f}x), fwd {base_fwd:.2f}m -> {best_fwd:.2f}m", flush=True)


if __name__ == "__main__":
    main()

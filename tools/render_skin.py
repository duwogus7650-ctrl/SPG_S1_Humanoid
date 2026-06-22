"""SPG S1 외피 렌더 검증 — 12_g1_ppo.build_model()로 모델을 짓고 정면/머리 뷰를 PNG로 저장.
사용: python tools/render_skin.py   (오프스크린; 디스플레이 불필요)
헬멧·바이저 위치 조정 후 결과를 눈으로 확인하기 위한 도구."""
import os, sys, importlib.util
import numpy as np
import imageio.v2 as imageio
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 숫자로 시작하는 모듈명이라 importlib로 로드(메인 가드 있어 import 안전).
def load_g1():
    spec = importlib.util.spec_from_file_location("g1ppo", os.path.join(ROOT, "12_g1_ppo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def head_world(m, d):
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    tpos = d.xpos[tid].copy(); tR = d.xmat[tid].reshape(3, 3).copy()
    # head_link 메시 중심(로컬 0.012,0,0.384)을 월드로
    return tpos + tR @ np.array([0.012, 0.0, 0.384])

def render(m, d, cam_kw, w=640, h=480):
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    for k, v in cam_kw.items():
        setattr(cam, k, v)
    r.update_scene(d, cam)
    img = np.ascontiguousarray(r.render())
    r.close()
    return img

def main():
    g1 = load_g1()
    m = g1.build_model(plate=True)
    g1.restyle(m)                 # 뷰어와 동일: 몸통 다크네이비 + 로고 제거(모델 레벨)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    hw = head_world(m, d)
    body_lookat = [0.0, 0.0, 0.9]

    # 전신 4방위(학습 중 Q/E 회전 시뮬) + 머리 클로즈업
    shots = {
        "e_front.png": dict(lookat=body_lookat, distance=2.6, azimuth=180, elevation=-8),
        "e_head.png":  dict(lookat=list(hw),    distance=0.78, azimuth=180, elevation=-4),
        "e_3q.png":    dict(lookat=body_lookat, distance=2.7, azimuth=215, elevation=-8),
        "e_side.png":  dict(lookat=body_lookat, distance=2.7, azimuth=270, elevation=-8),
        "e_back.png":  dict(lookat=body_lookat, distance=2.7, azimuth=0,   elevation=-8),
    }
    for name, ckw in shots.items():
        img = render(m, d, ckw)
        imageio.imwrite(os.path.join(ROOT, name), img)
        print("wrote", name, ckw)
    print("head world =", np.round(hw, 4))

if __name__ == "__main__":
    main()

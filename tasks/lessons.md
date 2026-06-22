# Lessons (세션 학습 누적)

- **휴머노이드 장식 좌표는 추정 금지 — 실측하라.** 헬멧/바이저를 torso_link-로컬 z=0.235에
  두니 머리가 아니라 가슴에 떨어졌다(실제 머리 z≈0.385). mujoco로 바디 프레임·메시 바운드를
  먼저 측정해 배치할 것. *왜 중요:* 추정 오프셋은 그럴듯해 보여도 한 부위가 통째로 엇나간다.
- **레포에 G1 모델이 2종.** `12_g1_ppo.py`=풀 29-dof(robot_descriptions, torso_link 있음),
  `deploy_mujoco_spg.py`/`unitree_walk_spg.py`/`run.py`=로컬 `unitree_g1/g1_12dof.xml`
  (다리전용, torso_link 없음·메시는 pelvis에 용접). 파일마다 어떤 모델인지 먼저 확인.
- **외피(_SKIN) 추가/변경 후엔 물리 불변을 실측 확인.** plate False/True에서 nq·nv·nu·nbody·
  총질량·body_mass·body_inertia 동일한지 체크. 바디에 명시 inertial이 있어 비충돌 geom은
  질량에 무영향이지만 "가정"하지 말고 매번 검증.
- **납작 박스 장식은 곡면에서 측면 돌출(판자)로 보인다.** 헬멧 같은 곡면엔 납작 타원체가
  훨씬 자연스럽게 감긴다.
- **Windows MuJoCo 오프스크린**: `MUJOCO_GL=osmesa`는 무효 → 기본 GL 사용. 패키지 scene의
  오프스크린 프레임버퍼가 640×480이라 렌더 크기를 그 이하로.

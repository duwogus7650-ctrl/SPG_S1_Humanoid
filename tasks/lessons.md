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
- **한 레포에 모델이 2종이면 외피/스타일 코드도 갈린다.** 풀 G1(torso_link, named material)과
  Unitree 12-dof(pelvis 용접, geom rgba)는 좌표·색지정 방식이 달라 모델별 측정·경로가 필요.
  공유 가능한 12-dof 로직은 `spg_skin.py`로, deploy처럼 외부 레포에 단독 복사되는 파일은
  로컬 모듈 import 금지 → 인라인 + 안전 폴백(try/except).
- **스타일이 build_model(spec)과 restyle(model) 두 곳에 흩어지면 프리뷰≠실제가 된다.**
  recolor/로고는 모든 뷰어가 공통 호출하는 restyle(단일 소스)에 두고, 프리뷰 도구(render_skin)도
  같은 build_model+restyle 경로를 타게 해 승인본=실제 렌더를 보장.
- **MuJoCo: geom의 명시 rgba가 기본값(0.5,0.5,0.5,1)과 다르면 material 색을 덮는다.** 12-dof
  모델 본체는 geom rgba=[0.7..]가 박혀 있어, spg_body 재질만 부여하면 색은 실버로 남고 광택만
  바뀐다(렌더로 적발). 색은 material rgba가 아니라 geom rgba가 결정하므로, 재질의 reflectance
  광택을 쓰려면 `g.material=...`와 `g.rgba=...`를 함께 지정해야 한다. *왜 중요:* "재질 줬으니 됐다"는
  추정이 색만 조용히 어긋나게 만든다 — 코드 검증(잔여 실버=0)만으론 못 잡고 렌더로만 드러났다.
- **외관 변경 후엔 코드 검증 + 실제 렌더 둘 다.** 멀티에이전트 리뷰(6차원·적대적 검증, 50건→확인
  40·기각 10)는 버그 0(물리 불변 비트동일)을 확인했지만, 이후 광택 통일 수정은 코드상 "잔여 실버 0"
  으로 통과해도 렌더에선 실버였다. 수치 점검과 픽셀 확인은 서로 대체 불가.

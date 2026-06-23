"""spg_skin.py — Unitree 12-dof G1(머리·팔·몸통이 pelvis에 용접된 deploy 모델)에
SPG S1 외관을 입히는 공유 헬퍼. unitree_walk_spg.py / deploy_mujoco_spg.py가 사용.

12번(robot_descriptions 풀 29-dof, torso_link 보유)과 달리 이 모델은:
 · named material이 없어 robot geom이 rgba를 직접 지정([0.7..] 실버 / [0.2..] 다크)
 · 머리(head_link)가 pelvis 로컬에 있음 — 실측 z=0.325~0.531(중심 0.428), 폭 ±0.078
→ 그래서 12번 _SKIN(torso_link 프레임)을 그대로 못 쓰고 이 모듈이 pelvis 프레임으로 처리한다.

적용 내용(전부 시각 전용·물리 불변):
 · 헬멧 돔 + 앰버 바이저 렌즈 + 가슴 앰버 코어 geom을 pelvis에 추가(비충돌·무질량)
 · 몸통 geom rgba를 다크네이비로 recolor, UNITREE 로고(logo_link) 투명화
물리 불변: 추가 geom은 contype=conaffinity=0이고, 바디에 명시 inertial이 있어 질량/관성 무영향.
"""
import mujoco

# SPG 팔레트(12번 restyle/build_model과 동일하게 유지) ----------------------------
_BODY = [0.105, 0.145, 0.225, 1.0]   # 본체 다크네이비 스틸
_DARK = [0.040, 0.055, 0.090, 1.0]   # 말단 near-black 네이비
# pelvis 로컬 프레임 외피(실측 좌표) — (타입, size, pos, 재질)
_SKIN = [
    ("ellipsoid", (0.086, 0.092, 0.115), (0.010, 0.0, 0.428), "spg_shell"),  # 헬멧 돔(머리 셸)
    ("ellipsoid", (0.044, 0.062, 0.024), (0.060, 0.0, 0.449), "spg_amber"),  # 앰버 바이저 렌즈(눈높이)
    ("box",       (0.010, 0.0075, 0.030), (0.080, 0.0, 0.295), "spg_core"),  # SPG 가슴 코어 마크
]
_GT = {"box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


def build(xml_path):
    """xml_path(12-dof G1 scene/model)에 SPG 외피를 입혀 컴파일한 MjModel을 반환."""
    spec = mujoco.MjSpec.from_file(str(xml_path))

    def addmat(name, rgba, em, sp, sh, rf):
        mt = spec.add_material(); mt.name = name
        mt.rgba = rgba; mt.emission = em; mt.specular = sp
        mt.shininess = sh; mt.reflectance = rf
    addmat("spg_shell", [0.055, 0.075, 0.135, 1.0], 0.0, 0.15, 0.30, 0.05)  # 매트 다크네이비 헬멧
    addmat("spg_amber", [1.0, 0.69, 0.0, 1.0], 0.25, 0.5, 0.40, 0.3)        # 시그니처 앰버(#FFB000)
    addmat("spg_core",  [1.0, 0.78, 0.25, 1.0], 0.95, 0.6, 0.40, 0.3)       # 발광 코어
    # 몸통 재질: 풀 G1 restyle의 metal/black과 동일 광택(reflectance 포함)으로 통일
    addmat("spg_body", _BODY, 0.0, 0.45, 0.55, 0.18)   # 본체 다크네이비 스틸
    addmat("spg_dark", _DARK, 0.0, 0.30, 0.45, 0.10)   # 말단 near-black 네이비

    # 몸통 recolor + 로고 제거. 12-dof는 named material이 없어 rgba로 매칭하되, spg_body/
    #  spg_dark 재질을 부여해 풀 G1과 같은 금속 광택을 준다. floor(material='grid')는 제외.
    unmatched = 0
    for g in spec.geoms:
        if getattr(g, "meshname", "") == "logo_link":   # UNITREE 로고 투명화
            g.material = ""; g.rgba = [0.0, 0.0, 0.0, 0.0]; continue
        if g.material:                                   # 머티리얼 지정 geom(floor/grid)은 제외
            continue
        r = list(g.rgba)
        # geom의 원래 rgba가 기본값과 다르면 material 색을 덮으므로, 색(rgba)과 재질을 함께 지정.
        if abs(r[0] - 0.7) < 0.06 and abs(r[1] - 0.7) < 0.06:      # 실버 → 네이비
            g.material = "spg_body"; g.rgba = _BODY
        elif abs(r[0] - 0.2) < 0.06 and abs(r[1] - 0.2) < 0.06:    # 다크 → near-black 네이비
            g.material = "spg_dark"; g.rgba = _DARK
        else:
            unmatched += 1                               # 업스트림 rgba 변경 시 시끄럽게 경고
    if unmatched:
        print("[SPG][warn] recolor 미매칭 robot geom %d개 — 업스트림 rgba 변경?" % unmatched, flush=True)

    # 헬멧/바이저/코어 외피를 pelvis에 추가(비충돌·무질량)
    b = spec.body("pelvis")
    for i, (typ, size, pos, mat) in enumerate(_SKIN):
        gg = b.add_geom(); gg.name = "spgskin_%d" % i
        gg.type = _GT[typ]; gg.size = list(size); gg.pos = list(pos)
        gg.material = mat; gg.contype = 0; gg.conaffinity = 0; gg.group = 2
        gg.density = 0.0                                  # 무질량 보장(inertiafromgeom=true 대비)
    return spec.compile()

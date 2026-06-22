# SPG S1 Humanoid

**Unitree G1 휴머노이드 보행 강화학습** — SPG Robotics S1 컨셉.
Unitree legged_gym 방식(레퍼런스 모션 없이 **속도명령 추종 + 보상셰이핑**)으로 G1이 걷는 법을 배운다.
MuJoCo 물리 + Stable-Baselines3 PPO, CPU로 동작(GPU 불필요). GitHub Actions에서 무료로 장시간 학습 가능.

![brand](https://img.shields.io/badge/SPG-S1%20Humanoid-ffb000) ![rl](https://img.shields.io/badge/RL-PPO%20%C2%B7%20MuJoCo-0b1f3a)

---

## 빠른 시작 (오프라인)

```bash
pip install -r requirements.txt        # 처음 한 번
python run.py                          # 자연 보행 즉시 재생(학습/체크포인트 불필요)
```
Windows는 **`run.bat` 더블클릭**으로도 실행됩니다.

### 실행 모드
| 명령 | 설명 |
|---|---|
| `python run.py` | 보행 레퍼런스(목표 걸음새) 즉시 재생 |
| `python run.py train` | 빈 정책부터 PPO 학습하며 실시간 관찰 |
| `python run.py play [폴더]` | 학습된 정책 재생(기본 `./checkpoints`) |

조작: `Q/E` 카메라 회전 · `R` 리셋(train/play) · `ESC` 종료. 자세한 내용은 [실행방법.md](실행방법.md).

---

## 학습 방식 (Unitree 그대로)

레퍼런스 모션을 모방하지 않는다. 대신 **속도명령을 추종**하도록 정교한 보상으로 걸음을 빚는다.
- **행동**: 다리 12관절 목표각(`target = action*0.25 + 기본각`). 팔/허리는 기본자세 고정.
- **관측(47)**: base 각속도 · 투영중력 · 속도명령 · 다리 관절각/속도 · 직전행동 · 보행시계(sin/cos)
- **보상**: 속도추종 + 직립/높이 유지 + 발스윙높이 · 접지패턴 · 미끄럼방지 · 다리벌어짐방지 + 부드러움
  (전부 Unitree legged_gym G1 스케일)

이 방식은 손으로 만든 레퍼런스가 유발하던 버그(무릎 과신전 등)를 **원천적으로** 없앤다.

## 클라우드 학습 (GitHub Actions · 무료)

`Actions → "SPG S1 학습 자동 녹화" → Run workflow` 로 헤드리스 학습을 돌리고,
**1시간마다 15초 클립**을 릴리스에 자동 업로드해 진행을 확인한다(6h 세그먼트 체이닝으로 12h+ 누적).

## 구조
```
12_g1_ppo.py          환경(G1Env) + PPO 학습 + 실시간 GUI(로봇·신경망·학습곡선)
play.py / run.py      오프라인 실행(정책 재생 / 런처) · run.bat(더블클릭)
tools/
  record_training.py  헤드리스 학습+자동 녹화(클라우드용)
  eval_policy.py       정책 보행 메트릭 평가(feedback-runner 연동)
  ablation_train.py    변형 병렬 학습/비교
.github/workflows/    record-training(장시간 학습) · smoke-test(CI) · ablation
```

자세한 개발 경위는 [개발일지.md](개발일지.md), 작업 로그는 [tasks/todo.md](tasks/todo.md).

## Unitree 사전학습 정책 보기 (검증된 G1 보행)

Unitree가 공개한 **검증된 G1 보행 정책**(motion.pt)을 SPG S1 Humanoid 화면으로 본다.
deploy_mujoco 로직을 **그대로** 쓰되 UI만 SPG로 바꿨다.

**① 우리 폴더에서 바로 (권장) — `unitree_walk_spg.py`**
```bash
pip install -r requirements.txt          # 처음 한 번
python unitree_walk_spg.py               # 첫 실행 시 Unitree 모델+motion.pt 자동 다운로드
python unitree_walk_spg.py --record w.mp4 --seconds 20
```
첫 실행이 `unitree_g1/`에 모델·정책을 받는다(이후 캐시). 조작: ↑↓ 속도명령 · Q/E 회전 · ESC.

**② Unitree 저장소 안에서 (drop-in) — `deploy_mujoco_spg.py`**
```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym && cd unitree_rl_gym && pip install -e .
cp /경로/SPG_S1_Humanoid/deploy_mujoco_spg.py deploy/deploy_mujoco/
cd deploy/deploy_mujoco && python deploy_mujoco_spg.py g1.yaml
```
> motion.pt 로드/추론은 본인 PC에서 실행된다(외부 정책). 우리 *자체* 학습 정책은 `python run.py play`로 본다.

---
*SPG Robotics · S1 Humanoid · MuJoCo · Stable-Baselines3 PPO*

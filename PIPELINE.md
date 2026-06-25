# SO-101 정밀 placing — IL 정책 비교 파이프라인 (ACT / Diffusion / SmolVLA / Pi0.5)

목표: sim 내부에서 4개 IL 정책을 **같은 50개 데이터셋**으로 학습하고, **MuJoCo에서 직접
롤아웃해 두 눈으로 시연 + 상대비교**한다.

## 분업 (중요)
- 이 Windows 머신: **CPU 전용 torch, lerobot/transformers 미설치** → 데이터 수집·씬·평가 env만.
- **학습(특히 SmolVLA/Pi0.5)은 GPU + lerobot 풀스택 = B200(Linux)에서.**
  - ACT/Diffusion: 소형이라 lerobot 설치 시 CPU로도 가능(느림).
  - SmolVLA(~0.5B), Pi0.5(~3B): **GPU 필수**.

관측/액션 규약(4정책 공통):
- 입력: `observation.images.front_left|front_right|wrist`(240×320×3) + `observation.state`(관절각 6)
- 출력: `action`(관절 절대각 6). VLA는 task 문장 `"place the block on the target"` 사용.
- 성공: 조적 per-target gauge — **target1 < 3mm, target2 < 5mm**(reach 한계로 target2 완화).

---

## 1) 데이터셋 수집  ── 이 머신(Windows)
```
python scripts/collect_dataset.py --num 50 --fresh
```
→ `out/episodes/ep_0000.npz … ep_0049.npz` (3캠 + state/action + eef_pose + block_pos/quat + dists).
전부 게이트 통과(target1<3mm, target2<5mm). 저장 시 **~20Hz로 솎임**(config.DATASET_SUBSAMPLE=5,
DATASET_FPS=20 — 100Hz 제어를 IL 적정 레이트로). `out/episodes/`를 B200으로 복사.
(기존 100Hz npz가 있으면 `python scripts/subsample_episodes.py`로 한 번 솎기.)
**레이트 정합**: rollout/compare는 정책 action을 DATASET_SUBSAMPLE만큼 반복 적용해 env(100Hz)와 맞춤(코드 반영됨).

## 2) LeRobot 데이터셋 변환  ── B200
```
python scripts/convert_to_lerobot.py --repo-id <USER>/so101_place --raw out/episodes --fps 20
```
→ 3캠 video 스키마 LeRobotDataset 생성(features: front_left/right/wrist + state + action + eef_pose).

## 3) 4개 정책 학습  ── B200 (GPU)
LeRobot CLI는 데이터셋 메타에서 입력 feature(3캠+state)를 자동 구성한다.
```
# ACT
lerobot-train --dataset.repo_id=<USER>/so101_place --policy.type=act \
    --output_dir=outputs/act --batch_size=8 --steps=40000 --policy.device=cuda

# Diffusion Policy
lerobot-train --dataset.repo_id=<USER>/so101_place --policy.type=diffusion \
    --output_dir=outputs/diffusion --batch_size=64 --steps=100000 --policy.device=cuda

# SmolVLA (사전학습 백본 로드 → 적은 데이터에 적합)
lerobot-train --dataset.repo_id=<USER>/so101_place --policy.type=smolvla \
    --output_dir=outputs/smolvla --batch_size=8 --steps=20000 --policy.device=cuda

# Pi0.5 (lerobot 0.5.1: type=pi05 확인됨. pi0 / pi0_fast 도 선택 가능)
lerobot-train --dataset.repo_id=<USER>/so101_place --policy.type=pi05 \
    --output_dir=outputs/pi05 --batch_size=8 --steps=30000 --policy.device=cuda
```
체크포인트: `outputs/<name>/checkpoints/last/pretrained_model`
(스텝/배치는 GPU 메모리·시간 따라 조정. 50개는 작은 데이터셋이라 과적합 주의 — VLA는 사전학습 덕에 유리.)

## 4) MuJoCo 시각 시연 + 상대비교  ── lerobot 설치된 머신
한 정책을 창에서 라이브로 시연(두 눈으로):
```
python scripts/rollout.py --ckpt outputs/act/checkpoints/last/pretrained_model --viewer --episodes 5
```
(`--viewer` 없으면 `out/rollout.mp4` 저장 → 헤드리스 B200에서 영상 받아 보기)

4종 한 번에 상대비교(held-out 시드, 성공률·per-block gauge 표):
```
python scripts/compare_policies.py --episodes 20 \
    --policy act=outputs/act/checkpoints/last/pretrained_model \
    --policy diffusion=outputs/diffusion/checkpoints/last/pretrained_model \
    --policy smolvla=outputs/smolvla/checkpoints/last/pretrained_model \
    --policy pi05=outputs/pi05/checkpoints/last/pretrained_model
```
→ 성공률 순 정렬표. 평가 시드(EVAL_SEED0=70000000+)는 학습 데이터(20260624+)와 분리 = 일반화 비교.

---

## 비교 철학
sim2real 수치예측이 아니라 **sim 내부 self-consistent 상대비교**(정책 순위·카메라구성 가설 점검).
모든 정책이 같은 데이터·env·평가 시드를 쓰므로 순위는 의미 있음.
target2 5mm 완화는 팔 reach 한계 때문이며 4정책에 동일 적용되어 비교 공정성에 영향 없음.

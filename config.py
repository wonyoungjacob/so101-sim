"""SO-101 정밀 placing 환경 공통 상수.

모든 값은 SI 단위(미터, 라디안).
좌표계: follower arm base가 원점. 로봇은 **+Y 방향(작업영역)**을 향한다(so_arm100.xml에서 base를
Z축 180° 회전). 실제 셋업(image.png)의 mm 좌표를 1/1000로 환산해 그대로 사용.
블록 시작 배치는 reset 때 A/B 두 종을 50/50 랜덤으로 고른다(그 외엔 결정론적).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- 씬 ---
# Windows 비ASCII 경로 회피: 항상 ROOT로 chdir 후 이 상대경로로 로드한다.
SCENE_REL = "assets/so101/place_scene.xml"

# --- 로봇 ---
# 5축 + 그리퍼(Jaw) = 6. 이 순서가 actuator/qpos 순서와 동일.
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
N_ACT = 6  # 액션 차원 = 관절 절대각 6
HOME_QPOS = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]  # Jaw=0(열림 근처)
GRASP_SITE = "grasp_site"  # 그리퍼 그립 중심 site (EEF 기준)

# --- 물체(콘크리트 블록 2개) ---
# 90(가로)×57(세로)×30(높이) mm. half-extent.
BLOCK_HALF = (0.045, 0.0285, 0.015)
BLOCK_Z = BLOCK_HALF[2]  # 바닥 위 안착 시 블록 중심 높이(0.015)
# 질량: 390×190×190mm·2.3kg → 밀도 163.4 kg/m³ × 부피(1.539e-4 m³) = 0.0251 kg.
BLOCK_MASS = 0.0251
BLOCKS = ["block1", "block2"]
BLOCK_JOINTS = ["block1_free", "block2_free"]

# 시작 배치 2종(블록별 (x,y,z, qw,qx,qy,qz)). reset 때 A/B 50/50.
#  A: 90변∥X, 57변∥Y(target과 동일 방향), yard 중앙·Y로 앞뒤(간격 20mm).
#  B: 90변∥Y, 57변∥X(Z 90° 회전), yard 중앙·X로 좌우(간격 20mm).
_QID = (1.0, 0.0, 0.0, 0.0)              # 회전 없음
_QZ90 = (0.70710678, 0.0, 0.0, 0.70710678)  # Z축 +90°
START_ARRANGEMENTS = {
    "A": [
        (-0.025, 0.2065, BLOCK_Z, *_QID),
        (-0.025, 0.2835, BLOCK_Z, *_QID),
    ],
    "B": [
        (-0.0635, 0.245, BLOCK_Z, *_QZ90),
        (0.0135, 0.245, BLOCK_Z, *_QZ90),
    ],
}

# --- 목표(target 2개) ---
# 내부영역 93×60mm. 성공판정용 site는 내부영역 중심·z=블록 중심 높이.
TARGET_SITES = ["target1", "target2"]
TARGET_HALF = (0.0465, 0.030)            # 내부영역 half (X,Y)
TARGET_BORDER = 0.005                    # 검정 테두리 폭 5mm
# 주의: 실제 셋업 좌표는 target1=0.1465, target2=0.2545(중심 X)였으나, 소형 프록시 암이
# target2를 '수직'으로 닿지 못해(reach 부족) 두 target을 함께 -X로 30mm 평행이동했다
# (수직 도달 가능한 최소량). 상대배치(테두리 간격 5mm)·yard·카메라·원점은 그대로.
TARGET_SHIFT_X = -0.030  # 적용한 평행이동량(원본 대비)
TARGET_POS = [
    (0.1165, 0.230, BLOCK_Z),   # 원본 0.1465
    (0.2245, 0.230, BLOCK_Z),   # 원본 0.2545
]

# --- 카메라 (front 2 + wrist 1, 실제 셋업과 동일) ---
# 정확한 방향/화각/해상도는 추후 ChArUco 캘리브레이션 값으로 교체(지금은 근사).
# 실제는 640×480 → 그때 CAM_W/H, place_scene.xml 카메라 pos/xyaxes만 바꾸면 됨.
CAM_NAMES = ["cam_front_left", "cam_front_right", "cam_wrist"]
# 실제 카메라는 640×480(4:3). sim은 종횡비·FOV 동일하게 320×240으로 렌더(다운스케일, 렌더 빠름).
# 실해상도 매칭이 필요하면 640/480으로. wrist는 추후 실제값 반영.
CAM_W = 320
CAM_H = 240

# --- 시작 블록 랜덤화 (비전 의존 강제) ---
# 두 90mm 블록을 150mm yard에 두므로 완전 자유배치는 기하학적으로 빡빡하다.
# → 배치 A/B를 베이스로 각 블록에 연속 jitter(위치+yaw)를 주고, 겹침/리치는 rejection으로 거른다.
RANDOMIZE_START = True       # reset 기본: 연속 랜덤(False면 A/B 그대로)
POS_JITTER = 0.010           # 블록 중심 위치 jitter 반경(m), x·y 각각 균일 ±
YAW_JITTER = 0.25            # 블록 yaw jitter(rad) ± (≈±14°). 성공률 보며 조정
REACH_RADIUS_MAX = 0.285     # 블록 중심 원점거리 상한(reach 보존; 먼 블록 grasp 신뢰성)
BLOCK_AABB_GAP = 0.005       # 두 블록 AABB 사이 최소 여유(m)
RESAMPLE_TRIES = 100         # rejection 재시도 횟수

# --- task/성공 판정 ---
# 시방서 허용오차 3mm를 '조적 검측식'으로 판정: 블록을 8꼭짓점(=실제 면)으로 보고
# 기준선·기준면(수평실/레벨/직선자)에서의 수직편차 중 최댓값(gauge, envs/masonry_metrics.py)이
# 이 값 이내면 그 블록 합격. (IoU·중심거리 아님 — 회전오차까지 면 끝에서 잡힌다.)
SUCCESS_THRESH = 0.003

# 데이터셋 프레임 레이트: env 제어는 100Hz(timestep 0.002×decimation 5)지만, IL엔 과도.
# 저장 시 stride만큼 솎아 ~20Hz로(매끈한 조작엔 충분, 용량·학습 효율↑). rollout은 이 stride만큼
# action을 반복 적용해 레이트를 맞춘다(scripts/rollout.py, compare_policies.py).
DATASET_SUBSAMPLE = 5
DATASET_FPS = 20
# target별 '엄격' 허용오차(시방서): target1=3mm, target2=5mm(reach 한계 완화). is_success 판정.
SUCCESS_THRESH_PER = [0.003, 0.005]
# 데이터셋 채택 게이트(완화): 한 번에 수직으로 내려놓되 두 블록 모두 이 값 이내면 채택.
# 사용자 요구(2026-06-25): 전부 ≤10mm + 엄격(3/5mm) 충족분이 ≥20%. 재안착/슬라이딩 금지
# (놓고 옮기는 보정은 IL에 치명적) → 수직 1회 placing만. 분포는 자연히 형성, 게이트로 거름.
COLLECT_THRESH = 0.010
# 단발 placing 체계적 오프셋 보정(feedforward, 재안착 아님): 먼 target2는 reach 한계에서
# IK+release가 늘 같은 방향으로 ~[+9.4,+7.7]mm·yaw+5.8° 치우친다(A+ 다시드 캘리브). 그만큼
# 미리 빼서 조준하면 단발로 목표에 안착. target1은 거의 0. 블록i↔target i.
PLACE_COMP_XY = [[0.0, 1.1], [6.9, 5.4]]   # [target1, target2] world mm
PLACE_COMP_YAW = [0.0, 4.5]                # deg (그립축을 이만큼 미리 돌려 블록 yaw 상쇄)
# rollout 시 env 절단 한도(전문가 데모는 이 값 무시). 100Hz 제어로 긴 에피소드라 크게.
MAX_STEPS = 6000  # env-제어스텝(100Hz). rollout은 정책스텝×DATASET_SUBSAMPLE 만큼 진행.
CONTROL_DECIMATION = 5  # env 1스텝당 mj_step 횟수 (제어주파수 = sim주파수/이 값)

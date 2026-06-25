"""실제 조적 검측을 모사한 오차 측정.

IoU도, 중심간 거리도 아니다. 현장에서 3mm 허용오차를 확인하는 방식 —
"기준선(수평실)·기준면(레벨/직선자)에서 실제 블록 면까지의 수직 편차를 자로 잰 값" —
을 시뮬레이션에서 그대로 모사한다.

블록을 한 점(중심)이 아니라 **8개 꼭짓점(=실제 벽돌면)**으로 보고, 기준자세(target pose,
연직·코스축 정렬)와 비교한다. 편차 벡터를 target 좌표축(=각 측정도구의 기준방향)으로
분해해 도구별 reading을 낸다:

  longitudinal : 코스축(길이방향) 끝맞춤 편차      ← 줄자/줄눈자(단부·줄눈)
  lateral      : 코스 수직방향 면 정렬 편차        ← 수평실(string line)
  level        : 연직 높이 편차(단차)             ← 레벨/레이저 레벨
  plumb        : 기울기를 블록 치수에 투영한 면편차 ← 다림추/수직 레이저
  gauge        : 위 수직 reading들의 최댓값(=어떤 자로도 잡히는 최악 편차) → 합격판정 기준
  yaw_deg/tilt_deg : 정렬/연직 회전오차(참고, 도)
  max_corner   : 꼭짓점 3D 최대거리(참고)

모든 거리값은 SI(미터). 각도는 도(deg).
중심거리와 달리, 회전(yaw/tilt)이 있으면 중심이 맞아도 면 끝에서 편차가 잡힌다
(직선자·수평실이 실제로 잡아내는 것과 동일).
"""

from __future__ import annotations

import numpy as np
import mujoco

# 블록 body 프레임 8꼭짓점 부호(±1). half-extent를 곱해 실제 꼭짓점.
_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _quat2mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def masonry_error(pos, quat, ref_pos, half, ref_quat=_IDENTITY_QUAT) -> dict:
    """블록 실제자세 vs 기준자세(target)의 조적식 편차.

    pos/quat     : 블록 실제 위치(3)·자세 쿼터니언(4, [w,x,y,z]).
    ref_pos      : 기준 위치(목표 중심, 3).
    half         : 블록 half-extent(3) — (x=90/2, y=57/2, z=30/2).
    ref_quat     : 기준 자세(기본 연직·코스축 정렬 = identity).
    반환: dict(거리=m, 각도=deg). 자세한 키 의미는 모듈 docstring 참고.
    """
    half = np.asarray(half, float)
    R = _quat2mat(quat)
    Rref = _quat2mat(ref_quat)
    local = _SIGNS * half                       # (8,3) body frame 꼭짓점
    actual = np.asarray(pos, float) + local @ R.T      # (8,3) world
    ref = np.asarray(ref_pos, float) + local @ Rref.T  # (8,3) world
    dev = actual - ref                          # (8,3) world 편차벡터
    devR = dev @ Rref                           # target 좌표계로 투영(world->ref)

    longitudinal = float(np.abs(devR[:, 0]).max())   # 코스축 X
    lateral = float(np.abs(devR[:, 1]).max())        # 수평실 Y
    level = float(np.abs(devR[:, 2]).max())          # 레벨 Z
    gauge = max(longitudinal, lateral, level)
    max_corner = float(np.linalg.norm(dev, axis=1).max())

    Rrel = Rref.T @ R
    yaw_deg = float(np.degrees(np.arctan2(Rrel[1, 0], Rrel[0, 0])))
    tilt_deg = float(np.degrees(
        np.arccos(np.clip(R[:, 2] @ Rref[:, 2], -1.0, 1.0))))
    plumb = float(np.tan(np.radians(tilt_deg)) * (2 * half[2]))  # 높이에 투영

    return {
        "longitudinal": longitudinal,
        "lateral": lateral,
        "level": level,
        "plumb": plumb,
        "gauge": gauge,
        "max_corner": max_corner,
        "yaw_deg": yaw_deg,
        "tilt_deg": tilt_deg,
    }

"""front/wrist 카메라가 작업영역(target·yard)을 바라보는 각도값을 출력.

각 카메라에 대해:
- 위치(pos, mm)
- 시선 방향(forward = -Zcam, 월드) 단위벡터
- 시선의 방위각(azimuth, +X기준 수평회전)·고각(elevation, 수평면 아래로 -)
- 카메라 프레임의 오일러각(월드 X→Y→Z 순, deg): roll/pitch/yaw
- 광축이 작업평면(z=0.03)에 닿는 조준점(mm) — target·yard와 비교
좌표계: 원점=로봇 베이스, +Y=작업영역 전방, +Z=위. (place_scene.xml 기준)
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

WORK_Z = 0.03  # 작업평면 높이(블록/타깃 영역 근사)


def euler_xyz_deg(R):
    """월드축 고정 X→Y→Z(roll,pitch,yaw) 분해(deg)."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)
    if abs(sy) < 0.9999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def main():
    import os
    os.chdir(C.ROOT)
    m = mujoco.MjModel.from_xml_path(C.SCENE_REL)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    print(f"좌표계: 원점=로봇 베이스, +Y=전방(작업영역), +Z=위.  작업평면 z={WORK_Z*1000:.0f}mm")
    print(f"target1={np.round(np.array(C.TARGET_POS[0])*1000)} target2={np.round(np.array(C.TARGET_POS[1])*1000)} "
          f"yard중심~(-25,245) mm\n")

    for name in C.CAM_NAMES:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        pos = d.cam_xpos[cid].copy()
        R = d.cam_xmat[cid].reshape(3, 3)        # 열=카메라 x,y,z축(월드)
        fwd = -R[:, 2]                            # mujoco 카메라는 -Z를 본다
        fwd = fwd / np.linalg.norm(fwd)
        az = np.degrees(np.arctan2(fwd[1], fwd[0]))     # +X기준 수평 방위각
        el = np.degrees(np.arcsin(fwd[2]))              # 고각(아래로 보면 음수)
        roll, pitch, yaw = euler_xyz_deg(R)
        # 광축이 z=WORK_Z 평면에 닿는 점
        t = (WORK_Z - pos[2]) / fwd[2] if abs(fwd[2]) > 1e-6 else np.nan
        hit = pos + t * fwd
        print(f"[{name}]")
        print(f"  pos(mm)        = {np.round(pos*1000, 1)}")
        print(f"  시선 forward   = {np.round(fwd, 4)}")
        print(f"  방위각 az      = {az:7.2f}°  (+X=0°, +Y=90°)")
        print(f"  고각 el        = {el:7.2f}°  (수평=0°, 아래로 음수)")
        print(f"  오일러 XYZ(deg)= roll={roll:7.2f}  pitch={pitch:7.2f}  yaw={yaw:7.2f}")
        print(f"  조준점@z{WORK_Z*1000:.0f} = {np.round(hit*1000, 1)} mm")
        print(f"  fovy           = {m.cam_fovy[cid]:.2f}°\n")


if __name__ == "__main__":
    main()

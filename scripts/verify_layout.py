"""새 씬 레이아웃 검증: 배치 A/B 렌더(양 카메라) + 각 target 리치 IK 체크.

- out/layout_{A,B}_{left,right}.png 저장.
- 각 target 중심 위로 grasp_site가 도달 가능한지 DLS IK로 확인(위치오차 mm 출력).
"""

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import config as C  # noqa: E402
from envs.so101_place_env import SO101PlaceEnv  # noqa: E402
from envs.ik import dls_ik, arm_dof_ids  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
V_DOWN_LOCAL = np.array([0.0, -1.0, 0.0])   # grasp_site 로컬 접근축 → 월드 아래
DOWN_WORLD = np.array([0.0, 0.0, -1.0])


def render_arrangements(env):
    for arr in ("A", "B"):
        env.reset(options={"arrangement": arr})
        for cam in C.CAM_NAMES:
            img = env._render_cam(cam)
            side = "left" if cam.endswith("left") else "right"
            Image.fromarray(img).save(OUT / f"layout_{arr}_{side}.png")
        d = env._info()["dists"]
        print(f"[render] arrangement {arr}: block-target dists(m) = {[round(x,3) for x in d]}")
    print("[render] saved out/layout_{A,B}_{left,right}.png")


def reach_check(env):
    m, d = env.model, env.data
    site = env._grasp_sid
    dof = arm_dof_ids(m, C.ARM_JOINTS[:5])  # 5축(Jaw 제외)
    print("[reach] grasp_site가 각 target 위(여러 높이)에 도달 가능한지 (위치오차 mm):")
    for ti, (tx, ty, _tz) in enumerate(C.TARGET_POS):
        line = [f"  target{ti+1} ({tx:.3f},{ty:.3f}):"]
        for hz in (0.10, 0.06, 0.03):
            mujoco.mj_resetData(m, d)
            d.qpos[env._arm_qadr] = C.HOME_QPOS
            mujoco.mj_forward(m, d)
            err = dls_ik(m, d, site, dof, np.array([tx, ty, hz]),
                         approach_local=V_DOWN_LOCAL, approach_world=DOWN_WORLD,
                         iters=400, damping=0.05)
            line.append(f"z={hz*100:.0f}cm→{err*1000:5.1f}mm")
        print(" ".join(line))


def main():
    env = SO101PlaceEnv()
    render_arrangements(env)
    reach_check(env)
    env.close()


if __name__ == "__main__":
    main()

"""인터랙티브 뷰어: SO-101 정밀 placing 환경을 이 컴퓨터에서 직접 본다.

- 기본: 전문가 시연(집기→놓기)을 실시간으로 반복 재생.
- --static: 움직임 없이 home 자세로 띄워 마우스로 둘러보기.
- --check: 창 없이 컨트롤러만 잠깐 돌려 정상 동작 확인(검증용).

GPU 불필요(디스플레이만 있으면 됨). 마우스: 좌클릭드래그=회전, 우클릭=이동, 휠=줌.
실행: python scripts/view.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from envs.so101_place_env import SO101PlaceEnv  # noqa: E402
# 주의: scripts.collect_demo(전문가)는 새 씬 기준으로 재작성 전이라 깨질 수 있어
# 시연 재생/--check 모드에서만 지연 import 한다. --static은 전문가 코드 불필요.


def reset_scene(env, arrangement=None):
    """팔 home + 블록 시작 배치로 초기화(새 씬: 블록 2개, 배치 A/B)."""
    opts = {"arrangement": arrangement} if arrangement else None
    env.reset(options=opts)


def expert_actions(plan):
    """전문가 계획을 (action 6,) 시퀀스로 펼친다(노이즈 없음)."""
    prev = np.array(C.HOME_QPOS[:5], float)
    for _name, q_arm, jaw, steps in plan:
        ramp = max(1, steps // 2)
        for i in range(steps):
            alpha = min(1.0, (i + 1) / ramp)
            arm = (1 - alpha) * prev + alpha * q_arm
            yield np.concatenate([arm, [jaw]]).astype(np.float32)
        prev = q_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", action="store_true", help="움직임 없이 둘러보기")
    ap.add_argument("--check", action="store_true", help="창 없이 동작만 검증")
    args = ap.parse_args()

    env = SO101PlaceEnv()
    m, d = env.model, env.data
    dt = m.opt.timestep * C.CONTROL_DECIMATION  # env 1스텝의 실제 시간(초)

    # 전문가 시연이 필요한 모드에서만 collect_demo를 지연 import.
    demo = None
    if not args.static:
        import scripts.collect_demo as demo  # noqa: E402

    if args.check:
        env.reset()
        plan = demo.plan_from_state(env, verbose=True)
        for action in expert_actions(plan):
            d.ctrl[:] = action
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(m, d)
        dists = [x * 1000 for x in env._info()["dists"]]
        print(f"[check] OK, per-block dist={[round(x,1) for x in dists]}mm (창 없이 검증)")
        return

    env.reset()
    with mujoco.viewer.launch_passive(m, d) as v:
        print("뷰어 실행 중. 창을 닫으면 종료. (정적 보기는 --static)")
        while v.is_running():
            if args.static:
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(0.02)
                continue
            # 매 회 새 배치(A/B)로 리셋하고 그 상태의 전문가 plan을 만들어 실시간 재생
            plan = demo.plan_from_state(env)
            for action in expert_actions(plan):
                if not v.is_running():
                    break
                t0 = time.time()
                d.ctrl[:] = action
                for _ in range(C.CONTROL_DECIMATION):
                    mujoco.mj_step(m, d)
                v.sync()
                time.sleep(max(0, dt - (time.time() - t0)))  # 실시간 페이싱
            time.sleep(0.8)       # 잠깐 멈췄다가
            env.reset()           # 새 배치로 반복


if __name__ == "__main__":
    main()

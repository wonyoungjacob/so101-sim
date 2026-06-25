"""저장된 에피소드를 MuJoCo 창에서 라이브로 재생(두 눈으로 시연).

이미지/GIF가 아니라 실제 MuJoCo 뷰어 창을 띄워 저장된 궤적을 그대로 보여준다.
저장 state(관절각)·block_pos·block_quat로 매 프레임 qpos를 복원→mj_forward→viewer.sync.
마우스로 시점 회전/줌 가능. 창을 닫으면 종료. (기본: 최적 ep 자동 + 반복재생)

사용:
  python scripts/view_episode.py                 # 최적 ep(=최소 gauge) 반복 재생
  python scripts/view_episode.py --ep 19 --speed 2 --once
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


def best_ep(out_ep: Path) -> int:
    best, bg = None, 1e9
    for p in sorted(out_ep.glob("ep_*.npz")):
        g = float(np.max(np.load(p)["dists"][-1]))
        if g < bg:
            bg, best = g, int(p.stem.split("_")[1])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=-1, help="-1=자동(최소 gauge)")
    ap.add_argument("--speed", type=float, default=2.0, help="재생 속도 배율")
    ap.add_argument("--once", action="store_true", help="1회 재생 후 정지(기본=반복)")
    args = ap.parse_args()

    out_ep = ROOT / "out" / "episodes"
    ep = best_ep(out_ep) if args.ep < 0 else args.ep
    d = np.load(out_ep / f"ep_{ep:04d}.npz")
    state, bpos, bquat = d["state"], d["block_pos"], d["block_quat"]
    gz = d["dists"][-1] * 1000
    T = len(state)
    dt = (C.DATASET_SUBSAMPLE * C.CONTROL_DECIMATION * 0.002) / max(args.speed, 1e-6)
    print(f"[view] ep_{ep:04d}  최종 gauge=[{gz[0]:.2f} {gz[1]:.2f}]mm  "
          f"{T}프레임  {args.speed}x  (창을 닫으면 종료)", flush=True)

    env = SO101PlaceEnv()
    m, data = env.model, env.data

    def apply(t):
        data.qpos[env._arm_qadr] = state[t]
        for i, adr in enumerate(env._block_qadr):
            data.qpos[adr:adr + 3] = bpos[t, i]
            data.qpos[adr + 3:adr + 7] = bquat[t, i]
        mujoco.mj_forward(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:
        # 작업영역을 보기 좋게 비추는 초기 시점
        viewer.cam.lookat[:] = [0.09, 0.245, 0.03]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = -90.0
        viewer.cam.elevation = -25.0
        while viewer.is_running():
            for t in range(T):
                if not viewer.is_running():
                    break
                apply(t)
                viewer.sync()
                time.sleep(dt)
            if args.once:
                # 마지막 프레임 유지하며 창 열어둠
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.1)
                break
            time.sleep(0.5)  # 반복 사이 잠깐
    env.close()


if __name__ == "__main__":
    main()

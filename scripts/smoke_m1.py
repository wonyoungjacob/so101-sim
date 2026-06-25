"""M1 smoke test: 환경 reset/step/관측 검증 + 카메라 프레임 저장."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.so101_place_env import SO101PlaceEnv  # noqa: E402
import config as C  # noqa: E402


def main() -> None:
    env = SO101PlaceEnv()
    print("action_space", env.action_space.shape,
          "low", np.round(env.action_space.low, 2),
          "high", np.round(env.action_space.high, 2))

    obs, info = env.reset()
    for k, v in obs.items():
        print(f"obs[{k}] shape={v.shape} dtype={v.dtype}")
    print("reset arrangement", info["arrangement"],
          "dists", [round(d, 4) for d in info["dists"]],
          "eef", np.round(info["eef_pose"][:3], 3))

    # 임의 액션 몇 스텝: 팔이 실제로 움직이는지(state 변화) 확인
    a0 = obs["state"].copy()
    action = np.array(C.HOME_QPOS, np.float32)
    action[0] += 0.3  # Rotation 살짝
    action[1] += 0.3  # Pitch 살짝
    for _ in range(20):
        obs, r, term, trunc, info = env.step(action)
    moved = float(np.linalg.norm(obs["state"] - a0))
    print(f"after 20 steps: state-delta={moved:.3f} reward={r:.4f} term={term} trunc={trunc}")

    (ROOT / "out").mkdir(exist_ok=True)
    Image.fromarray(obs["front_left"]).save(ROOT / "out" / "m1_step_left.png")
    Image.fromarray(obs["front_right"]).save(ROOT / "out" / "m1_step_right.png")
    print("saved out/m1_step_{left,right}.png")

    assert moved > 0.05, "팔이 거의 안 움직임 — 제어 확인 필요"
    assert obs["front_left"].shape == (C.CAM_H, C.CAM_W, 3)
    env.close()
    print("[M1] OK")


if __name__ == "__main__":
    main()

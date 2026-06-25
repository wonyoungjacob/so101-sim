"""수집한 데이터셋에서 '가장 잘 놓인' 에피소드를 골라 시연 GIF로 만든다.

출력 2종:
 1) out/best_ep{N}_3cam.gif    — 저장된 그대로의 3캠(front_left|front_right|wrist) 타일.
       정책이 실제로 보게 될 '데이터셋 그 자체'.
 2) out/best_ep{N}_overview.gif — 같은 에피소드를 3인칭 자유카메라로(보기 쉬움).
       저장된 state(관절각)·block_pos·block_quat로 매 프레임 qpos를 복원→mj_forward→overview 렌더.
       (시드 재현·물리 재실행 불필요. 저장된 궤적을 그대로 다시 그림.)

가장 잘 놓인 기준: 두 블록 최종 조적 gauge의 최댓값이 가장 작은 에피소드(--ep로 지정 가능).

사용:
  python scripts/replay_episode.py                 # 최적 ep 자동, 두 GIF
  python scripts/replay_episode.py --ep 11 --stride 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.so101_place_env import SO101PlaceEnv  # noqa: E402


def rank_episodes(out_ep: Path):
    """저장된 npz들의 최종 조적 gauge로 순위. 반환 [(ep_idx, g0_mm, g1_mm)] 정렬."""
    rows = []
    for p in sorted(out_ep.glob("ep_*.npz")):
        idx = int(p.stem.split("_")[1])
        d = np.load(p)["dists"][-1] * 1000.0  # (2,) mm = 조적 gauge
        rows.append((idx, float(d[0]), float(d[1])))
    rows.sort(key=lambda r: max(r[1], r[2]))
    return rows


def save_gif(frames, path, stride, duration_ms):
    imgs = [Image.fromarray(f) for f in frames[::stride]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=False)
    return len(imgs)


def make_3cam_gif(npz_path, out_path, stride, duration_ms):
    d = np.load(npz_path)
    fl, fr, wr = d["front_left"], d["front_right"], d["wrist"]
    tiled = [np.concatenate([fl[t], fr[t], wr[t]], axis=1) for t in range(len(fl))]
    return save_gif(tiled, out_path, stride, duration_ms)


def overview_camera(model):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.09, 0.245, 0.03]
    cam.distance = 0.62
    cam.azimuth = -90.0
    cam.elevation = -24.0
    return cam


def make_overview_gif(env, npz_path, out_path, stride, duration_ms):
    """저장된 state·block 자세로 qpos를 복원해 overview 카메라로 렌더."""
    d = np.load(npz_path)
    state = d["state"]            # (T,6) 관절각
    bpos = d["block_pos"]         # (T,2,3)
    bquat = d["block_quat"]       # (T,2,4)
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = overview_camera(env.model)
    frames = []
    for t in range(0, len(state), stride):
        env.data.qpos[env._arm_qadr] = state[t]
        for i, adr in enumerate(env._block_qadr):
            env.data.qpos[adr:adr + 3] = bpos[t, i]
            env.data.qpos[adr + 3:adr + 7] = bquat[t, i]
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=cam, scene_option=env._vopt)
        frames.append(renderer.render())
    renderer.close()
    return save_gif(frames, out_path, 1, duration_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=-1, help="-1=자동(최적)")
    ap.add_argument("--stride", type=int, default=2, help="프레임 추출 간격(20Hz 기준)")
    ap.add_argument("--duration", type=int, default=60, help="GIF 프레임당 ms")
    ap.add_argument("--no-overview", action="store_true")
    args = ap.parse_args()

    out_ep = ROOT / "out" / "episodes"
    rows = rank_episodes(out_ep)
    if not rows:
        raise SystemExit(f"no episodes in {out_ep}")

    print("[ranking] 최종 조적 gauge 작은 순 top 8 (ep: target1, target2 mm)")
    for idx, d0, d1 in rows[:8]:
        print(f"  ep_{idx:04d}: [{d0:5.2f} {d1:5.2f}]  max={max(d0, d1):5.2f}")

    ep = rows[0][0] if args.ep < 0 else args.ep
    sel = next(r for r in rows if r[0] == ep)
    print(f"\n[선택] ep_{ep:04d}  최종 gauge=[{sel[1]:.2f} {sel[2]:.2f}]mm")

    npz = out_ep / f"ep_{ep:04d}.npz"
    g1 = ROOT / "out" / f"best_ep{ep:04d}_3cam.gif"
    n1 = make_3cam_gif(npz, g1, args.stride, args.duration)
    print(f"[1] 3캠 데이터셋 GIF: {g1.name}  ({n1} frames)")

    if not args.no_overview:
        env = SO101PlaceEnv()
        g2 = ROOT / "out" / f"best_ep{ep:04d}_overview.gif"
        n2 = make_overview_gif(env, npz, g2, args.stride, args.duration)
        env.close()
        print(f"[2] overview GIF: {g2.name}  ({n2} frames)")


if __name__ == "__main__":
    main()

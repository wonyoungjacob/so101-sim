"""M3: out/episodes/*.npz (원시 시연) → LeRobotDataset.

⚠️ 이 스크립트는 lerobot 풀스택이 깔린 환경(B200 Linux)에서 실행한다.
Windows 시현 머신에는 lerobot을 설치하지 않는다(분업 구조).

LeRobotDataset 스키마:
  observation.images.front_left  : (H,W,3) uint8  영상
  observation.images.front_right : (H,W,3) uint8  영상
  observation.images.wrist       : (H,W,3) uint8  영상 (손목 캠, 실제 셋업과 동일)
  observation.state              : (6,)  관절각(=action과 동일 표현)
  action                         : (6,)  관절 절대각 목표
  + 추가 컬럼 eef_pose(7) : 추후 action_keys 전환(EEF/delta) 실험용 보존

사용:
  python scripts/convert_to_lerobot.py --repo-id <user>/so101_place \
      --raw out/episodes --fps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="예: yourname/so101_place")
    ap.add_argument("--raw", default=str(ROOT / "out" / "episodes"))
    ap.add_argument("--fps", type=int, default=C.DATASET_FPS,
                    help=f"데이터셋 fps(기본 {C.DATASET_FPS}=서브샘플 후 실제 레이트)")
    ap.add_argument("--push", action="store_true", help="허브 업로드")
    args = ap.parse_args()

    # lerobot은 서버 전용 의존성 → 함수 내부에서 import
    # lerobot 0.5.x: 모듈 경로에서 'common' 제거됨(lerobot.datasets.*)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = Path(args.raw)
    eps = sorted(raw.glob("ep_*.npz"))
    if not eps:
        raise SystemExit(f"no episodes in {raw}")
    sample = np.load(eps[0])
    h, w = sample["front_left"].shape[1:3]

    features = {
        "observation.images.front_left": {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.images.front_right": {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.images.wrist": {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.state": {
            "dtype": "float32", "shape": (6,), "names": ["joints"]},
        "action": {
            "dtype": "float32", "shape": (6,), "names": ["joints"]},
        "eef_pose": {
            "dtype": "float32", "shape": (7,), "names": ["pose"]},
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, use_videos=True)

    for ep_path in eps:
        d = np.load(ep_path)
        n = len(d["action"])
        for t in range(n):
            ds.add_frame({
                "observation.images.front_left": d["front_left"][t],
                "observation.images.front_right": d["front_right"][t],
                "observation.images.wrist": d["wrist"][t],
                "observation.state": d["state"][t].astype(np.float32),
                "action": d["action"][t].astype(np.float32),
                "eef_pose": d["eef_pose"][t].astype(np.float32),
                "task": "place the block on the target",
            })
        ds.save_episode()
        print(f"saved {ep_path.name} ({n} frames)")

    print(f"[M3] LeRobotDataset 생성 완료: {args.repo_id}  episodes={len(eps)}")
    if args.push:
        ds.push_to_hub()
        print("[M3] pushed to hub")


if __name__ == "__main__":
    main()

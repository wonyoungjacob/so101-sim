"""기존 out/episodes/*.npz를 시간축으로 stride만큼 솎아 ~20Hz로 줄인다(용량·fps 정정).

100Hz로 기록된 에피소드(평균 ~3300프레임)를 IL에 적정한 20Hz(stride 5)로 다운샘플.
모든 시계열 배열을 동일 stride로 슬라이스해 정합 유지. 한 번만 실행(재실행 시 더 줄어듦).

사용: python scripts/subsample_episodes.py            # config.DATASET_SUBSAMPLE 사용
      python scripts/subsample_episodes.py --stride 5
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
    ap.add_argument("--stride", type=int, default=C.DATASET_SUBSAMPLE)
    args = ap.parse_args()
    out_ep = ROOT / "out" / "episodes"
    eps = sorted(out_ep.glob("ep_*.npz"))
    if not eps:
        raise SystemExit(f"no episodes in {out_ep}")

    tot_before = tot_after = 0
    for p in eps:
        d = np.load(p)
        n = len(d["action"])
        sub = {k: d[k][:: args.stride] for k in d.files}  # 모든 배열 시간축 솎기
        np.savez_compressed(p, **sub)
        tot_before += n
        tot_after += len(sub["action"])
    mb = sum(q.stat().st_size for q in out_ep.glob("ep_*.npz")) / 1e6
    print(f"[subsample] stride={args.stride}  {len(eps)} eps  "
          f"frames {tot_before}->{tot_after}  (~{C.DATASET_FPS}Hz)  총 {mb:.0f}MB")
    print("  rollout/compare는 action을 stride만큼 반복 적용해 레이트 일치(이미 반영).")


if __name__ == "__main__":
    main()

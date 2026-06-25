"""정밀 placing 시연 데이터셋 N개 수집 (3캠: front_left/right + wrist).

collect_demo.run_episode 재사용. 블록 시작은 연속 랜덤(위치+yaw)이고, 매 에피소드
reset된 실제 블록자세에서 IK를 다시 풀어 시연한다(collect_demo.plan_from_state).

데이터셋 품질 게이트:
- 파지·이송이 정상 완료(=각 블록이 목표 근처에 안착)한 에피소드만 채택.
- 파국적 grasp 실패(블록 미파지로 수십~수백 mm)는 제외(--valid-thresh, 기본 30mm).
- 3mm 이내 '엄격 성공'은 별도로 집계만(동적 안착편차로 block2가 종종 3~8mm).
  → 라벨은 "전문가가 블록을 집어 목표에 올바르게 내려놓은" 양질 시연.

저장: out/episodes/ep_0000.npz .. ep_{N-1}.npz  (convert_to_lerobot.py 입력 호환)
각 npz: front_left/front_right/wrist (T,H,W,3) uint8, state/action (T,6),
        eef_pose (T,7), block_pos/target_pos/dists.

사용:
  python scripts/collect_dataset.py --num 50 --fresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from envs.so101_place_env import SO101PlaceEnv  # noqa: E402
from scripts.collect_demo import run_episode_precise  # noqa: E402


def save_preview(traj, path):
    """첫 채택 에피소드의 마지막 프레임 3캠을 가로로 붙여 미리보기 저장."""
    fl = traj["front_left"][-1]
    fr = traj["front_right"][-1]
    wr = traj["wrist"][-1]
    grid = np.concatenate([fl, fr, wr], axis=1)
    Image.fromarray(grid).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50, help="수집할 에피소드 수")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="시도 상한(0=num*8). 게이트 통과가 더뎌도 무한루프 방지")
    ap.add_argument("--fresh", action="store_true",
                    help="기존 out/episodes/ep_*.npz 삭제 후 새로 수집")
    ap.add_argument("--seed0", type=int, default=20260624,
                    help="시드 베이스(재현용)")
    args = ap.parse_args()

    out_ep = ROOT / "out" / "episodes"
    out_ep.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for p in out_ep.glob("ep_*.npz"):
            p.unlink()
        print(f"[fresh] cleared old episodes in {out_ep}")

    max_attempts = args.max_attempts or args.num * 8
    env = SO101PlaceEnv()

    # 정밀 전문가(공중정렬+모르타르 안착+사후 재안착, 노이즈 없음)로 수집.
    # 수직 1회 placing(보정 없음). 게이트=두 블록 조적 gauge ≤ COLLECT_THRESH(10mm) → 채택.
    # 엄격(target1<3mm·target2<5mm = is_success) 충족분 비율도 집계(목표 ≥20%).
    # 성능: 실패 시도는 record=False로 빠르게 검사, 통과분만 record=True 재실행해 렌더·저장.
    kept = 0
    attempt = 0
    strict_ok = 0          # 채택분 중 엄격(3/5mm) 충족 수
    per_block = []         # 채택분 per-block 최종 gauge(mm)
    while kept < args.num and attempt < max_attempts:
        seed = args.seed0 + attempt
        _o, info, _t = run_episode_precise(env, record=False, seed=seed)   # 빠른 검사
        dmm = np.asarray(info["dists"]) * 1000.0
        ok = bool(np.all(dmm <= C.COLLECT_THRESH * 1000.0))                 # 게이트 ≤10mm
        strict = bool(info["is_success"])                                   # 엄격 3/5mm
        print(f"[try {attempt:03d}] arr={info['arrangement']:<3} "
              f"gauge={np.round(dmm, 2)}mm -> "
              f"{'KEEP' if ok else 'drop'}{' (3/5mm)' if (ok and strict) else ''}",
              flush=True)
        if ok:
            _o, info, traj = run_episode_precise(env, record=True, seed=seed)  # 렌더·저장
            traj = {k: v[::C.DATASET_SUBSAMPLE] for k, v in traj.items()}       # ~20Hz 솎기
            np.savez_compressed(out_ep / f"ep_{kept:04d}.npz", **traj)
            if kept == 0:
                save_preview(traj, ROOT / "out" / "dataset_preview_3cam.png")
            per_block.append(np.asarray(info["dists"]) * 1000.0)
            strict_ok += int(strict)
            kept += 1
        attempt += 1

    env.close()
    per_block = np.asarray(per_block) if per_block else np.zeros((0, 2))
    print("\n" + "=" * 60)
    print(f"[dataset] 저장 {kept}/{args.num}  (시도 {attempt}, "
          f"채택률 {kept/max(attempt,1)*100:.0f}%)  *오차=조적 gauge, 수직 1회 placing")
    if kept:
        sr = strict_ok / kept * 100
        print(f"  게이트: 전부 ≤{C.COLLECT_THRESH*1000:.0f}mm")
        print(f"  엄격(target1<3mm,target2<5mm) 충족: {strict_ok}/{kept} = {sr:.0f}% "
              f"{'[OK >=20%]' if sr >= 20 else '[!] <20% 목표미달'}")
        print(f"  per-block gauge mm: mean={per_block.mean(0).round(2)} "
              f"max={per_block.max(0).round(2)} min={per_block.min(0).round(2)}")
        print(f"  미리보기 3캠: out/dataset_preview_3cam.png")
    if kept < args.num:
        print(f"  ⚠ 목표 미달: 시도 상한({max_attempts}) 도달. --max-attempts 늘리세요.")
    print(f"  저장 위치: {out_ep}")


if __name__ == "__main__":
    main()

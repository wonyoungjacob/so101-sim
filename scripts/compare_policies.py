"""여러 LeRobot 정책(ACT/Diffusion/SmolVLA/Pi0.5)을 SO101PlaceEnv에서 롤아웃 평가·비교.

sim 내부 상대비교: 학습에 쓰지 않은 held-out 시작배치(eval 시드)로 각 정책을 N 에피소드
롤아웃해 성공률(조적 per-target gauge: target1<3mm, target2<5mm)과 per-block 오차 분포를
집계 → 정책 순위표.

⚠️ 정책 추론에 torch+lerobot 필요(+VLA는 GPU). B200(Linux)에서 실행.
이 머신(Windows, CPU, lerobot 미설치)에선 ACT/Diffusion만 lerobot 설치 시 가능.

사용:
  python scripts/compare_policies.py \
      --policy act=outputs/act/checkpoints/last/pretrained_model \
      --policy diffusion=outputs/diffusion/.../pretrained_model \
      --policy smolvla=outputs/smolvla/.../pretrained_model \
      --policy pi05=outputs/pi05/.../pretrained_model \
      --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from envs.so101_place_env import SO101PlaceEnv  # noqa: E402

EVAL_SEED0 = 70000000  # 학습 데이터 시드(20260624+)와 겹치지 않는 held-out 평가 시드


def load_policy(ckpt: str):
    """LeRobot 체크포인트(디렉터리)에서 정책 로드. 정책 타입은 저장된 config로 자동 판별."""
    import torch
    policy = None
    try:
        from lerobot.common.policies.factory import make_policy_from_pretrained
        policy = make_policy_from_pretrained(ckpt)
    except Exception:
        # 구/신 버전 호환: 타입별 클래스 직접 시도
        from lerobot.common.policies.factory import get_policy_class
        import json
        cfg = json.loads((Path(ckpt) / "config.json").read_text())
        policy = get_policy_class(cfg["type"]).from_pretrained(ckpt)
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    return policy, device


def obs_to_batch(obs, device):
    import torch

    def img(x):
        t = torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
        return t.unsqueeze(0).to(device)

    return {
        "observation.images.front_left": img(obs["front_left"]),
        "observation.images.front_right": img(obs["front_right"]),
        "observation.images.wrist": img(obs["wrist"]),
        "observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device),
    }


def eval_policy(policy, device, env, seeds):
    import torch
    succ = 0
    per_block = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        # 정책 ~20Hz → env 100Hz: action을 DATASET_SUBSAMPLE만큼 반복 적용해 레이트 일치.
        for _ in range(C.MAX_STEPS // C.DATASET_SUBSAMPLE):
            with torch.no_grad():
                action = policy.select_action(obs_to_batch(obs, device))
            a = action.squeeze(0).cpu().numpy()
            for _rep in range(C.DATASET_SUBSAMPLE):
                obs, _r, term, trunc, info = env.step(a)
                if term or trunc:
                    done = True
                    break
            if done:
                break
        succ += int(info["is_success"])
        per_block.append(np.asarray(info["dists"]) * 1000.0)
    per_block = np.asarray(per_block)
    return succ, per_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", default=[],
                    help="name=ckpt_dir (여러 번). 예: act=outputs/act/.../pretrained_model")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    if not args.policy:
        raise SystemExit("최소 하나의 --policy name=ckpt 필요")

    seeds = [EVAL_SEED0 + i for i in range(args.episodes)]
    env = SO101PlaceEnv()
    rows = []
    for spec in args.policy:
        name, ckpt = spec.split("=", 1)
        print(f"[eval] {name}  ({ckpt})")
        policy, device = load_policy(ckpt)
        succ, pb = eval_policy(policy, device, env, seeds)
        rows.append((name, succ, pb))
        print(f"   success {succ}/{args.episodes}  "
              f"per-block gauge mean={pb.mean(0).round(2)}mm")
    env.close()

    print("\n" + "=" * 64)
    print(f"{'policy':<12}{'success':>10}{'rate':>8}"
          f"{'tgt1 mm':>12}{'tgt2 mm':>12}")
    print("-" * 64)
    for name, succ, pb in sorted(rows, key=lambda r: -r[1]):
        rate = succ / args.episodes * 100
        m = pb.mean(0)
        print(f"{name:<12}{succ:>7}/{args.episodes:<2}{rate:>7.0f}%"
              f"{m[0]:>11.2f}{m[1]:>12.2f}")
    print("=" * 64)
    print(f"평가 기준: 조적 gauge per-target (target1<{C.SUCCESS_THRESH_PER[0]*1000:.0f}mm, "
          f"target2<{C.SUCCESS_THRESH_PER[1]*1000:.0f}mm), held-out 시드 {args.episodes}개")


if __name__ == "__main__":
    main()

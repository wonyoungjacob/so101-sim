"""학습된 LeRobot 정책을 SO101PlaceEnv에서 롤아웃 — MuJoCo로 직접 시연.

- 정책 입력: observation.images.front_left/right/wrist + observation.state(관절각 6)
- 정책 출력: action(관절 절대각 6) → env.step
- 시연: --viewer 면 MuJoCo 창에서 라이브로 본다(두 눈으로 확인). 아니면 영상/PNG 저장.
- 성공 판정: 조적 per-target gauge(target1<3mm, target2<5mm).

정책 추론에 torch+lerobot 필요(+VLA는 GPU). B200(Linux) 또는 lerobot 설치된 머신에서 실행.
ACT/Diffusion은 CPU로도 시연 가능(느림). SmolVLA/Pi0.5는 GPU 권장.

사용:
  python scripts/rollout.py --ckpt outputs/act/.../pretrained_model --viewer --episodes 5
  python scripts/rollout.py --ckpt <dir> --episodes 5            # 영상 저장(헤드리스)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from envs.so101_place_env import SO101PlaceEnv  # noqa: E402

EVAL_SEED0 = 70000000  # held-out(학습 미사용) 평가 시드


def load_policy(ckpt: str):
    """LeRobot 0.5.x 체크포인트에서 정책 로드(config.json의 type으로 클래스 판별)."""
    import torch
    import json
    # lerobot 0.5.x: 모듈 경로에서 'common' 제거, make_policy_from_pretrained 없음
    # → get_policy_class(type).from_pretrained(ckpt)
    from lerobot.policies.factory import get_policy_class
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    ptype = cfg.get("type") or cfg.get("policy_type")
    policy = get_policy_class(ptype).from_pretrained(ckpt)
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


def save_video(frames, path):
    try:
        import imageio
        imageio.mimsave(path, frames, fps=20)
        print(f"[rollout] saved {path}")
    except Exception as e:
        from PIL import Image
        d = Path(path).with_suffix("")
        d.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(d / f"{i:04d}.png")
        print(f"[rollout] imageio 미사용({e}) → PNG 시퀀스 {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="학습된 정책 체크포인트 디렉터리")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--viewer", action="store_true", help="MuJoCo 창에서 라이브 시연")
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0, help="held-out 평가 시드 베이스")
    args = ap.parse_args()

    import torch
    policy, device = load_policy(args.ckpt)
    env = SO101PlaceEnv()

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    n_success = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed0 + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        frames = []
        done = False
        # 정책은 ~20Hz(데이터셋 레이트). env는 100Hz라 정책 action을 DATASET_SUBSAMPLE만큼
        # 반복 적용해 레이트를 맞춘다. 정책 스텝 예산 = MAX_STEPS/SUBSAMPLE.
        for _ in range(C.MAX_STEPS // C.DATASET_SUBSAMPLE):
            with torch.no_grad():
                action = policy.select_action(obs_to_batch(obs, device))
            a = action.squeeze(0).cpu().numpy()
            for _r in range(C.DATASET_SUBSAMPLE):
                obs, _rew, term, trunc, info = env.step(a)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(0.005)
                elif _r == C.DATASET_SUBSAMPLE - 1:
                    frames.append(np.asarray(obs["front_left"]))
                if term or trunc:
                    done = True
                    break
            if done:
                break
        n_success += int(info["is_success"])
        dmm = [round(x * 1000, 1) for x in info["dists"]]
        print(f"[ep {ep}] per-block gauge={dmm}mm success={info['is_success']}")
        if frames and ep == 0:
            save_video(frames, str(ROOT / "out" / "rollout.mp4"))

    print(f"[rollout] success {n_success}/{args.episodes}  "
          f"(target1<{C.SUCCESS_THRESH_PER[0]*1000:.0f}mm, target2<{C.SUCCESS_THRESH_PER[1]*1000:.0f}mm)")
    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()

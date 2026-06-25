"""학습 체크포인트(outputs/)를 주기적으로 HF에 백업 — 정전/세션종료 대비 off-B200 보관.

현재 로그인 계정 네임스페이스의 model repo <user>/so101-checkpoints 로 올린다(기본 public →
로컬/다른 계정에서 인증 없이 회수 가능). HF는 해시로 중복 제거하므로 새 체크포인트만 올라간다.

사용(B200, 백그라운드 데몬):
  nohup python scripts/hf_backup.py --interval 900 > logs/sync.log 2>&1 &
회수(어디서나, public이면 인증 불필요):
  hf download <user>/so101-checkpoints --repo-type model --local-dir outputs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import whoami, create_repo, upload_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", help="백업할 디렉터리(기본 outputs)")
    ap.add_argument("--repo", default="", help="대상 repo(기본 <user>/so101-checkpoints)")
    ap.add_argument("--interval", type=int, default=900, help="업로드 주기 초(기본 900=15분)")
    ap.add_argument("--private", action="store_true", help="비공개로 생성(기본 public)")
    args = ap.parse_args()

    user = whoami()["name"]
    repo = args.repo or f"{user}/so101-checkpoints"
    create_repo(repo, repo_type="model", private=args.private, exist_ok=True)
    vis = "private" if args.private else "PUBLIC"
    print(f"[backup] {repo} ({vis})  every {args.interval}s  src={args.out}", flush=True)
    print(f"[backup] https://huggingface.co/{repo}", flush=True)

    out = Path(args.out)
    while True:
        if out.exists() and any(out.rglob("*.safetensors")):
            try:
                upload_folder(
                    folder_path=str(out),
                    path_in_repo="outputs",
                    repo_id=repo,
                    repo_type="model",
                    commit_message="ckpt sync",
                    allow_patterns=["*.safetensors", "*.json", "*.txt",
                                    "*.yaml", "*.yml", "*.bin", "*.md"],
                )
                print("[backup] synced", flush=True)
            except Exception as e:
                print(f"[backup] ERROR: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

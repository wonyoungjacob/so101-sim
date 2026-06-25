"""현재 씬의 top-down 좌표계 그림을 out/scene_layout.png 로 그린다.

원점 = follower arm(로봇). 단위 mm. X→오른쪽, Y→위, Z→화면 밖.
포함: 로봇(원점), 카메라 2, yard(블록 시작영역, 배치 A/B), target 2(−30mm 이동 반영,
원래 위치는 점선). config.py 값을 그대로 읽어 동기화.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import config as C  # noqa: E402

M = 1000.0  # m → mm


def rect_centered(ax, cx, cy, hw, hh, **kw):
    ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, **kw))


def main():
    fig, ax = plt.subplots(figsize=(11, 7.5))

    # --- yard 150x150, 좌하단 (-100,170) ---
    yard_bl = (-100, 170)
    ax.add_patch(Rectangle(yard_bl, 150, 150, facecolor="#cc8552",
                           edgecolor="#7a4a26", lw=1.5, alpha=0.85, zorder=1))
    ax.text(-25, 170 + 150 + 8, "Yard 150x150\nbottom-left (-100,170,0)", ha="center",
            va="bottom", fontsize=9, color="#7a4a26")

    # --- 블록(배치 A: 실선, 배치 B: 점선) 90x57 ---
    bh = (C.BLOCK_HALF[0] * M, C.BLOCK_HALF[1] * M)  # (45, 28.5)
    for (x, y, z, qw, qx, qy, qz) in C.START_ARRANGEMENTS["A"]:
        rect_centered(ax, x * M, y * M, bh[0], bh[1], facecolor="none",
                      edgecolor="#333", lw=1.8, zorder=3)
    for (x, y, z, qw, qx, qy, qz) in C.START_ARRANGEMENTS["B"]:
        # B는 90변이 Y축 → half가 (28.5,45)
        rect_centered(ax, x * M, y * M, bh[1], bh[0], facecolor="none",
                      edgecolor="#888", lw=1.2, ls=":", zorder=3)
    ax.text(-25, 150, "Block start A(solid)/B(dotted)\n90x57x30mm, 0.0251kg", ha="center",
            va="top", fontsize=8, color="#333")

    # --- target 2개: 내부 93x60 + 검정 테두리 5mm. 현재(이동 후) + 원래(점선) ---
    in_hw, in_hh = C.TARGET_HALF[0] * M, C.TARGET_HALF[1] * M  # 46.5, 30
    b = C.TARGET_BORDER * M  # 5
    shift = C.TARGET_SHIFT_X * M  # -30
    for i, (cx, cy, cz) in enumerate(C.TARGET_POS):
        cx, cy = cx * M, cy * M
        # 검정 테두리(외곽 채움) + 내부 영역(연두)
        rect_centered(ax, cx, cy, in_hw + b, in_hh + b, facecolor="#111",
                      edgecolor="none", zorder=4)
        rect_centered(ax, cx, cy, in_hw, in_hh, facecolor="#7fdf7f",
                      edgecolor="none", zorder=5)
        ax.text(cx, cy, f"T{i+1}", ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=6)
        # 원래 위치(이동 전): 점선 외곽
        ox = cx - shift
        rect_centered(ax, ox, cy, in_hw + b, in_hh + b, facecolor="none",
                      edgecolor="#999", lw=1.2, ls="--", zorder=4)
    ax.text(C.TARGET_POS[1][0] * M, 230 + 45,
            f"Target inner 93x60 + black border 5mm (gap 5mm)\n"
            f"shifted -X by {abs(shift):.0f}mm (dashed = original pos)\n"
            f"centers now: T1({C.TARGET_POS[0][0]*M:.1f},230) T2({C.TARGET_POS[1][0]*M:.1f},230)",
            ha="center", va="bottom", fontsize=8, color="#225522")

    # --- 로봇(원점) + 카메라 ---
    # 카메라 간 거리 770mm, 원점 중심 대칭 → X=±385, 높이 450
    for (x, y, z, name, col) in [
        (0, 0, 0, "Follower Arm (0,0,0)", "#b22222"),
        (-385, 0, 450, "Cam1 (-385,0,450)", "#1f6fb2"),
        (385, 0, 450, "Cam2 (385,0,450)", "#1f6fb2"),
    ]:
        ax.plot(x, y, "X", color=col, ms=13, zorder=7)
        ax.text(x, y - 18, name, ha="center", va="top", fontsize=9, color=col)
    ax.annotate("", xy=(385, -42), xytext=(-385, -42),
                arrowprops=dict(arrowstyle="<->", color="#1f6fb2", alpha=0.6, lw=1))
    ax.text(0, -46, "cam-to-cam 770mm (symmetric about origin)", ha="center",
            va="top", fontsize=8, color="#1f6fb2")

    # 카메라 시선(작업중심 0.09,0.245 방향)
    wc = (90, 245)
    for cx in (-385, 385):
        ax.annotate("", xy=wc, xytext=(cx, 0),
                    arrowprops=dict(arrowstyle="->", color="#1f6fb2", alpha=0.35, lw=1))

    # --- 좌표축 ---
    ax.annotate("", xy=(-380, 470), xytext=(-450, 470),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.annotate("", xy=(-450, 540), xytext=(-450, 470),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
    ax.text(-372, 470, "X", va="center", fontsize=11)
    ax.text(-450, 548, "Y", ha="center", fontsize=11)
    ax.text(-450, 455, "Z (out of page)", ha="center", va="top", fontsize=8, color="#2a7a2a")

    ax.set_aspect("equal")
    ax.set_xlim(-520, 360)
    ax.set_ylim(-70, 600)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("SO-101 precise placing scene layout (origin = Follower Arm, mm)")
    ax.grid(True, ls=":", alpha=0.4)

    out = ROOT / "out" / "scene_layout.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

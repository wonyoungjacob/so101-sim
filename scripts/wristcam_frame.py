"""wrist cam 마운트 STL 분석: 32x32 카메라 창면의 중심·법선 추출.

README_wristcam.md 절차(방법 C, STL 폴백):
- 면적 ≈ 1024(=32x32)인 평면 면이 카메라 면.
- 광학중심 = 창중심 + 18mm × (보는 방향 법선).
좌표는 마운트 자체(디자인 원점) 기준. (손목링크 변환은 별도 필요)
"""

import sys
from collections import defaultdict
import numpy as np


def parse_ascii_stl(path):
    normals, tris = [], []
    cur = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == "facet":
                normals.append([float(s[2]), float(s[3]), float(s[4])])
                cur = []
            elif s[0] == "vertex":
                cur.append([float(s[1]), float(s[2]), float(s[3])])
                if len(cur) == 3:
                    tris.append(cur)
    return np.array(normals), np.array(tris)  # (F,3),(F,3,3)


def tri_area(t):
    return 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\user\Downloads\Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl"
    normals, tris = parse_ascii_stl(path)
    F = len(tris)
    allv = tris.reshape(-1, 3)
    lo, hi = allv.min(0), allv.max(0)
    print(f"facets={F}  bbox(min)={np.round(lo,2)}  bbox(max)={np.round(hi,2)}  "
          f"size={np.round(hi-lo,2)}")

    # 평면 클러스터: (양자화 법선, 평면거리 d) 별로 면적·면적가중 중심 누적
    clusters = defaultdict(lambda: {"area": 0.0, "csum": np.zeros(3), "n": np.zeros(3), "cnt": 0})
    for i in range(F):
        n = normals[i]
        nn = n / (np.linalg.norm(n) + 1e-12)
        d = float(nn @ tris[i, 0])
        key = (round(nn[0], 2), round(nn[1], 2), round(nn[2], 2), round(d, 1))
        a = tri_area(tris[i])
        c = tris[i].mean(0)
        cl = clusters[key]
        cl["area"] += a
        cl["csum"] += a * c
        cl["n"] += nn
        cl["cnt"] += 1

    rows = []
    for key, cl in clusters.items():
        if cl["area"] < 1e-6:
            continue
        center = cl["csum"] / cl["area"]
        nrm = cl["n"] / (np.linalg.norm(cl["n"]) + 1e-12)
        rows.append((cl["area"], center, nrm, cl["cnt"]))
    rows.sort(reverse=True, key=lambda r: r[0])

    print("\n[가장 큰 평면 면 12개]  area | center(mm) | normal")
    for a, c, n, cnt in rows[:12]:
        print(f"  area={a:7.1f}  c={np.round(c,2)}  n={np.round(n,3)}  facets={cnt}")

    print("\n[면적 ~1024(32x32) 후보: 800~1300]")
    for a, c, n, cnt in rows:
        if 800 <= a <= 1300:
            print(f"  *area={a:7.1f}  center={np.round(c,3)}  normal={np.round(n,3)}  facets={cnt}")
            oc = c + 18.0 * n
            print(f"     → 광학중심(창중심+18mm·n) = {np.round(oc,3)}  시선 = {np.round(n,3)}")


if __name__ == "__main__":
    main()

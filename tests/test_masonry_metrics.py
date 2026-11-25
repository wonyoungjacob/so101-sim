"""조적 검측식 오차(envs/masonry_metrics.py)가 현장 측정 방식대로 나오는지 확인한다.

렌더링·시뮬레이션 없이 계산만 검사한다.
"""

import numpy as np
import pytest

import config as C
from envs.masonry_metrics import masonry_error

HALF = np.array(C.BLOCK_HALF)  # (45, 28.5, 15) mm
REF = np.array([0.1, 0.2, C.BLOCK_Z])
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def yaw_quat(deg):
    a = np.radians(deg) / 2
    return (np.cos(a), 0.0, 0.0, np.sin(a))


def test_perfect_placement_is_zero():
    m = masonry_error(REF, IDENTITY, REF, HALF)
    for key in ("longitudinal", "lateral", "level", "plumb", "gauge", "max_corner"):
        assert m[key] == pytest.approx(0.0, abs=1e-12)
    assert m["yaw_deg"] == pytest.approx(0.0, abs=1e-9)
    assert m["tilt_deg"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("axis, key", [(0, "longitudinal"), (1, "lateral"), (2, "level")])
def test_pure_translation_reads_on_matching_tool(axis, key):
    shift = np.zeros(3)
    shift[axis] = 0.002  # 2 mm
    m = masonry_error(REF + shift, IDENTITY, REF, HALF)
    assert m[key] == pytest.approx(0.002)
    assert m["gauge"] == pytest.approx(0.002)
    assert m["max_corner"] == pytest.approx(0.002)


def test_yaw_is_caught_at_block_ends_even_when_center_matches():
    """중심이 정확해도 회전이 있으면 면 끝에서 편차가 잡혀야 한다(중심거리 지표와의 차이)."""
    deg = 2.0
    m = masonry_error(REF, yaw_quat(deg), REF, HALF)
    t = np.radians(deg)
    hx, hy = HALF[0], HALF[1]
    # 꼭짓점 (hx, hy)가 회전하면서 생기는 축별 최대 편차
    expected_lat = hx * np.sin(t) + hy * (1 - np.cos(t))
    assert m["yaw_deg"] == pytest.approx(deg)
    assert m["lateral"] == pytest.approx(expected_lat, rel=1e-6)
    assert m["gauge"] > C.SUCCESS_THRESH * 0.5  # 2°만 돌아도 3 mm 기준의 절반 이상
    assert m["level"] == pytest.approx(0.0, abs=1e-12)


def test_gauge_is_max_of_tool_readings():
    m = masonry_error(REF + np.array([0.001, -0.0025, 0.0005]), yaw_quat(0.5), REF, HALF)
    assert m["gauge"] == pytest.approx(max(m["longitudinal"], m["lateral"], m["level"]))

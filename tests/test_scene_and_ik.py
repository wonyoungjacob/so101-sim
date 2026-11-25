"""MJCF 장면 로드와 DLS IK를 확인한다. 렌더러를 만들지 않으므로 GPU·디스플레이 없이 돈다."""

import os

import mujoco
import numpy as np
import pytest

import config as C
from envs.ik import arm_dof_ids, dls_ik


@pytest.fixture(scope="module")
def model():
    # MuJoCo는 Windows의 비ASCII 절대경로를 못 읽으므로 저장소 루트 기준 상대경로로 로드한다.
    os.chdir(C.ROOT)
    return mujoco.MjModel.from_xml_path(C.SCENE_REL)


def test_scene_has_expected_joints_cameras_and_sites(model):
    for j in C.ARM_JOINTS + C.BLOCK_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0, j
    for cam in C.CAM_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0, cam
    for s in C.TARGET_SITES + [C.GRASP_SITE]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) >= 0, s
    assert model.nu == C.N_ACT


def test_targets_match_config(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, pos in zip(C.TARGET_SITES, C.TARGET_POS):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        np.testing.assert_allclose(data.site_xpos[sid][:2], pos[:2], atol=1e-6)


def test_ik_reaches_point_above_target1(model):
    data = mujoco.MjData(model)
    arm = C.ARM_JOINTS[:5]  # Jaw 제외
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm]
    data.qpos[qadr] = C.HOME_QPOS[:5]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, C.GRASP_SITE)
    goal = np.array(C.TARGET_POS[0]) + np.array([0.0, 0.0, 0.05])  # target1 위 5 cm

    err = dls_ik(model, data, sid, arm_dof_ids(model, arm), goal, iters=400)

    assert err < 1e-3  # 1 mm 이내

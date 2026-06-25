"""SO-101용 간단 DLS(damped least-squares) IK.

grasp_site의 목표 위치(+선택적 자세)를 만족하도록 팔 5축을 푼다(Jaw 제외).
position 제어 환경의 웨이포인트 관절각을 사전 계산하는 용도(키네마틱).
"""

from __future__ import annotations

import mujoco
import numpy as np


def arm_dof_ids(model, joint_names) -> np.ndarray:
    return np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
         for j in joint_names]
    )


def dls_ik(
    model,
    data,
    site_id: int,
    dof_ids: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    approach_local: np.ndarray | None = None,
    approach_world: np.ndarray | None = None,
    iters: int = 200,
    damping: float = 0.05,
    pos_tol: float = 1e-4,
    step: float = 0.5,
    ori_weight: float = 1.0,
) -> float:
    """data.qpos의 dof_ids 관절을 갱신해 site를 target_pos로 이동.

    자세 제어 모드(택1):
    - target_quat: 완전한 자세 정렬(6제약).
    - approach_local + approach_world: site의 approach_local 축만 월드 approach_world
      방향에 정렬(yaw 자유, 5제약). top-down 그랩에 적합.
    둘 다 None이면 위치만(3제약).

    반환: 최종 위치 오차(m).
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    # dof -> 해당 관절 qpos 주소 (hinge 1-dof 가정)
    qadr = np.array(
        [model.jnt_qposadr[np.where(model.jnt_dofadr == d)[0][0]] for d in dof_ids]
    )
    lo = np.array([model.jnt_range[np.where(model.jnt_dofadr == d)[0][0], 0] for d in dof_ids])
    hi = np.array([model.jnt_range[np.where(model.jnt_dofadr == d)[0][0], 1] for d in dof_ids])

    use_quat = target_quat is not None
    use_axis = approach_local is not None and approach_world is not None
    use_ori = use_quat or use_axis
    aw = None if approach_world is None else approach_world / np.linalg.norm(approach_world)
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[site_id]
        err_p = target_pos - pos
        R = data.site_xmat[site_id].reshape(3, 3)

        err_r = None
        if use_quat:
            cur_q = np.zeros(4)
            mujoco.mju_mat2Quat(cur_q, data.site_xmat[site_id])
            dq_quat = np.zeros(4)
            neg = np.zeros(4)
            mujoco.mju_negQuat(neg, cur_q)
            mujoco.mju_mulQuat(dq_quat, target_quat, neg)
            err_r = np.zeros(3)
            mujoco.mju_quat2Vel(err_r, dq_quat, 1.0)
        elif use_axis:
            # 현재 approach 축(월드) → 목표 방향으로 정렬. err = a × d (yaw 자유).
            a = R @ approach_local
            a = a / np.linalg.norm(a)
            err_r = np.cross(a, aw)

        # 위치우선 제어: ori_weight<1이면 자세제약 영향을 줄여 과제약(5-DOF) 시 위치를 살림.
        if use_ori:
            err = np.concatenate([err_p, ori_weight * err_r])
        else:
            err = err_p

        if np.linalg.norm(err_p) < pos_tol and (not use_ori or np.linalg.norm(err_r) < 1e-3):
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = (jacp[:, dof_ids] if not use_ori
             else np.vstack([jacp[:, dof_ids], ori_weight * jacr[:, dof_ids]]))
        # DLS: dq = J^T (JJ^T + λ²I)^-1 err
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(JJt.shape[0]), err)
        q = data.qpos[qadr] + step * dq
        data.qpos[qadr] = np.clip(q, lo, hi)

    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target_pos - data.site_xpos[site_id]))

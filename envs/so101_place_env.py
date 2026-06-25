"""SO-101 정밀 placing 환경 (gymnasium).

- 액션: 관절 절대각 6-D (Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw), position 제어.
- 관측: 앞 카메라 2장(RGB) + 관절각(state, 6).
- 물체: 콘크리트 블록 2개(block_i → target_i 매핑). 성공 = 모든 블록이 대응 목표 3mm 이내.
- 시작 배치: reset 때 배치 A/B를 50/50 랜덤(그 외엔 결정론적). options={"arrangement":"A"|"B"}로 강제 가능.

Windows 비ASCII 경로 문제를 피하려고 import 시 ROOT로 chdir 후 상대경로로 모델을 로드한다.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config as C
from envs.masonry_metrics import masonry_error

os.chdir(C.ROOT)


class SO101PlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_size: tuple[int, int] | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(C.SCENE_REL)
        self.data = mujoco.MjData(self.model)

        h, w = render_size or (C.CAM_H, C.CAM_W)
        self.cam_h, self.cam_w = h, w
        self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        # 관측 렌더용 시각 옵션: site 마커(빨간 grasp_site·성공 site 등)를 숨긴다.
        # 실제 카메라엔 없는 디버그 마커라 정책 입력에서 제외(IK는 site_xpos를 그대로 사용).
        self._vopt = mujoco.MjvOption()
        self._vopt.sitegroup[:] = 0

        m = self.model
        # 관절 qpos 주소 / 액추에이터 id 캐시 (config 순서와 동일)
        self._arm_qadr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in C.ARM_JOINTS]
        )
        # 블록 free-joint qpos 주소 / body id (리스트)
        self._block_qadr = [
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in C.BLOCK_JOINTS
        ]
        self._block_bid = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in C.BLOCKS
        ]
        self._target_sid = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in C.TARGET_SITES
        ]
        self._grasp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, C.GRASP_SITE)

        # 액션 공간 = 액추에이터 ctrl 범위 (position 제어 → 관절 절대각)
        low = m.actuator_ctrlrange[:, 0].astype(np.float32)
        high = m.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "front_left": spaces.Box(0, 255, (h, w, 3), np.uint8),
                "front_right": spaces.Box(0, 255, (h, w, 3), np.uint8),
                "wrist": spaces.Box(0, 255, (h, w, 3), np.uint8),
                "state": spaces.Box(-np.inf, np.inf, (C.N_ACT,), np.float32),
            }
        )
        self._step = 0
        self.arrangement = None  # 마지막 reset에서 사용한 배치("A"/"B")

    # --- 내부 유틸 ---
    def _render_cam(self, cam_name: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=cam_name,
                                    scene_option=self._vopt)
        return self._renderer.render()

    def _eef_pose(self) -> np.ndarray:
        """grasp_site의 world pose [x,y,z, qw,qx,qy,qz]."""
        pos = self.data.site_xpos[self._grasp_sid].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._grasp_sid])
        return np.concatenate([pos, quat])

    def _block_pos(self, i: int) -> np.ndarray:
        return self.data.xpos[self._block_bid[i]].copy()

    def _block_quat(self, i: int) -> np.ndarray:
        """블록 자세 쿼터니언 [w,x,y,z] (조적식 오차의 회전 성분에 필요)."""
        adr = self._block_qadr[i]
        return self.data.qpos[adr + 3:adr + 7].copy()

    def _target_pos(self, i: int) -> np.ndarray:
        return self.data.site_xpos[self._target_sid[i]].copy()

    def _get_obs(self) -> dict:
        return {
            "front_left": self._render_cam("cam_front_left"),
            "front_right": self._render_cam("cam_front_right"),
            "wrist": self._render_cam("cam_wrist"),
            "state": self.data.qpos[self._arm_qadr].astype(np.float32),
        }

    def _info(self) -> dict:
        n = len(self._block_bid)
        # 조적 검측식 오차(수평실/레벨/직선자/다림추 모사). 기준자세=목표중심·연직·코스축 정렬.
        masonry = [
            masonry_error(self._block_pos(i), self._block_quat(i),
                          self._target_pos(i), C.BLOCK_HALF)
            for i in range(n)
        ]
        # 합격판정 기준 거리 = gauge(어떤 자로도 잡히는 최악 수직편차). 중심거리가 아님.
        dists = [m["gauge"] for m in masonry]
        return {
            "eef_pose": self._eef_pose(),
            "block_pos": [self._block_pos(i) for i in range(n)],
            "block_quat": [self._block_quat(i) for i in range(n)],
            "target_pos": [self._target_pos(i) for i in range(n)],
            "dists": dists,            # = masonry gauge (조적식 governing 오차)
            "masonry": masonry,        # 도구별 분해 reading(longitudinal/lateral/level/...)
            "arrangement": self.arrangement,
            # 합격 = 블록 i가 대응 target i의 per-target 임계값 이내(target1 3mm, target2 5mm).
            "is_success": all(d < C.SUCCESS_THRESH_PER[i] for i, d in enumerate(dists)),
        }

    # --- 시작 배치 샘플링 ---
    @staticmethod
    def _yaw_quat(yaw: float) -> np.ndarray:
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    @staticmethod
    def _aabb_half(yaw: float) -> np.ndarray:
        """yaw 회전된 블록 footprint의 axis-aligned 반치수(x,y)."""
        hx, hy = C.BLOCK_HALF[0], C.BLOCK_HALF[1]
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        return np.array([hx * c + hy * s, hx * s + hy * c])

    def _sample_start(self):
        """배치 A/B 베이스 + 연속 jitter(위치+yaw). 겹침/리치는 rejection.

        반환: 블록별 (x,y,z,qw,qx,qy,qz) 리스트. arrangement는 'A+'/'B+'로 표기.
        """
        rng = self.np_random
        base = "A" if rng.random() < 0.5 else "B"
        base_poses = C.START_ARRANGEMENTS[base]
        for _ in range(C.RESAMPLE_TRIES):
            cand, aabbs, ok = [], [], True
            for (x, y, z, qw, qx, qy, qz) in base_poses:
                base_yaw = 2 * np.arctan2(qz, qw)  # Z회전만 가정
                yaw = base_yaw + rng.uniform(-C.YAW_JITTER, C.YAW_JITTER)
                px = x + rng.uniform(-C.POS_JITTER, C.POS_JITTER)
                py = y + rng.uniform(-C.POS_JITTER, C.POS_JITTER)
                if np.hypot(px, py) > C.REACH_RADIUS_MAX:
                    ok = False
                    break
                cand.append((px, py, z, *self._yaw_quat(yaw)))
                aabbs.append((np.array([px, py]), self._aabb_half(yaw)))
            if not ok:
                continue
            # 두 블록 AABB 비겹침 체크
            (c0, h0), (c1, h1) = aabbs[0], aabbs[1]
            d = np.abs(c0 - c1)
            if np.all(d < (h0 + h1 + C.BLOCK_AABB_GAP)):  # 겹침
                continue
            return base + "+", cand
        # 재시도 실패 시 베이스 그대로
        return base, list(base_poses)

    # --- gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # 팔: home 자세, ctrl도 동일 위치로(가만히 유지)
        self.data.qpos[self._arm_qadr] = C.HOME_QPOS
        self.data.ctrl[:] = C.HOME_QPOS

        # 시작 배치: options 지정(A/B) > 연속 랜덤(RANDOMIZE_START) > A/B 50/50
        opt_arr = (options or {}).get("arrangement")
        if opt_arr in C.START_ARRANGEMENTS:
            self.arrangement = opt_arr
            poses = C.START_ARRANGEMENTS[opt_arr]
        elif C.RANDOMIZE_START:
            self.arrangement, poses = self._sample_start()
        else:
            self.arrangement = "A" if self.np_random.random() < 0.5 else "B"
            poses = C.START_ARRANGEMENTS[self.arrangement]
        for adr, pose in zip(self._block_qadr, poses):
            self.data.qpos[adr:adr + 7] = pose

        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._get_obs(), self._info()

    def step(self, action):
        action = np.clip(
            np.asarray(action, np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.data.ctrl[:] = action
        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        info = self._info()
        reward = -float(np.sum(info["dists"]))
        terminated = bool(info["is_success"])
        truncated = self._step >= C.MAX_STEPS
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return self._render_cam("cam_front_left")

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None

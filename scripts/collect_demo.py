"""M2: 스크립트 전문가(IK 기반)로 정밀 placing 시연을 생성·기록.

새 씬(블록 2개, 90×57×30mm, 로봇 +Y, 배치 A/B 50/50)용 전문가.
- 리셋된 *실제* 블록 자세에서 매 에피소드 IK를 풀어 배치 A/B 모두 대응.
- 집기: 블록 짧은변(57mm)을 가로질러 위에서 파지(full-quat 정렬).
- 놓기: target에 90→X 방향으로 맞춰 놓음(배치 B는 이송 중 90° 회전).
- block1→target1, block2→target2 순차.

각 스텝마다 (관측 2캠+state, 액션 6, eef pose, 블록/목표/거리)를 기록.
기본은 1 에피소드 진단. --episodes N --save 로 다량 수집(out/episodes/ep_XXXX.npz).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from envs.so101_place_env import SO101PlaceEnv  # noqa: E402
from envs.ik import arm_dof_ids, dls_ik  # noqa: E402
from envs.masonry_metrics import masonry_error  # noqa: E402

ARM5 = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
WORLD_UP = np.array([0.0, 0.0, 1.0])

OPEN_JAW = 0.8     # gap ≈ 80mm (57mm 블록보다 넉넉히 벌림)
CLOSE_JAW = 0.30   # gap ≈ 41mm (<57mm → 블록 압착)
RELEASE_JAW = 0.62  # gap ≈ 65mm (57mm 블록 막 빠짐). 놓기 후 최소폭 개방으로 끌림/팝 최소화
GRASP_Z = 0.016    # 그랩 시 grasp_site(손끝) 높이(블록 중심 0.015 부근)
HOVER = 0.07       # 접근/이송 시 위로 띄우는 높이
# 그리퍼는 single-acting: grasp_site가 고정 jaw 쪽(+local X)에 치우쳐 있고 moving jaw만
# 벌어진다. 블록이 두 jaw '사이'에 오도록 grasp 목표를 grip축(+) 방향으로 이만큼 민다.
GRIP_OFFSET = 0.025
# 놓기 오프셋. GRIP_OFFSET과 같게 두는 게 현재 가장 안정적(블록별 파지 안착 편차가 있어
# 상수 보정으로 sub-3mm까지 못 잡음 → 정밀 튜닝은 후속). block1 2~4mm, block2 3~7mm 수준.
PLACE_OFFSET = 0.025


def make_grasp_quat(grip_axis_world):
    """top-down 그랩용 target_quat.

    grasp_site 로컬축 매핑: 로컬 X = 손가락 벌어지는 축, 로컬 -Y = 접근(아래).
    → 로컬 X → grip_axis(수평), 로컬 Y → 월드 위(+Z)로 정렬.
    """
    x = np.array(grip_axis_world, float)
    x[2] = 0.0
    x /= np.linalg.norm(x)
    y = WORLD_UP
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])  # 열 = 로컬축의 월드 이미지(로컬→월드 회전)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _solve(m, d, gid, dof, pos, quat, seed, ori_weight=1.0):
    d.qpos[:6] = list(seed) + [0.0]
    err = dls_ik(m, d, gid, dof, np.asarray(pos, float),
                 target_quat=quat, iters=500, damping=0.02, ori_weight=ori_weight)
    return d.qpos[:5].copy(), err


def _block_short_axis(blk_quat):
    """블록 짧은 수평변(57mm=블록 로컬 Y) 방향의 월드 벡터."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(blk_quat, float))
    return R.reshape(3, 3) @ np.array([0.0, 1.0, 0.0])


def plan_from_state(env, verbose=False):
    """현재(reset된) env 상태의 블록·목표로 전문가 위상 리스트를 만든다.

    반환: [(name, arm5_target, jaw, steps), ...] (block1→target1, block2→target2).
    """
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = env.data.qpos  # 현재 블록/팔 자세 복사
    mujoco.mj_forward(m, d)
    gid = env._grasp_sid
    dof = arm_dof_ids(m, ARM5)

    # 놓기: 블록 90변 → 월드 X (grip축 월드 Y). 부호는 로봇쪽(−Y)으로 둬 reach 보존.
    q_place_quat = make_grasp_quat([0.0, -1.0, 0.0])
    seed = list(C.HOME_QPOS[:5])
    phases = []
    errs = []

    for i in range(len(C.BLOCKS)):
        blk = env._block_pos(i).copy()
        blk_quat = d.qpos[env._block_qadr[i] + 3:env._block_qadr[i] + 7].copy()
        tgt = np.array(C.TARGET_POS[i], float)
        grip = _block_short_axis(blk_quat)
        grip_h = grip.copy(); grip_h[2] = 0.0; grip_h /= np.linalg.norm(grip_h)
        # grip축 부호: grasp_site(=블록+grip*offset)가 로봇(원점)쪽으로 가도록 선택 → reach 보존.
        if np.dot(grip_h[:2], blk[:2]) > 0:
            grip_h = -grip_h
        q_pick_quat = make_grasp_quat(grip_h)
        # grasp_site는 블록 중심에서 grip축 방향으로 GRIP_OFFSET 떨어진 곳을 겨냥
        # (그래야 블록이 fixed/moving jaw 사이에 위치).
        pick_xy = blk[:2] + grip_h[:2] * GRIP_OFFSET
        place_xy = np.array([tgt[0], tgt[1]]) + np.array([0.0, -1.0]) * PLACE_OFFSET

        def at(pos, quat):
            nonlocal seed
            q, e = _solve(m, d, gid, dof, pos, quat, seed)
            seed = list(q)
            errs.append(e)
            return q

        q_pre = at([pick_xy[0], pick_xy[1], GRASP_Z + HOVER], q_pick_quat)
        q_grasp = at([pick_xy[0], pick_xy[1], GRASP_Z], q_pick_quat)
        q_lift = at([pick_xy[0], pick_xy[1], GRASP_Z + HOVER], q_pick_quat)
        q_trans = at([place_xy[0], place_xy[1], GRASP_Z + HOVER], q_place_quat)
        q_place = at([place_xy[0], place_xy[1], GRASP_Z], q_place_quat)
        phases += [
            (f"pregrasp{i+1}", q_pre, OPEN_JAW, 35),
            (f"descend{i+1}", q_grasp, OPEN_JAW, 60),
            (f"close{i+1}", q_grasp, CLOSE_JAW, 40),
            (f"lift{i+1}", q_lift, CLOSE_JAW, 45),
            (f"transport{i+1}", q_trans, CLOSE_JAW, 60),
            (f"place{i+1}", q_place, CLOSE_JAW, 60),
            (f"release{i+1}", q_place, OPEN_JAW, 22),
            (f"retreat{i+1}", q_trans, OPEN_JAW, 30),
        ]
    if verbose:
        print(f"[plan] arrangement={env.arrangement} IK pos-err(mm)="
              f"{[round(e*1000,1) for e in errs]}")
    return phases


def build_plan(env):
    """편의용: env를 리셋하고 그 상태의 plan을 반환(view.py·진단)."""
    env.reset()
    return plan_from_state(env)


# ---- 정밀 전문가(sub-3mm) : 측정 기반 placing + 폐루프 정렬 ----
# 핵심: 상수 오프셋 추측 대신 잡은 직후 블록↔grasp_site '실제' 상대자세 T_rel을 매번 측정해
#   원하는 블록자세(목표중심·연직·코스축 정렬)를 만들 grasp_site 목표를 역산한다.
#   놓기 전 폐루프로 (현재 블록오차 측정→재배치)를 반복해 gauge<place_tol까지 정렬한다.
ALIGN_Z = C.BLOCK_Z + 0.0020   # 정렬 중 블록을 바닥서 2mm 띄워 자유롭게 위치(접지 stick 회피)


def _qmul(a, b):
    r = np.zeros(4); mujoco.mju_mulQuat(r, np.asarray(a, float), np.asarray(b, float))
    return r


def _qconj(q):
    r = np.zeros(4); mujoco.mju_negQuat(r, np.asarray(q, float))
    return r


def _qrot(q, v):
    r = np.zeros(3); mujoco.mju_rotVecQuat(r, np.asarray(v, float), np.asarray(q, float))
    return r


def run_episode_precise(env, record=True, seed=None, verbose=False, place_ow=0.6):
    """노이즈 없는 전문가 1 에피소드 — **수직 1회 placing**(재안착/슬라이딩 없음).

    각 블록: grasp → 든 직후 측정한 그립 오프셋으로 목표 위로 이송 → **수직 1회 하강** 안착 →
    깨끗이 개방(고정 jaw 수직 disengage). **놓은 뒤 보정 없음**(놓고 옮기는 동작은 IL에 치명적).
    결과 오차 = 단발 placing 정확도 그대로. 채택은 collect_dataset의 게이트(≤10mm)로 거른다.
    """
    obs, info = env.reset(seed=seed)
    m = env.model
    scratch = mujoco.MjData(m)
    gid = env._grasp_sid
    dof = arm_dof_ids(m, ARM5)
    traj = {k: [] for k in
            ["front_left", "front_right", "wrist", "state", "action", "eef_pose",
             "block_pos", "block_quat", "target_pos", "dists"]}
    st = {"obs": obs, "info": info, "prev": np.array(C.HOME_QPOS[:5], float)}

    def drive(q_target, jaw, steps, jaw_from=None):
        # jaw_from 주면 jaw를 jaw_from→jaw로 천천히 램프(개방 시 클램프해제 팝 최소화).
        q_target = np.asarray(q_target, float)
        ramp = max(1, steps // 2)
        for i in range(steps):
            alpha = min(1.0, (i + 1) / ramp)
            arm_cmd = (1 - alpha) * st["prev"] + alpha * q_target
            jcmd = jaw if jaw_from is None else (
                jaw_from + (jaw - jaw_from) * min(1.0, (i + 1) / steps))
            action = np.concatenate([arm_cmd, [jcmd]]).astype(np.float32)
            if record:
                traj["front_left"].append(st["obs"]["front_left"])
                traj["front_right"].append(st["obs"]["front_right"])
                traj["wrist"].append(st["obs"]["wrist"])
                traj["state"].append(st["obs"]["state"])
                traj["action"].append(action)
            o, _r, _term, _trunc, info2 = env.step(action)
            st["obs"], st["info"] = o, info2
            if record:
                traj["eef_pose"].append(info2["eef_pose"])
                traj["block_pos"].append(np.asarray(info2["block_pos"]))
                traj["block_quat"].append(np.asarray(info2["block_quat"]))
                traj["target_pos"].append(np.asarray(info2["target_pos"]))
                traj["dists"].append(np.asarray(info2["dists"]))
        st["prev"] = q_target

    def ik(pos, quat, ow=1.0):
        q5, _ = _solve(m, scratch, gid, dof, np.asarray(pos, float), quat,
                       st["prev"], ori_weight=ow)
        return q5

    def grasp(i):
        """블록 i를 현재 자세에서 top-down으로 집어 든다(시작·재안착 공용)."""
        blk = env._block_pos(i).copy()
        blk_quat = env._block_quat(i).copy()
        grip = _block_short_axis(blk_quat); grip[2] = 0.0
        grip /= np.linalg.norm(grip)
        if np.dot(grip[:2], blk[:2]) > 0:   # grip축 부호: 원점쪽 → reach 보존
            grip = -grip
        q_pick = make_grasp_quat(grip)
        pick = blk[:2] + grip[:2] * GRIP_OFFSET
        drive(ik([pick[0], pick[1], GRASP_Z + HOVER], q_pick), OPEN_JAW, 35)
        drive(ik([pick[0], pick[1], GRASP_Z], q_pick), OPEN_JAW, 50)
        drive(st["prev"], CLOSE_JAW, 40)                                   # 압착
        drive(ik([pick[0], pick[1], GRASP_Z + HOVER], q_pick), CLOSE_JAW, 45)  # 들기

    def place(i):
        """든 블록을 target i 위로 이송 후 '수직 1회' 하강해 내려놓는다(보정·재안착 없음).

        든 직후 블록↔grasp_site 오프셋(p_rel, 로컬)을 측정해, 도달가능한 top-down 자세(q_place)로
        블록 중심이 목표에 오도록 grasp_site 목표를 역산 → hover 위 → 수직 1회 하강 → 깨끗이 개방.
        먼 reach에선 위치우선 IK(place_ow<1)로 위치를 살린다(접지가 수평을 잡아줌). 누름 없음.
        """
        tgt = np.array(C.TARGET_POS[i], float)
        OW = place_ow
        # feedforward 보정: 체계적 오프셋(xy·yaw)을 미리 빼서 조준(재안착 아님, 여전히 단발).
        comp = np.array(C.PLACE_COMP_XY[i]) / 1000.0
        yawc = np.radians(C.PLACE_COMP_YAW[i])
        c, s = np.cos(-yawc), np.sin(-yawc)
        gd = np.array([c * 0.0 - s * (-1.0), s * 0.0 + c * (-1.0)])  # [0,-1]을 -yawc 회전
        q_place = make_grasp_quat([gd[0], gd[1], 0.0])  # 그립축(블록 yaw 상쇄 반영) top-down
        # 든 직후 블록 중심을 grasp_site 로컬에서 측정(자세 무관 상수 오프셋)
        ep = env._eef_pose()
        p_rel = _qrot(_qconj(ep[3:]), env._block_pos(i) - ep[:3])
        # 블록 중심을 (목표xy − 보정, 안착높이)에 둘 grasp_site 위치 = p_b_star − R(q_place)·p_rel
        p_b_star = np.array([tgt[0] - comp[0], tgt[1] - comp[1], C.BLOCK_Z])
        g_target = p_b_star - _qrot(q_place, p_rel)
        # 이송(목표 위 hover) → 수직 1회 하강 안착 → 짧은 정착 → 점진 개방 → 수직 disengage → 후퇴
        drive(ik([g_target[0], g_target[1], g_target[2] + HOVER], q_place, OW), CLOSE_JAW, 70)
        drive(ik(g_target, q_place, OW), CLOSE_JAW, 55)        # 수직 1회 하강
        drive(st["prev"], CLOSE_JAW, 25)                        # 짧은 정착
        drive(st["prev"], OPEN_JAW, 55, jaw_from=CLOSE_JAW)     # 점진 개방
        ep = env._eef_pose()                                    # 고정 jaw 수직 분리
        drive(ik([ep[0], ep[1], ep[2] + 0.030], ep[3:], OW), OPEN_JAW, 50)
        ep = env._eef_pose()
        drive(ik([ep[0], ep[1], GRASP_Z + HOVER], ep[3:], OW), OPEN_JAW, 30)
        drive(st["prev"], OPEN_JAW, 20)                         # 블록 정착 대기(측정 안정)

    def gauge_mm(i):
        return masonry_error(env._block_pos(i), env._block_quat(i),
                             np.array(C.TARGET_POS[i], float), C.BLOCK_HALF)["gauge"] * 1000

    for i in range(len(C.BLOCKS)):
        grasp(i)
        place(i)                       # 수직 1회 placing — 놓은 뒤 보정 없음
        if verbose == "debug":
            print(f"   [blk{i}] gauge={gauge_mm(i):.2f}mm")

    if verbose:
        g = np.array(st["info"]["dists"]) * 1000
        print(f"[precise] arr={st['info']['arrangement']} gauge={np.round(g, 2)}mm "
              f"success={st['info']['is_success']}")
    return st["obs"], st["info"], {k: np.asarray(v) for k, v in traj.items()}


def run_episode(env, record=True, rng=None, noise=0.0, options=None, verbose=False):
    """전문가 1 에피소드 실행. reset 후 그 상태에서 plan을 만들어 수행."""
    obs, info = env.reset(options=options)
    plan = plan_from_state(env, verbose=verbose)
    prev_arm = np.array(C.HOME_QPOS[:5], float)
    traj = {k: [] for k in
            ["front_left", "front_right", "wrist", "state", "action", "eef_pose",
             "block_pos", "block_quat", "target_pos", "dists"]}
    for _name, q_arm, jaw, steps in plan:
        ramp = max(1, steps // 2)
        for i in range(steps):
            alpha = min(1.0, (i + 1) / ramp)
            arm_cmd = (1 - alpha) * prev_arm + alpha * q_arm
            if noise > 0 and rng is not None:
                arm_cmd = arm_cmd + rng.normal(0, noise, size=5)
            action = np.concatenate([arm_cmd, [jaw]]).astype(np.float32)
            if record:
                traj["front_left"].append(obs["front_left"])
                traj["front_right"].append(obs["front_right"])
                traj["wrist"].append(obs["wrist"])
                traj["state"].append(obs["state"])
                traj["action"].append(action)
            obs, r, term, trunc, info = env.step(action)
            if record:
                traj["eef_pose"].append(info["eef_pose"])
                traj["block_pos"].append(np.asarray(info["block_pos"]))
                traj["block_quat"].append(np.asarray(info["block_quat"]))
                traj["target_pos"].append(np.asarray(info["target_pos"]))
                traj["dists"].append(np.asarray(info["dists"]))
        prev_arm = q_arm
    return obs, info, {k: np.asarray(v) for k, v in traj.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--save", action="store_true", help="에피소드 npz 저장")
    ap.add_argument("--noise", type=float, default=0.0015,
                    help="팔 명령 가우시안 노이즈 std(rad). 시연 다양성용")
    ap.add_argument("--only-success", action="store_true", help="성공 에피소드만 저장")
    ap.add_argument("--arrangement", choices=["A", "B"], default=None,
                    help="배치 고정(기본: 50/50 랜덤)")
    args = ap.parse_args()

    env = SO101PlaceEnv()
    out_ep = ROOT / "out" / "episodes"
    out_ep.mkdir(parents=True, exist_ok=True)

    n_success = 0
    for ep in range(args.episodes):
        rng = np.random.default_rng(1000 + ep)
        ep_noise = 0.0 if ep == 0 else args.noise
        opts = {"arrangement": args.arrangement} if args.arrangement else None
        obs, info, traj = run_episode(
            env, record=args.save or ep == 0, rng=rng, noise=ep_noise,
            options=opts, verbose=(ep == 0))
        dists_mm = np.asarray(info["dists"]) * 1000
        ok = info["is_success"]
        n_success += int(ok)
        print(f"[ep {ep:04d}] arr={info['arrangement']} "
              f"per-block dist={np.round(dists_mm, 1)}mm success={ok} "
              f"steps={len(traj['action'])}")
        if ep == 0:
            Image.fromarray(obs["front_left"]).save(ROOT / "out" / "m2_final_left.png")
            Image.fromarray(obs["front_right"]).save(ROOT / "out" / "m2_final_right.png")
        if args.save and (ok or not args.only_success):
            np.savez_compressed(out_ep / f"ep_{ep:04d}.npz", **traj)

    print(f"[M2] success {n_success}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()

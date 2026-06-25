"""실제 카메라 캘리브레이션(ChArUco) → MuJoCo 카메라 pos/xyaxes/fovy 계산.

문서값: 각 카메라의 T_robot_cam(카메라프레임→robot프레임), intrinsic(fx,fy,cx,cy,640×480).
robot frame: +X 전방, +Y 좌, +Z 상.  sim(MuJoCo) world: +Y 전방, +X 우, +Z 상.
  → mujoco = (-robotY, robotX, robotZ)
OpenCV 카메라 프레임: +X 우, +Y 하, +Z 전방(시선).
MuJoCo 카메라 프레임: 시선 -Z, +X 우, +Y 상.  → xyaxes = [right(+X), up(=-down)].
출력 pos/xyaxes/fovy를 place_scene.xml의 카메라에 그대로 넣는다.
"""

import numpy as np

# --- 최종(클릭검증) extrinsic: T_robot_cam (3x4) ---
T_LEFT = np.array([
    [-0.708199, -0.474342,  0.522929, 0.019325],
    [-0.704992,  0.514945, -0.487665, 0.381782],
    [-0.037960, -0.714024, -0.699091, 0.467825],
])
T_RIGHT = np.array([
    [ 0.782823, -0.407924,  0.469879,  0.030757],
    [-0.618921, -0.588391,  0.520320, -0.335228],
    [ 0.064222, -0.698136, -0.713079,  0.459139],
])
# intrinsic (fy, image height) → vertical FOV
FY = {"left": 341.038770, "right": 332.974567}
IMG_H = 480


def robot_to_mujoco(v):
    """robot frame 벡터/점 → mujoco world (-Ry, Rx, Rz)."""
    x, y, z = v
    return np.array([-y, x, z])


def cam_xml(name, T, fy):
    R = T[:, :3]
    t = T[:, 3]
    right_cv = R[:, 0]    # OpenCV +X (우)
    down_cv = R[:, 1]     # OpenCV +Y (하)
    fwd_cv = R[:, 2]      # OpenCV +Z (시선)

    pos = robot_to_mujoco(t)
    right = robot_to_mujoco(right_cv)
    up = robot_to_mujoco(-down_cv)      # MuJoCo +Y = 위 = -down
    view = robot_to_mujoco(fwd_cv)      # 시선(확인용)

    fovy = np.degrees(2 * np.arctan((IMG_H / 2) / fy))
    # 직교성 확인
    ortho = float(np.dot(right, up))
    z_axis = np.cross(right, up)        # = -view 이어야
    print(f"[{name}]")
    print(f"  pos    = {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
    print(f"  xyaxes = {right[0]:.4f} {right[1]:.4f} {right[2]:.4f} "
          f"{up[0]:.4f} {up[1]:.4f} {up[2]:.4f}")
    print(f"  fovy   = {fovy:.2f}")
    print(f"  (check) right·up={ortho:+.4f}  view_dir={view.round(3)}  "
          f"-(x×y)={(-z_axis).round(3)}")


if __name__ == "__main__":
    cam_xml("cam_front_left", T_LEFT, FY["left"])
    cam_xml("cam_front_right", T_RIGHT, FY["right"])

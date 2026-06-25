"""M0 검증: SO-101(SO-ARM100 프록시) 모델 로드 + 오프스크린 렌더.

- assets/so101/scene.xml 을 로드한다(아직 place_scene.xml 이 없으면).
- `home` 키프레임을 적용하고 한 프레임을 오프스크린 렌더해 out/m0.png 로 저장.
- 관절/액추에이터 이름과 개수를 출력해 구조를 확인한다.

GPU 없는 Windows에서도 동작하도록 mujoco.Renderer(오프스크린)만 사용한다.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# Windows에서 경로에 한글 등 비ASCII가 있으면 MuJoCo의 C++ 파일 I/O가
# "Illegal byte sequence"로 실패한다. repo 루트로 chdir 후 *상대경로*로 로드하면
# fopen이 ASCII 상대경로 바이트만 받으므로 회피된다.
os.chdir(ROOT)

# place_scene.xml 이 생기면 그걸 우선 사용, 없으면 기본 scene.xml.
SCENE_REL = "assets/so101/place_scene.xml"
if not (ROOT / SCENE_REL).exists():
    SCENE_REL = "assets/so101/scene.xml"


def main() -> None:
    print(f"[M0] cwd={os.getcwd()}")
    print(f"[M0] scene(rel): {SCENE_REL}")
    model = mujoco.MjModel.from_xml_path(SCENE_REL)
    data = mujoco.MjData(model)

    # --- 구조 출력 ---
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    act_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    cam_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    ]
    print(f"[M0] joints ({model.njnt}): {joint_names}")
    print(f"[M0] actuators ({model.nu}): {act_names}")
    print(f"[M0] cameras ({model.ncam}): {cam_names}")
    print(f"[M0] nq={model.nq} nv={model.nv} nbody={model.nbody}")

    # --- home 키프레임 적용(있으면) ---
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("[M0] applied keyframe 'home'")
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # --- 오프스크린 렌더 ---
    OUT.mkdir(exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        # 카메라가 정의돼 있으면 첫 카메라로, 아니면 자유시점(-1)
        cam = cam_names[0] if cam_names and cam_names[0] else -1
        renderer.update_scene(data, camera=cam if isinstance(cam, str) else -1)
        rgb = renderer.render()

    png = OUT / "m0.png"
    Image.fromarray(rgb).save(png)
    print(f"[M0] rendered {rgb.shape} -> {png}")
    print(f"[M0] pixel mean={rgb.mean():.1f} (0이면 검은 화면=렌더 실패 의심)")
    print("[M0] OK")


if __name__ == "__main__":
    main()

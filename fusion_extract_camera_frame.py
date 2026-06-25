# -*- coding: utf-8 -*-
"""
fusion_extract_camera_frame.py

Fusion 360 안에서 실행하는 스크립트.
카메라가 들어가는 평면 면의 중심과 법선을 정밀하게 뽑는다.
거기에 렌즈 오프셋을 더해 카메라 광학중심을 계산한다.

실행 방법
1. Fusion 360 에서 UTILITIES > ADD-INS > Scripts and Add-Ins.
2. Scripts 탭에서 새 스크립트를 만들고 이 파일 내용을 붙여넣는다.
3. 카메라 마운트 디자인을 연 상태로 Run.
4. 안내가 뜨면 카메라가 들어가는 평면 면을 클릭한다.
5. 결과가 메시지로 뜨고 홈 폴더에 wristcam_fusion_frame.json 으로 저장된다.

주의
- Fusion API 내부 길이 단위는 cm 다. 이 스크립트가 mm 로 환산한다.
- 좌표와 법선은 디자인 원점 기준이다. 즉 마운트 자체 좌표계다.
  SO-101 손목 링크 기준으로 쓰려면 마운트를 손목에 조립한 상태에서
  측정하거나, 손목->마운트 변환을 따로 곱해야 한다.
- 법선 방향이 카메라가 보는 방향과 반대로 나오면 SIGN 을 -1.0 으로.
"""

import adsk.core
import adsk.fusion
import traceback
import json
import os

OFFSET_MM = 18.0   # 구멍 정중앙에서 카메라 광학중심까지 (집게 방향)
SIGN = 1.0         # 법선이 반대 방향이면 -1.0


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        sel = ui.selectEntity(
            "카메라가 들어가는 평면 면을 클릭하세요", "PlanarFaces")
        face = adsk.fusion.BRepFace.cast(sel.entity)

        plane = adsk.core.Plane.cast(face.geometry)
        n = plane.normal
        n.normalize()

        # 면 중심. centroid 가 없으면 바운딩 박스 중심으로 대체
        try:
            c = face.centroid
        except Exception:
            bb = face.boundingBox
            c = adsk.core.Point3D.create(
                (bb.minPoint.x + bb.maxPoint.x) / 2.0,
                (bb.minPoint.y + bb.maxPoint.y) / 2.0,
                (bb.minPoint.z + bb.maxPoint.z) / 2.0)

        # cm -> mm
        cx, cy, cz = c.x * 10.0, c.y * 10.0, c.z * 10.0
        nx, ny, nz = n.x * SIGN, n.y * SIGN, n.z * SIGN

        ox = cx + OFFSET_MM * nx
        oy = cy + OFFSET_MM * ny
        oz = cz + OFFSET_MM * nz

        data = {
            "units": "mm",
            "frame": "design origin (mount frame)",
            "hole_center": [round(cx, 4), round(cy, 4), round(cz, 4)],
            "view_normal": [round(nx, 6), round(ny, 6), round(nz, 6)],
            "lens_offset_mm": OFFSET_MM,
            "camera_optical_center": [round(ox, 4), round(oy, 4), round(oz, 4)],
        }
        text = json.dumps(data, indent=2, ensure_ascii=False)

        out = os.path.join(os.path.expanduser("~"),
                           "wristcam_fusion_frame.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)

        ui.messageBox(text + "\n\n저장됨\n" + out,
                      "카메라 프레임 (마운트 원점 기준)")

    except Exception:
        if ui:
            ui.messageBox("실패\n{}".format(traceback.format_exc()))

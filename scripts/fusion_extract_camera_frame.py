# Fusion 360 Script: wrist cam 창중심/법선/광학중심을 '손목(그리퍼) 링크 좌표계'로 추출
#
# 사용:
#  1) 마운트를 SO-101 손목/그리퍼 링크에 조립한 디자인을 연다.
#  2) UTILITIES > ADD-INS > Scripts and Add-Ins (Shift+S) > My Scripts에 새 스크립트로
#     만들고 이 내용을 붙여넣은 뒤 Run.
#  3) 안내대로 (a) 32x32 카메라 창면, (b) 기준 손목/그리퍼 컴포넌트를 차례로 클릭.
#     (선택) 손끝 꼭짓점 2개도 클릭하면 프레임 검증용 중점도 같이 계산.
#  4) 결과가 메시지박스 + 홈폴더 wristcam_fusion_frame.json 에 저장된다.
#  ※ 법선이 카메라가 보는 반대로 나오면 아래 SIGN을 -1.0으로 바꿔 다시 Run.
#
# 주의: Fusion 내부 길이 단위는 cm → 본 스크립트가 mm로 변환해 출력.

import adsk.core
import adsk.fusion
import traceback
import json
import os

LENS_OFFSET_MM = 18.0   # 창 중심에서 집게방향으로 카메라 광학중심까지(mm)
SIGN = 1.0              # 법선 방향 반대면 -1.0


def _to_mm(p):
    return [round(p.x * 10.0, 3), round(p.y * 10.0, 3), round(p.z * 10.0, 3)]


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('활성 Fusion 디자인이 없습니다.')
            return

        # (a) 카메라 창면 선택
        faceSel = ui.selectEntity('카메라가 들어가는 32x32 창면(평면)을 클릭', 'PlanarFaces')
        face = adsk.fusion.BRepFace.cast(faceSel.entity)
        plane = adsk.core.Plane.cast(face.geometry)

        # (b) 기준 손목/그리퍼 컴포넌트(occurrence) 선택
        occSel = ui.selectEntity('기준이 될 손목/그리퍼 링크 컴포넌트를 클릭', 'Occurrences')
        occ = adsk.fusion.Occurrence.cast(occSel.entity)

        # 루트(월드) 좌표에서 창중심 c, 법선끝점 c2 (법선은 두 점차로 안전하게)
        c = face.centroid
        n = plane.normal
        n.normalize()
        c2 = adsk.core.Point3D.create(c.x + SIGN * n.x,
                                      c.y + SIGN * n.y,
                                      c.z + SIGN * n.z)

        # 월드 -> 손목링크 로컬 변환 = inverse(occ.transform2)
        toLocal = occ.transform2.copy()
        toLocal.invert()
        c.transformBy(toLocal)
        c2.transformBy(toLocal)

        win = _to_mm(c)                       # 창중심 (링크좌표, mm)
        nx, ny, nz = (c2.x - c.x), (c2.y - c.y), (c2.z - c.z)
        ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
        nrm = [round(nx / ln, 5), round(ny / ln, 5), round(nz / ln, 5)]
        opt = [round(win[i] + LENS_OFFSET_MM * nrm[i], 3) for i in range(3)]

        out = {
            'units': 'mm',
            'frame': 'selected wrist/gripper link (occurrence) local',
            'window_center_mm': win,
            'view_normal_unit': nrm,
            'lens_offset_mm': LENS_OFFSET_MM,
            'camera_optical_center_mm': opt,
        }

        # (선택) 손끝 꼭짓점 2개 → 프레임 검증용 중점
        try:
            ans = ui.messageBox('손끝 검증점도 찍을까요?\n(예: 양쪽 손가락 끝 꼭짓점 2개를 차례로 클릭)',
                                'wrist cam frame', adsk.core.MessageBoxButtonTypes.YesNoButtonType)
            if ans == adsk.core.DialogResults.DialogYes:
                v1 = adsk.fusion.BRepVertex.cast(ui.selectEntity('손가락 끝 꼭짓점 1', 'Vertices').entity)
                v2 = adsk.fusion.BRepVertex.cast(ui.selectEntity('손가락 끝 꼭짓점 2', 'Vertices').entity)
                p1 = v1.geometry
                p2 = v2.geometry
                p1.transformBy(toLocal)
                p2.transformBy(toLocal)
                mid = adsk.core.Point3D.create((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
                out['fingertip1_mm'] = _to_mm(p1)
                out['fingertip2_mm'] = _to_mm(p2)
                out['fingertip_mid_mm'] = _to_mm(mid)
                out['note_fingertip'] = '내 MuJoCo 프레임에선 약 (0,-92,0)mm 근처여야 함'
        except Exception:
            pass  # 손끝 단계는 선택이므로 취소돼도 무시

        path = os.path.join(os.path.expanduser('~'), 'wristcam_fusion_frame.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        msg = ('[손목링크 좌표계, mm]\n'
               '창중심  = {}\n법선    = {}\n광학중심 = {}\n'.format(
                   out['window_center_mm'], out['view_normal_unit'],
                   out['camera_optical_center_mm']))
        if 'fingertip_mid_mm' in out:
            msg += '손끝중점 = {}  (≈(0,-92,0) 기대)\n'.format(out['fingertip_mid_mm'])
        msg += '\n저장: {}'.format(path)
        ui.messageBox(msg)

    except Exception:
        if ui:
            ui.messageBox('실패:\n{}'.format(traceback.format_exc()))

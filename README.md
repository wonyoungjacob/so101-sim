# SO-101 정밀 Placing — 최소 MuJoCo 모방학습 환경

건설 로봇팔 연구용. RoboManipBaselines 논문의 "환경/데이터셋/정책의 최소 구성요소만
직접 만들고 학습은 검증된 라이브러리(LeRobot) 재사용" 철학을 따른다.

**Task**: SO-101(5축+그리퍼)이 블록을 집어 목표 위치에 **3mm 이내**로 놓는다(결정론적,
추후 조적/랜덤화/RL로 확장). 학습=B200 Linux(LeRobot), 시현=GPU 없는 Windows(MuJoCo).

## 구성요소
```
assets/so101/        SO-ARM100 MJCF(=SO-101 프록시) + 메쉬
  so_arm100.xml      로봇(6 position actuator). grasp_site, kp=150 수정 적용
  place_scene.xml    로봇 + 바닥 + 블록(20mm) + 목표 site + front 카메라 2개
config.py            공통 상수(위치/카메라/3mm 임계/스텝)
envs/
  so101_place_env.py gymnasium 환경(액션=관절각6, 관측=2캠+관절각)
  ik.py              DLS IK(완전자세 / 수직축-yaw자유 모드)
scripts/
  m0_check.py        모델 로드 + 렌더 확인
  smoke_m1.py        환경 reset/step 검증
  collect_demo.py    IK 전문가로 시연 생성 → out/episodes/*.npz
  convert_to_lerobot.py  npz → LeRobotDataset (B200)
  rollout.py         학습 정책 롤아웃·시현 (lerobot 필요)
```

## 핵심 설계 결정 / 튜닝(중요)
- **경로 한글 회피**: MuJoCo는 비ASCII 절대경로에서 파일 I/O 실패 → 모든 스크립트가
  repo 루트로 `os.chdir` 후 **상대경로**로 모델 로드.
- **액션 = 관절 절대각 6**(실기 SO-101 서보 명령과 1:1, 5-DOF IK 회피). 데이터셋엔
  EEF pose(7)도 병행 저장 → 추후 `action_keys`만 바꿔 EEF/delta 실험 가능.
- **관측 = 앞 카메라 2장 + 관절각**(wrist 캠 미사용, 실셋업과 일치).
- **grasp_site는 손가락 끝쪽**(`pos="0 -0.092 0"`): 작은 블록을 손끝으로 집어야
  손끝-바닥 충돌 없이 파지 가능.
- **블록 20mm/20g + 서보 kp=150**: 소형 그리퍼로 안정 파지되는 조합(원본 25mm는 미끄러짐).
- **전문가 IK**: 그랩=완전자세(손가락 X축 정렬로 깨끗한 straddle), 놓기=수직축만
  정렬·yaw 자유(목표 위치 IK 0.1mm). place 목표 +2mm 보정으로 baseline 0.58mm.
- **시연 다양성**: 작업은 고정, 팔 명령에 가우시안 노이즈 std=0.0015rad → 100% 성공 +
  소량 다양성. (0.002 이상은 3mm 초과 빈발.)

## 실행

### Windows (시현/수집)
```bash
pip install -r requirements-win.txt
python scripts/m0_check.py                 # 렌더 확인 → out/m0.png
python scripts/smoke_m1.py                 # 환경 검증
python scripts/collect_demo.py --episodes 40 --save --only-success
```

### B200 (학습) — M3, M4
```bash
pip install -r requirements-server.txt
# out/episodes/ 를 서버로 복사 후
# M3: LeRobotDataset 변환
python scripts/convert_to_lerobot.py --repo-id <you>/so101_place --fps 10
# M4: ACT 학습 (lerobot 버전에 맞게 인자 조정)
lerobot-train --dataset.repo_id=<you>/so101_place --policy.type=act \
    --output_dir=outputs/act_so101 --batch_size=8 --steps=50000
# 롤아웃·시현 (헤드리스 렌더 가능 머신)
python scripts/rollout.py --ckpt outputs/act_so101/checkpoints/last/pretrained_model
```

## 검증 현황
- M0 렌더 OK / M1 환경 OK / M2 전문가 **noise0.0015에서 10/10 성공, baseline 0.58mm**.
- M3/M4 스크립트 작성 완료(서버에서 실행). rollout은 lerobot+torch 필요.

# RoadPush AI 설치·실행·배포 가이드

## 저장소와 Drive 번들의 역할

- Git 저장소: 현재 production Python package, 설정, schema, 축약 로직 문서,
  개발 이력 요약, requirements만 포함한다.
- Drive ZIP: Git 내용에 더해 모델, 최종 160장 입력, semantic cache, 과거
  runner/renderer, 전체 실험 보고서·이미지, frozen HTML review를 포함한다.
- 모델과 데이터의 재배포 라이선스는 `manifests/model_assets_manifest.json`
  항목별 메모를 확인해야 한다.

## 설치

Windows에서 `setup.bat`를 실행하고 다음 중 하나를 고른다.

1. CPU only — PyTorch CPU wheel을 설치한다.
2. GPU — PyTorch 2.7.1 CUDA 12.8 wheel을 설치한다. NVIDIA RTX 4060 Laptop
   GPU와 RTX 5070 Ti Laptop GPU를 대상으로 한 공통 선택이다. 드라이버는
   CUDA 12.8 runtime을 지원해야 한다.

설치기는 공통 의존성 뒤 현재 package를 editable mode로 설치하고
`import roadpush_ai`를 확인한다.

설치 후 production entry point는 다음과 같다.

- `roadpush-phase6 --help`: 사진/semantic 입력에서 Sidewalk·Road 후처리와
  surface 선택까지 실행한다. 모델 추론 의존성 전체가 import되므로 반드시
  `setup.bat` 설치가 끝난 환경에서 사용한다.
- `roadpush-phase7a --help`: accepted S1 경계 생성·검증을 실행한다.
- `roadpush-phase7b --help`: Phase 7-A handoff에서 연결 통행 폭을 계산한다.

각 실행은 기존 디렉터리를 덮어쓰지 않는 새 `--output` 경로를 사용한다.

## 새 사진과 실험 도구

Drive ZIP을 푼 뒤 새 입력은 `experiments/raw_data/<run-name>/`에 둔다.
`experiments/tools/registry.yaml`은 Phase별 accepted runner, renderer, 보고서,
review의 위치를 찾는 단일 인덱스다.

```powershell
python experiments/tools/run_phase_review.py --list
python experiments/tools/run_phase_review.py --phase phase7b --mode frozen --dry-run
```

현재 `frozen` mode는 검증된 HTML을 안전하게 복사한다. 과거 연구 runner는
Drive에 보존되지만 입력 계약이 Phase마다 달라 자동 `runner` mode는 고의로
잠겨 있다. 새 연산은 해당 Phase의 registry entry와 원 보고서를 확인하고,
반드시 새 빈 output 경로에서 실행한다. 이 제한은 잘못된 manifest를 다른
Phase runner에 자동 투입하는 것보다 안전하며, production package의 API와는
별개다.

## 최종 검증

```powershell
python experiments/tools/validate_review_bundle.py `
  --reviews-root experiments/reviews
```

배포 전 다음도 확인한다.

- `python -m compileall -q src`
- `python -c "import roadpush_ai"`
- JSON Schema 두 파일이 파싱되는지 확인
- Git 추적 목록에 `experiments`, `models`, `data`, `outputs`, `scripts`, 모델
  weight가 없는지 확인
- ZIP에는 위 Drive-only 자산이 존재하고 160-input manifest가 정확히
  160건인지 확인

최종 알고리즘과 백엔드 연동 계약은
[`PHASE7B_PRODUCTION_LOGIC.md`](PHASE7B_PRODUCTION_LOGIC.md)를 따른다.

# Anomaly TTA - Agent Handover Status Report

다음 에이전트가 이어서 작업을 원활하게 수행할 수 있도록 작성된 현황 인수인계 자료입니다.

## 1. 배경 및 목적 (Background & Objective)
- **목표**: BTTA(Boundary Test-Time Adaptation) 세팅에서 SVM Pseudo-labeling 성능 최적화
- **이슈 발견**: 기존 `tail 1%` (`--active-svm-tail-pseudo-label-fraction 0.01`) 옵션을 사용했을 때, 정상 데이터의 판단 정확도는 높아지나 SVM 경계가 지나치게 보수적으로 좁게(Tight) 형성되는 문제 발생. 특히 `test6` 등 어려운 분할(split)에서 adaptation에 활용될 pseudo-label 샘플 수가 극도로 부족해져 성능 저하 발생(Under-adaptation).
- **해결 과정**: 
  1. SVM Confidence Threshold를 낮춰보는 실험(0.15, 0.20)을 진행했으나 근본적인 해결이 되지 않음.
  2. 단순 퍼센트 컷(1%) 대신 **순수 SVM Decision Value(Margin) 기반의 Confidence 기준으로 샘플을 추출하는 기능(Uncapped)**을 구현. 그러나 0.15나 0.8 등 고정된 절대 Margin 값으로 자를 경우, 거의 모든 샘플(99%)이 선택되어 노이즈가 폭발하는 문제 발생. (SVM의 스케일 차이 때문)
- **최종 해결 방향**: SVM Confidence 값이 가장 높은 순서대로 줄을 세우되, 전체 스트림의 **최대 5%** 까지만 추출하도록 상한선(Cap)을 두는 **Capped Confidence 5% Tail** 전략으로 선회. 현재 이 조건으로 3개 시드 전체를 재평가 중.

## 2. 실험 공통 설정 (Experimental Setup)
- **적용 대상**: RobustAD, AeBAD
- **시드(Seed)**: 0, 1, 2 (평균 계산 시 이 세 시드 사용)
- **새로운 Tail 옵션**: 
  - `--active-svm-tail-pseudo-label-fraction 0.05` (5% 상한)
  - `--active-svm-tail-pseudo-label-mode svm_confidence`
  - `--active-svm-tail-confidence-margin 0.15`
  - `--active-svm-confidence-threshold 0.15`
- **TTA 스코어**: `adapted_ema`
- **배치 크기**: 32
- **파라미터 스코프**: `decoder_bn_only`
- **장비**: GPU 1

## 3. 현재 실행 현황 및 결과 위치 (Current Status)

| 조건 | GPU 위치 | 진행 상태 | 결과 저장소 경로 (OUT_ROOT) |
| :--- | :--- | :--- | :--- |
| **Capped 5% Confidence Tail (conf 0.15)** | GPU 1 | 🏃‍♂️ 진행 중 (Seed 0) | `outputs/seed012_btta_sptail0.05_tailconf0.15_conf0.15_bs32_decoder_bn_only_gpu1/` |
| **Capped 5% Confidence Tail (conf 0.15)** | GPU 0 | 🏃‍♂️ 진행 중 (Seed 1) | `outputs/seed012_btta_sptail0.05_tailconf0.15_conf0.15_bs32_decoder_bn_only_gpu1/` |

> [!WARNING]
> 유저님의 지시("앞으로 개념검증은 seed0으로만 돌려 지금거는 일단 실행했으니까 seed 0,1 정도만 보자")에 따라 GPU 0번에서 돌고 있던 Seed 2 스케줄을 취소했습니다.
> 현재 GPU 1에서는 Seed 0을, GPU 0에서는 Seed 1을 실행 중입니다.

## 4. 인수인계자 (Next Agent)의 다음 작업 (Next Steps)

1. **실험 완료 대기 및 모니터링**:
   - `ps aux | grep run_capped_sptail0.05` 명령어나 `nvidia-smi`를 통해 실험이 종료되는지 모니터링하세요. (또는 시스템이 백그라운드 태스크 완료 알림을 줄 때까지 대기)

2. **결과 집계 및 비교표 작성 (요청사항)**:
   - 위 OUT_ROOT에 생성된 `robustad_detailed.csv` 데이터를 집계 스크립트를 통해 정리하세요.
   - 기존의 `1% Score Tail` 방식과 현재의 `5% Capped Confidence Tail` 방식을 비교하세요.
   - **중요 출력 규칙 (User Rule)**: 표 출력 시 반드시 `$평균 \pm 표준편차$` 형태의 포맷을 맞춰서 출력해야 합니다. (예: `$51.90 \pm 0.66$`)

3. **심층 분석 수행 (요청사항)**:
   - `test6`의 성능이 이번 5% 변경으로 인해 얼마나 복구되었는지 확인하고, 여전히 문제가 있다면 Tail 선택 샘플 수(`active_tail_pseudo_label_count`)와 정확도를 로그에서 대조 분석해 보세요.

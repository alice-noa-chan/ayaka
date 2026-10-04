# v2 비추론이 v1 추론을 크게 넘기 위한 학습 가설

사용자 목표는 **v2 reasoning off가 v1 large + worked-steps route를 크게 넘는 것**이다.
frozen-native + Swift 보정만으로 예상한 공개 85–90%는 이 목표를 만족하지 않는다.
그 예상은 유지하고, 필요한 학습 가설과 성공 기준을 상향한다. 이 문서는 성능
달성 보고나 과금 실행 승인이 아니다. `main`은 공개 v1으로 보존한다.

## 비교 기준과 성공의 크기

- 기준 모델은 공개 **ayaka-large(12B)**와 frozen v1 worked-steps 정책을 포함한
  시스템이다. v1 direct나 실패한 E4B 200-step pilot으로 기준을 바꾸지 않는다.
- 같은 독립 test의 원본 입력·후보 순서·soft gold·평가 버전에서 v1 시스템과
  v2 off를 짝지어 비교한다. v1의 기존 정책은 gated이므로 모든 문항에서 생성이
  발생해야 하는 것은 아니다. 설정과 실제 경로·사용량·fallback을 모두 기록한다.
- 잠정 성공선은 Choice/Noul의 thresholded gold credit **+5%p 이상**,
  typed equal-type chance-corrected 점수 **+5점 이상**, case 단위 paired bootstrap의
  aggregate 차이 95% 구간 하한 **>0**이다. Score는 RPS와 nMAE가 악화되지 않아야 한다.
  유형·영한일·날짜/수치·정보 부족의 회귀도 별도로 공개한다. 평균 개선으로 숨기지 않는다.
- v2 off는 생성 토큰 0이며 온라인 계산기·외부 teacher 호출 없이 판정한다.
  사용자가 `on + high`를 요청할 때의 강제 실행 요구는 그대로 유지한다.
- 기존 공개231 결과는 v1 direct 192/231=83.1%, worked-steps 208/231=90.0%다.
  220/231=95.2%는 약 +5.2%p와 오답 23→11을 뜻하는 **목표 예시**다.
  공개 문항을 학습/보정/선택에 사용하거나 이를 독립 test의 성공 증거로 부르지 않는다.
  이 정확도 예시는 최신 JevBench typed/Speed/Cost composite나 공식 sealed 점수가 아니다.

## 학습 가설

frozen+policy는 강한 저비용 제어군으로 유지한다. 더 큰 정확도 향상은 **검증된
reasoned teacher의 최종 분포를 원본 입력의 direct student로 증류**하는 가설로
검증한다. 학생에게 풀이를 입력해 주거나 배포 시 풀이를 생성하게 만드는 방식은
이 arm의 비추론 목표를 검사하지 못한다.

[Distilling System 2 into System 1](https://arxiv.org/html/2407.06023v3)은
중간 출력 없이 최종 출력만 증류하는 방법을 평가했다. 일부 과제에서는 teacher보다
좋은 결과도 있었지만, 복잡한 수학 CoT에서는 성공하지 못했다. 따라서 Ayaka의
날짜·수치 계산에서 큰 개선을 보장하는 근거로 쓰지 않는다. 우리의 가설은
gold 검증, 유형별 확률 품질, 원본 입력과 heldout 규칙의 일반화를 함께 검사한다.

v1 teacher만 복제하면 v1의 오류도 복제할 수 있다. v1이 틀리는 train 사례는
독립적으로 검증한 gold 또는 검증된 더 강한 teacher를 사용한다. teacher가 없거나
도움이 없으면 gold-only 직접 판정으로 남긴다. 기존 정답 사례도 replay해 회귀를 억제한다.
검증 계산기는 자료 준비에만 사용하며 추론 서버에서 호출하지 않는다.

작은 판독 head만으로 backbone에 없는 계산 능력이 생긴다고 가정하지 않는다.
우선 실제 native output head를 유지한 direct LoRA + gold-only와
direct LoRA + gold-anchored teacher KL을 비교하는 구성을 준비한다. LoRA 크기,
학습 문항·길이·총 steps는 실측 처리량과 완주 비용을 확인한 뒤 고정한다.
Clef evidence head는 별도 절제군이며 자동으로 기본 학습 구조로 승격하지 않는다.

## 현재 구현된 CPU 준비 경로

`ayaka/training/direct_distillation.py`는 모델을 로드하지 않고, 저장된 teacher 관측과
원본 자료로 기존 Trainer용 `TrainItem`을 만든다.

1. train/router_train/dev/calibration/test 다섯 split의 provenance와 중복을 검사한다.
   teacher는 train 문항에만 연결한다. 원본 state·질문·후보 설명/순서·soft gold·lineage를
   fingerprint로 묶고 Noul의 false/true 순서를 맞춘다.
2. caller의 독립 gold verifier가 증거로 다시 계산한 분포와 저장된 정답이 일치해야 한다.
   verifier가 단순히 저장된 target을 읽는 구현이면 실제 검증이 아니다. 이 독립성은
   verifier 소스와 corpus에 대한 별도 검토가 필요하며 CPU 계약만으로 증명하지 않는다.
3. 생성 토큰 0·length 종료·fallback으로 기록된 관측, 잘못된 확률, 변경된 문항의
   teacher 관측을 거부한다. trace 본문은 이 계약에서 다시 검사하지 않는다.
   원본 입력이 길이 한도를 넘으면 잘라서 학습하지 않는다.
4. gold NLL이 개선되고 Brier가 악화되지 않으며 유형별 판정이 유지되는 teacher만
   사용한다. hard Choice/Noul teacher는 정답이어야 한다. Score RPS/nMAE 회귀도 거른다.
   거른 teacher 때문에 해당 gold 학습 사례 자체를 삭제하지 않는다.
5. student 입력에는 원본 state와 질문/선택지만 들어간다. trace CE 위치는 만들지 않는다.
   `verified_traces`와 이전 teacher metadata가 새 item에 섞이지 않도록 제거한다.

`LossWeights(gold_nll_with_teacher=True, distill=0.2)`는 모든 문항의 gold NLL을 유지하고
teacher가 있는 문항에 detached forward KL을 더한다. 0.2는 새 opt-in 준비 함수의
기본 보조 가중치이며 최적이라는 실측 주장은 아니다. `distill=0`은 gold-only 절제군이다.
기존 v1은 기본값 `gold_nll_with_teacher=False`로 기존 teacher-KL 대체 동작을 유지한다.
호출자는 준비 함수와 Trainer에 같은 loss weights를 넘겨야 한다.

준비 보고서는 split/content/token/teacher fingerprints, teacher 수용/거부 사유와
직접 학습 토큰 수를 기록한다. `promotable=false`, `execution_attested=false`를 유지한다.
선언된 hash는 backend 실행 증명이 아니며 정답 일치가 풀이의 모든 단계 검증도 아니다.

## 아직 필요한 작업과 비용 조건

- Claude의 독립 설계 판단 및 새 모듈 코드 리뷰. 요청은
  `.dev/codex-direct-goal-to-claude-20261004.md`에 남겼으며 아직 답신이 없다.
- 실제 사용할 corpus에 대한 독립 gold verifier와 decontamination, 기존 saved teacher의
  model/prompt/원본 입력 binding. 새 teacher를 만드는 비용도 전체 예산에 포함한다.
- 정확한 weights/tokenizer/readout/LoRA 설정에 연결한 학습 runner, checkpoint/resume,
  학습 뒤 calibration→dev 선택→설정 freeze→독립 test 순서.
- 실제 GPU의 logits/cache parity, 처리량·VRAM·serial latency, teacher 준비부터
  학습/test/회수까지 **잔액 안에서 완주**하는 사전 admission. R15 전체 시간 cap 문제는
  Claude 측 Swift 담당 범위이며 해결 확인 전 과금 실행하지 않는다.
- 첫 CPU tiny-LoRA optimizer step은 데이터/손실 연결 검사다. 실제 성능 향상이나
  전체 생산 학습의 완료 증거가 아니다. 낮은 예산으로 실행 가능한 완전한 workload를
  먼저 정하며, 중도 시간 종료를 기본 학습 계획으로 쓰지 않는다.

완료한 단위: `7daf9d6` gold NLL 보존, `974f95c` 직접 증류 준비.
관련 CPU 테스트 46 passed / 5.67s, Ruff lint/format 통과.
마지막 전체 CPU 회귀 검사: **1049 passed, 1 skipped / 142.91s**,
소스 변경 0. 영수증은 `.dev/codex-direct-distillation-full-20261004.json`에 있다.
whole-tree Ruff lint/format도 228 Python files에서 통과했다.
새로운 GPU/API inference와 학습된 checkpoint는 없다.

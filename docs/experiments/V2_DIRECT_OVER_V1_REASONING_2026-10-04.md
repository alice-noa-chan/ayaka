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
   새 `ayaka/data/direct_verification.py`는 지원하는 authored curriculum v3의 원본
   문서·질문·후보를 해석해 gold를 다시 계산한다. 저장된 target이나 trace, generator의
   계산 함수를 읽지 않는다. 지원하지 않는 문서 형식은 명시적으로 거부한다.
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
준비 item에는 `direct_distillation=True`가 붙는다. 실제 Trainer는 forward 전에
gold NLL 보존 설정, 양수 gold 가중치와 유한한 비음수 distillation 가중치를 검사한다.
trace/proposal/image 학습 필드가 들어온 item도 거부한다. 설정을 전달하지 않아
기존 v1의 KL 대체 경로로 되돌아가는 실수를 학습 시작 전에 검출한다.

준비 보고서는 split/content/token/teacher fingerprints, teacher 수용/거부 사유와
직접 학습 토큰 수를 기록한다. `promotable=false`, `execution_attested=false`를 유지한다.
선언된 hash는 backend 실행 증명이 아니며 정답 일치가 풀이의 모든 단계 검증도 아니다.

## 저장·재검증·학습 일정 연결

`ayaka/training/direct_bundle.py`는 CPU 전용 CLI다. 이미 로컬에 있는 고정 revision의
tokenizer와 config만 읽으며 모델 weights, optimizer, GPU 실행을 시작하지 않는다.

```bash
python -m ayaka.training.direct_bundle prepare \
  --corpus /path/to/five-split-jsonl-directory \
  --config /path/to/electra-config.json \
  --teacher-reads /path/to/saved-teacher-reads.json \
  --out /path/to/new-bundle-directory \
  --steps 3 --rows-per-step 32 --seed 20261004 --distill-weight 0.2
python -m ayaka.training.direct_bundle audit --bundle /path/to/new-bundle-directory
```

teacher가 없는 gold-only 제어군은 `--teacher-reads`를 생략하고 `--distill-weight 0`을
사용한다. 실제 대규모 학습의 steps를 위 예제의 3으로 고정하라는 뜻은 아니다.
`--mechanics-only`는 random tiny 모델 검사에만 사용한다.

- 원본 다섯 split, 저장된 teacher, 실제 `TrainItem`, recipe와 준비 보고서를 저장한다.
  파일 checksum, 전체 Ayaka Python 소스 hash, 실제 tokenizer serialization과 chat
  template을 묶는다. 기존 출력 디렉터리를 덮어쓰지 않는다.
- audit은 저장한 token/item을 신뢰해 불러오기만 하지 않는다. 원본 gold 검증,
  teacher 선별, tokenization과 유한한 전체 일정까지 다시 계산해 비교한다.
  item이나 보고서를 수정한 뒤 checksum만 새로 써도 불일치를 검출한다.
- train과 모든 heldout 문항의 **전체** 원본 입력을 실제 tokenizer로 검사한다.
  문맥 초과나 실제 vocab 밖의 token을 거부하며 잘라서 통과시키지 않는다.
  이 native LM arm은 label이 있는 최대 26개 후보를 지원한다. 그 이상은 별도
  pointer/hybrid 절제군이 필요하며 자동 fallback으로 학습 구조를 바꾸지 않는다.
- `training_batches(..., start_step=...)`는 고정된 step/rows/seed 일정과 재현 가능한
  whole-step 재개 위치를 제공한다. optimizer/checkpoint/RNG 상태 저장·복원까지
  구현되었다는 뜻은 아니다.
- independent gold 검증은 현재 authored 날짜·수치·규칙 자료의 10개 family에
  한정된다. 자연 문서의 judge/score 정답이나 teacher 풀이의 전 단계를 검증하는
  일반 verifier가 아니다. 검증 계산은 자료 준비에서만 실행한다.

`ayaka/training/direct_preflight.py`는 meta 장치에 실제 native text 구조와 LoRA를
구성한다. 실제 output head, 임베딩과의 weight tying, 각 projection의 LoRA 개수,
전체 파라미터와 native context를 검사한다. 선언한 target에 LoRA가 붙지 않거나
14B 제한을 넘으면 bundle을 만들지 않는다. 이 구조 기록을 저장하고 audit에서
재구성해 비교한다. 실제 weight bytes와 GPU cache/logits parity는 별도 검증이다.

## 실제 CPU 준비 결과와 범위

고정된 `google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, native LM readout과 LoRA rank 32/alpha 64를
사용해 로컬 tokenizer·config와 meta 구조로 실제 준비 묶음을 만들고 재검증했다.

| 항목 | 확인한 값 |
|---|---:|
| native text 파라미터 | 11,907,350,272 |
| 공식 full-checkpoint element 수 | 11,959,730,224 |
| 추가 adapter/decision 파라미터 | 145,042,438 |
| 보수적인 전체 파라미터 수 | **12,104,772,662** |
| native output head | 262,144 × 3,840, input과 tied, bias 없음 |
| LoRA 위치 | q/k/o/gate/up/down 각 48개, v 40개 |
| 준비 문항 | split별 96개, 총 480개 |
| 실제 train 일정 | 3 steps × 32 rows = 96 rows, 각 row 1회 |
| 일정의 원본 입력 토큰 | 15,250 |
| split별 최대 입력 길이 | 207–212 tokens |
| teacher 관측 / 수용 수 | **0 / 0** |
| 실제 optimizer steps / model forward | **0 / 0** |

이 묶음은 영어 authored mechanics 자료와 gold-only 제어군이다. production 품질 학습
corpus를 완성했다는 뜻이 아니며 독립된 자연 문항에서 성능을 측정하지 않았다.
96 train rows의 source lineage는 26개이므로 96개의 독립 규칙으로도 부르지 않는다.
`promotable=false`, `execution_attested=false`를 유지한다. 실제 GPU/API 사용은 없었다.
로컬 receipt는 `.dev/direct-native-cpu-20261004/receipt.json`이며 `.dev`는 Git 제외다.

## 아직 필요한 작업과 비용 조건

- Claude의 독립 설계 판단 및 새 모듈 코드 리뷰. 요청은
  `.dev/codex-direct-goal-to-claude-20261004.md`와
  `.dev/codex-direct-runner-to-claude-20261004.md`에 남겼다. Swift 후속 snapshot과
  기존 Clef head/objective 리뷰는 받았지만 이 direct 학습 가설/새 모듈 답신은 아직 없다.
- 실제 품질 학습 corpus의 준비와 decontamination, authored 형식 밖의 독립 gold
  검증, 실제 saved teacher의 model/prompt/원본 입력 및 실행 provenance 연결.
  새 teacher를 만드는 비용도 전체 예산에 포함한다.
- 정확한 weights/tokenizer/readout/LoRA 설정에 연결한 학습 runner, checkpoint/resume,
  학습 뒤 calibration→dev 선택→설정 freeze→독립 test 순서.
- 실제 GPU의 logits/cache parity, 처리량·VRAM·serial latency, teacher 준비부터
  학습/test/회수까지 **잔액 안에서 완주**하는 사전 admission. R15 전체 시간 cap 문제는
  Claude 측 Swift 담당 범위이며 해결 확인 전 과금 실행하지 않는다. 최신 Swift 소스의
  fake-clock CPU 재검사에서도 전체 60초 설정이 첫 priority의 work 600초/cleanup 630초로
  늘어나는 반례가 유지되었다. 신규 GPU/API/process 없이 확인한 결과다.
- 첫 CPU tiny-LoRA optimizer step은 데이터/손실 연결 검사다. 실제 성능 향상이나
  전체 생산 학습의 완료 증거가 아니다. 낮은 예산으로 실행 가능한 완전한 workload를
  먼저 정하며, 중도 시간 종료를 기본 학습 계획으로 쓰지 않는다.

완료한 단위: `7daf9d6` gold NLL 보존, `974f95c` 직접 증류 준비,
`2447831` Trainer 사전 guard, `4df88fc` 원본 문서 gold verifier,
`19f6c3e` immutable CPU bundle, `9f2a8dd` native meta 구조 검사,
`80c8eb3` 실제 구조/heldout 문맥과 bundle 연결. 단위별 코드/테스트 수정 → lint/format →
재검사 → 관련 테스트 → diff 검토 → 별도 커밋 순서로 검증했다.
마지막 전체 CPU 회귀 검사: **1190 passed, 1 skipped / 166.10s**,
소스 변경 0. 영수증은 `.dev/codex-direct-bundle-full-20261004.json`에 있다.
whole-tree Ruff lint/format도 234 Python files에서 통과했다.
새로운 GPU/API inference와 학습된 checkpoint는 없다.

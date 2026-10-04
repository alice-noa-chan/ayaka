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
  whole-step 재개 위치를 제공한다. 후속 `direct_state.py`와 `run_direct.py`는
  LoRA/AdamW/scheduler/RNG를 함께 저장하고 복원한다. CPU native tiny에서
  activation checkpointing on/off 모두 연속 실행과 정확히 일치했다.
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
| 준비 train 일정 | 3 steps × 32 rows = 96 rows, 각 row 1회 |
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

- Claude의 direct 설계·verifier/guard/bundle/meta 리뷰를 받았다. 세 arm(gold-only,
  teacher-final KL, frozen-base replay KL), 자연 정책 문서의 필요성과 frozen+policy
  제어군도 넘어야 한다는 조건에 합의했다. mixed-batch와 외부 corpus anchor 지적을
  반영했다. 새 kernel/state/runner/raw registry 코드의 추가 독립 리뷰는 요청 중이다.
- 실제 품질 학습 corpus의 준비와 decontamination, authored 형식 밖의 독립 gold
  검증, 실제 saved teacher의 model/prompt/원본 입력 및 실행 provenance 연결.
  새 teacher를 만드는 비용도 전체 예산에 포함한다.
- native runner, checkpoint/resume, calibration-only T fitting, 고정 dev 진단과
  export/reload parity, test 원문의 별도 보관은 구현했다. 실제 승격용 test 실행,
  matched arm의 dev 선택과 freeze, public composite 비교는 더 필요하다.
  현재 development bundle에는 test 원문이 없으며 opaque commitment만 있다.
- 실제 GPU의 logits/cache parity, 처리량·VRAM·serial latency, teacher 준비부터
  학습/test/회수까지 **잔액 안에서 완주**하는 사전 admission. 전체 11개 stage의
  prepaid estimator와 측정 후 재admission을 구현했다. 실제 provider 잔액·단가·과금
  시작 시각과 외부 회수/정리 supervisor는 별도로 검증해야 한다.
  Swift R15는 Claude의 `96b69aa` 이후 독립 fake-clock 재현에서 해결됐다.
  1분 full plan은 launch 전 거부, 120분 plan의 모든 deadline은 7200초 안에 유지된다.
  vLLM 설치/download가 Swift runner cap 밖이라는 제한은 전체 비용에 포함해야 한다.
- 첫 CPU tiny-LoRA optimizer step은 데이터/손실 연결 검사다. 실제 성능 향상이나
  전체 생산 학습의 완료 증거가 아니다. 낮은 예산으로 실행 가능한 완전한 workload를
  먼저 정하며, 중도 시간 종료를 기본 학습 계획으로 쓰지 않는다.

완료한 단위: `7daf9d6` gold NLL 보존, `974f95c` 직접 증류 준비,
`2447831` Trainer 사전 guard, `4df88fc` 원본 문서 gold verifier,
`19f6c3e` immutable CPU bundle, `9f2a8dd` native meta 구조 검사,
`80c8eb3` 실제 구조/heldout 문맥과 bundle 연결. 단위별 코드/테스트 수정 → lint/format →
재검사 → 관련 테스트 → diff 검토 → 별도 커밋 순서로 검증했다.
이전 전체 CPU 회귀 검사: **1190 passed, 1 skipped / 166.10s**,
소스 변경 0. 영수증은 `.dev/codex-direct-bundle-full-20261004.json`에 있다.
whole-tree Ruff lint/format도 234 Python files에서 통과했다.
새로운 GPU/API inference와 실제 pretrained 기반 학습 checkpoint는 없다.

## 후속 native 실행·최적화 상태

`optimization.py`는 실제 출력 head와 typed loss를 유지하면서 인스턴스별 kernel을
적용한다. 설치 누락/미지원 요청은 오류로 반환한다. Liger의 전체 vocab CE patch는
직접 판정 손실에 맞지 않아 사용하지 않는다. 실제 native Gemma4 unified norm의
weight 의미(offset=0)와 tanh GeGLU에 functional kernel을 적용하고 기존 PEFT
projection/parameter 객체를 유지한다. norm48개는 scale-free이므로 native를 유지한다.

실제 pinned 12B의 meta topology에서 FA2 지원 head 층40개, head_dim512의 native SDPA
층8개를 확인했다. FA2는 공식 지원 범위(head dimension ≤256)에만 적용하며 큰 head를
임의로 분할하지 않는다. [FA2 공식 지원 문서](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support),
[Liger functional API](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/functional.py).
CPU reference 함수 주입으로 실제 Trainer의 loss/prob/gradient parity와 rollback을
검사했다. **실제 CUDA FA2/Liger 실행이나 속도 향상을 측정한 것은 아니다.**

bundle은 현재 v5다. kernel 설정과 실제 topology, 원본 human file binding 및 direct
input encoding/serving recipe를 저장한다. 이전 준비 receipt는 보존하며 최신 소스로
다시 준비해야 한다.
실행 시 묶음 밖에 고정한 manifest SHA256을 요구해 전체 payload를 다시 생성한
다른 corpus가 내부 audit만으로 원래 묶음을 대체하지 못하게 한다.

`native_snapshot.py`는 캐시에 있는 고정 revision의 config/index/safetensors bytes와
header를 검사한다. strict offline loader는 누락된 text tensor의 random 초기화를
거부한다. 실제로 쓸 bytes를 묶는 것이며 publisher 원본의 별도 authenticity 증명은 아니다.
CPU random Gemma 33 shards의 원본 LM logits와 strict 재로딩 logits가 일치했다.

`run_direct.py`는 audit → native parity → backward/optimizer/IO profile → 전체 비용
재admission → 고정된 모든 step → calibration → 고정 dev 진단 → native export
→ offline reload 확률 일치 순서다. 부족한 비용으로 steps를 자동 줄이지 않는다.
실제 native profile/train은 `--execute`, CUDA, matching snapshot, 외부 bundle anchor,
이미 과금된 setup/download/teacher 시간과 quoted prepaid plan을 명시해야 한다.
CLI 자체가 cloud를 할당하거나 과금을 승인하지는 않는다.
test 원문은 개발 묶음에서 제거했고, 별도 holdout 저장소는 dev 선택 고정 후 연다.

`direct_natural.py`는 HelpSteer2 20,324 rows, CommonsenseQA 9,741 rows,
MASSIVE ko/ja 각각11,514 rows를 로컬에서 읽고 모두 변환했다(download0).
raw 인간 라벨/원본 입력/후보/lineage와 변환 gold를 대조하며 기존 source-group hash를
유지한다. HelpSteer ordinal encoding은 소수 평균도 보존하지만 실제 cached ratings의
소수 값은 0개였다. MASSIVE는 명시적인 balanced binary intent propositions이며
60-way Choice 성능으로 보고하지 않는다. rubric/schema를 공유하므로 원본 인간 라벨
검증과 문법의 독립 검증을 구분한다. **일반 rehearsal이며 자연 정책 문서 reasoning
자료와 독립 test 성공을 대신하지 않는다.**

`frozen_replay.py`는 LoRA를 실제로 비활성화하고 natural train 입력만 한 번 읽는다.
기본 native 분포를 정답/추론 teacher와 별도로 저장하며 `base_replay` coefficient는
기본0이다. 활성화해도 gold NLL과 teacher KL을 대체하지 않는다. saved read의 입력·
후보 순서·native bytes digest를 optimizer binding에 고정하고 resume에서 같은 원본
관측을 요구한다. 추가 base 모델과 매 step reference forward는 없다. coefficient의
최적값과 실제 회귀 방지 효과는 matched pilot에서 측정해야 한다.

후속 커밋: `52ccc99` mixed batch/rounding 회귀, `9b27cb2` kernels, `9008fa4`
kernel recipe binding, `4ba7529` optimizer continuation, `8b1ad28` strict native snapshot,
`19b71c2` whole prepaid plan, `d008103` direct pipeline, `e6d83e3` raw human gold,
`0d2624e` external corpus anchor, `3523e92` frozen-base replay. 각 단위는 코드/관련 테스트
→ Ruff fix → format → lint/format 재검사 → 관련 테스트 → diff review → 별도 commit으로 진행했다.

이전 통합: **1316 passed, 1 skipped / 189.12s**, source 변경0,
whole-tree Ruff lint/format **248 files** 통과.
receipt `.dev/codex-native-runner-kernels-full-20261004.json`은 before/after source hash와
실제 Git Bash 테스트 환경을 기록했다. tiny CPU의 optimizer/export 결과를 실제
pretrained 모델 성능으로 계산하지 않는다.

당시 실제 pinned12B meta/tokenizer control은
`.dev/direct-native-kernels-cpu-20261004/receipt.json`에 저장했다. 외부 manifest anchor:

```text
19fc8ffa49959df6332e18f866e64d21747e287e676951fa0fcd56998a2d8a5a
```

이 control은 v3 gold-only authored480문항/teacher0, planned96train rows/optimizer0이다.
FA2+Liger 설정과 실제 meta topology를 재구성했고 source/tokenizer/full workload audit이
일치했다. **현재 캐시에는 12B config/tokenizer만 있고 native weight shard가 없다.**
actual snapshot inspection은 ready=False로 기록했다. 본 품질 corpus, saved teacher,
GPU 런타임 확인이나 전체 cloud workflow 완료 증거가 아니다.

## Swift prior 일치·cache·holdout 후속

`faac5ba`, `efc4567`: `swift_evidence.py`는 Swift의 실제 parser/renderer/canonical
answer token을 사용한다. 전체 입력을 재토큰화하거나 자르지 않고 offset으로
state-only prefix와 질문 suffix를 나눈다. 경계를 가로지르는 BPE token은 suffix에
남긴다. 실제 renderer에서 선택지 장식을 도출해 `labeled`의 semantic key가 추가돼도
prior는 실제 서빙 입력과 같으며 description pooling은 원래 설명을 따른다.

실제 offline fast tokenizer와 random Gemma/Granite에서 native output bias/scaling을
포함한 HFReader 확률/centered logits가 4 prompt variants × 2 state formats ×
full/deep/shared cache에서 일치했다. 고정된 실제 12B tokenizer에서도 24 typed inputs의
전체 ids와 canonical answer ids가 일치했다. `.dev/codex-swift-evidence-labeled-native-inputs-20261004.json`
참조. **실제 pretrained 확률이나 GPU 성능을 측정한 것은 아니다.**

`83c2b07`: feature extraction의 `cache_strategy=copy_on_write`는 stock non-offloaded
DynamicCache의 attention layer 객체·counter만 복사하고 immutable prefix KV tensor를
공유한다. sliding window가 이미 이전 상태를 버렸어도 원본 branch를 보존한다.
실제 tiny Gemma/Granite의 full-row/deepcopy 결과와 일치하고 원본 cache는 변하지 않았다.
recurrent/custom/static/offloaded cache는 명시적으로 거부하며 기본 deepcopy는 유지한다.
branch가 복사하는 prefix KV bytes는 0이다. 이것은 total peak VRAM이나 속도 실측이 아니다.

`a69fe56`: `evidence_permutation.py`는 record/epoch별로 실제 프롬프트를 다시 렌더링한다.
순서는 gold를 보지 않고 정하며 soft gold와 Score ordinals를 semantic label에 맞게 옮긴다.
dev에서는 canonical/reverse/cyclic의 최대 3개 고정 view를 모두 비교해 probability range,
TV, argmax 변화, NLL/RPS 범위를 보고한다. cached tensor의 순열이나 best-view 선택으로
causal 순서 의존성을 숨기지 않는다. actual tiny backbone의 순서별 변동도 확인했다.

`cfc774f`: v4 development bundle은 train/router_train/dev/calibration만 포함한다.
prepare 단계에서 test gold와 전체 문맥을 검증한 뒤 원문은 별도 `OUT-holdout`에 저장한다
(`--holdout-out`으로 다른 격리 경로 지정 가능). development bundle에는 opaque test
state/content/source/translation/template/rule hashes와 counts/context receipt만 남는다.
학습용 업로드에는 development bundle만 포함하고 holdout 원문은 별도로 보관한다.

학습 audit와 runner는 private holdout 파일을 열지 않는다. 학습 runner의 자동 test 평가도
제거했으며 export는 `independent_test_required=True`, `promotable=False`다.
`open_holdout`은 외부에 고정한 holdout manifest·dev 선택 digest, original bytes와
development bundle binding을 확인한다. 선택 checkpoint/policy bytes는 후속 evaluator가
독립 확인해야 한다. 이는 저장 경계이며 OS 접근 제어나 never-seen attestation이 아니다.
기존 legacy standalone five-split 준비와 v1 checkpoint 동작은 유지한다.

각 단위는 코드/테스트 → lint 수정 → format → 재검사 → 관련 테스트 → diff → 별도
상세 commit으로 진행했다. 고정 `841134a` snapshot은 **1484 passed, 1 skipped /
367.71s**, source 변경0, Ruff261 files로 통과했다. Git archive에서 빠진 ignored
calibration/dev 두 fixture만 hash 확인 후 연결했다. private test/artifacts는 복사하지
않았다. `.dev/codex-evidence-holdout-frozen-841134a-20261004-r2.json` 참조.
Swift 소유 파일과 사용자 artifacts는 수정하거나 커밋에 포함하지 않았다.

## 직접 학습·Swift 입력의 연결

`6e17791`: evidence head뿐 아니라 기존 직접 학습 Trainer도 Swift의 full chat input과
canonical letter readout을 쓰는 opt-in encoder를 추가했다. original candidate order와
soft gold는 유지하고, 정수순으로 표시하는 Score는 원래 candidate ID/ordinal 의미로
되돌린다. fractional Score는 integer Swift API로 조용히 반올림하지 않고 거부한다.
원본 전체 입력·description span·label ID·gold·serving recipe가 바뀌거나 legacy/Swift
행이 섞이면 forward 전에 실패한다. trace·proposal·image는 이 direct arm에 들어가지 않는다.

실제 random Gemma/Granite에서 **기존 Trainer**의 raw logits/확률이 Swift HFReader와
4 variants × 2 formats × full/shared prefix 경로에서 일치했다. native full LM과
공동 typed loss·모든 backbone gradient도 일치했다. 관련196 CPU tests 통과.
이것은 pretrained 지능 향상을 측정한 결과가 아니다.

`bc75502`: **현재 직접 bundle schema는 v5**다. `input_encoding`과 exact `input_recipe`를
prepare/audit·teacher filter·학습·calibration/dev·frozen replay·export에 고정한다.
CLI `--input-encoder swift_canonical --prompt-variant labeled --state-format compact`
등으로 선택한다. `ayaka_segmented`가 기본이며 기존 v1 입력/체크포인트는 유지한다.
v3/v4 bundle과 이전 replay receipt는 새로 준비해야 한다. schema 이름만 고치지 않는다.

Swift teacher의 `direct_probs`는 같은 messages/input IDs/canonical IDs/recipe/candidate
mapping의 `direct_readout_binding`을 요구한다. 다른 prompt의 직접 분포와 비교해
teacher 이득을 계산하지 않는다. 기존 legacy encoder의 teacher contract는 유지한다.
digest는 provenance/internal consistency이며 backend execution attestation은 아니다.

작은 실제 LoRA 모델에서 fixed schedule 완주 → calibration/dev → export reload가
통과했다. step1부터 재개한 최종 trainable tensor는 연속 실행과 완전히 같았다.
평가는 더 큰 serving context를 사용하며 원본이 넘치면 truncation 없이 거부한다.
관련169 tests, 최종 Swift bundle15 tests 및 최종39 tests가 통과했다.

고정 `bc75502` 전체 통합은 **1582 passed, 1 skipped / 422.62s**, source 변경0,
Ruff lint/format268 files로 통과했다. `.dev/codex-swift-direct-full-bc75502-20261004.json`
참조. 이 snapshot 뒤 상대 세션에서 추가한 uncommitted API/candidates 경로는 포함하지 않는다.

같은 고정 소스에서 실제 cached pinned12B tokenizer+config로 v5 authored control을
다시 준비하고 전체 audit을 통과했다. `.dev/direct-native-swift-input-bc75502-cpu-20261004/receipt.json`
참조. 480 authored questions 중 train96rows는 각1회, 세 유형 각32개, 총10715 input
tokens다. source lineage는26개이며 질문96개를 독립 case96개로 부르지 않는다.
`labeled/compact` 선택은 입력 연결 검사 설정이며 dev 품질에 의한 승격이 아니다.

```text
bundle manifest: 499ef21fb1278e50b6c4c2f09278f96b231439029978c3af7bac172e32fc543d
holdout manifest: d1223c0f4409fb86c65bd1505e1a200af63e2f3099a33e82716bbbd371e34d33
```

teacher0·optimizer0·downloads0이며 meta FA2/Liger topology를 고정했다. native weights
ready=False다. 실제 pretrained probabilities, CUDA kernel 실행·속도 또는 품질 학습을
수행한 것이 아니다. 큰 tokenizer의 반복 준비/audit은 CPU에서 먼저 완료하고 고정해야
한다. GPU 서버에서 이를 불필요하게 다시 준비해 과금 시간을 쓰지 않는 경로도 보강한다.

남은 실제 준비는 자연 정책·다단계 품질 corpus와 teacher 관측, matched pilot·최종 독립
평가, 실제 native weights와 CUDA kernel/속도, provider 잔액·견적에 묶인 전체 cloud
완주·회수·정리 경로다. feature store/head training runner와 selected checkpoint/policy
bytes를 고정하는 독립 최종 evaluator도 별도 완성이 필요하다.
이후 작업 전까지 학습 직전 준비 전체 완료나 v1 추론 대비 향상을 주장하지 않는다.

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

## Swift 관측 → 비추론 teacher와 CPU 준비 최적화

`c68bc76`의 `ayaka.training.teacher_artifacts`는 저장된 Swift 직접 판정과 풀이 후
판정을 직접 학습용 teacher 확률로 변환한다. 외부에 고정한 train 원문 byte SHA와
direct/paired read의 **순서 있는 JSON 목록** fingerprint를 요구한다. 현재 파일에서
새로 계산한 hash만으로 외부 anchor를 대체하지 않는다.

원래 train 문맥·후보·soft gold·source lineage와 별도 verifier의 정답을 대조하고,
실제 tokenizer로 직접 입력·풀이 생성 입력·풀이 뒤 최종 입력을 재생성한다. 표시 글자
A/B/... 순서를 검사해 dict 삽입 순서 변경으로 후보 확률을 뒤집는 공격을 거부한다.
Noul과 정수 Score는 원래 candidate ID/순서로 되돌린다. 숫자 없음·높은 확신도·강제
1,024토큰 관측도 auto eligibility로 제외하지 않는다.

완료된 풀이의 확률만 내보내고 풀이 텍스트는 학습 입력에 넣거나 내보내지 않는다.
빈 풀이와 length-capped 풀이는 제외하되 input/output/reasoning tokens는 집계한다.
teacher가 정답 기준으로 직접 판정보다 나아졌는지는 기존 distillation 준비가 다시
필터링한다. 저장되지 않은 upstream 실패 비용은 별도 collection runner usage log가
필요하다. hash는 backend/loaded-weight 실행 증명이 아니며 export는 promotable=False다.

CLI는 `--train`, `--config`, `--input-encoding`, `--teacher-identity`,
`--direct-reads`, `--paired-reads`, 세 `--expected-*-sha256`, 새 `--out`을 받는다.
입력은 명시적 `swift_canonical`이다. production tokenizer는 cached pinned model
revision을 따른다. random tiny 검사에만 `--mechanics-only --mechanics-tokenizer
LOCAL_FAST_TOKENIZER`를 허용하며 production override나 download는 하지 않는다.
actual tiny raw-logit gather·실제 offline subprocess CLI 포함22tests 통과와 독립
리뷰를 완료했다. common Swift validator의 후보 순서 허점도 Claude에게 `.dev`로 전달했다.

`404fb90`은 순수 CPU 준비 scope 안에서 tokenizer backend 직렬화를 재사용한다.
scope 진입/정상 종료에 전체 바이트를 검사하고 중간에는 template·special tokens 등
설정과 객체를 확인한다. 변경은 저장 전에 실패하며 종료된 scope는 copied context에도
hash를 남기지 않는다. 기존 hash/row/recipe bytes, 실제 토큰·gold·source·전체 schedule
재생성은 유지한다. 영구 cache나 serving 변경은 없고 writer/optimizer 전에 scope가 끝난다.
처음부터 padding/truncation이 설정된 backend를 HF 호출이 정규화하면 실패할 수 있다.
동시/일시적 tokenizer 변조를 격리하는 장치로 부르지 않는다. 소스가 바뀌므로 이전
bundle은 새로 준비해야 한다. 관련80 CPU tests 및 Ruff가 통과했다.

고정 `404fb90`과 actual cached Gemma12B tokenizer에서 동일12문항의 기존 unscoped
입력 준비 **9.4736초 → scoped1.4567초**, 약 **6.50배** 개선을 관측했다. 직렬화12회→2회,
prepared rows는 완전히 같았다. tokenizer load14.7783초는 비교 구간과 별도다.
`.dev/codex-native-tokenizer-scope-404fb90-20261004.json` 참조. 단일 CPU 관측을
GPU 학습·실서비스 throughput 개선으로 확대하지 않는다.

같은 고정 소스의 authored480문항 v5 준비+전체 regeneration audit은 process import와
tokenizer load를 포함해 **35.8613초**에 통과했다. train96행은 각1회, 유형별32행,
10715 tokens, source lineages26개다. test는 별도 holdout이다.
`.dev/direct-native-swift-input-404fb90-cpu-20261004/receipt.json` 및
`.dev/codex-native-preparation-time-404fb90-20261004.json` 참조.
이전 고정 `bc75502`의 native control과 `train_items.jsonl`, `preparation.json`,
`recipe.json`, `test_commitment.json` 네 파일의 전체 bytes도 같았다.
`.dev/codex-native-preparation-equivalence-404fb90-20261004.json` 참조.

```text
bundle manifest: 7098e1cb0d2f68ad5552690100e266cb006ec6f4e1d9a1b0ba8ebdd5b7bd5f97
holdout manifest: 7df7de48c197093051bf7805b175f348879f20ac6639d8e339d8aa4a692bf431
```

이 control은 teacher0/optimizer0/download0/native weights ready=False이며 FA2/Liger
**meta topology** 검사다. 실제 pretrained 품질·CUDA 실행이나 준비 전체 완료를 뜻하지 않는다.

최종 고정 `404fb90` Git archive 전체 CPU 검증은 **1743 passed, 1 skipped / 341.39초**,
source 변경0, stable_pass=True다. Ruff lint/format274files도 통과했다.
`.dev/codex-teacher-scope-full-404fb90-20261004.json` 참조. archived source의 실제
import 경로를 확인했고, 필요한 ignored calibration/dev 두 fixture만 SHA를 확인해
연결했다. private test/artifacts/deploy는 복사하지 않았다. archive 이후 상대 세션의
unstaged API/media/Swift 변경은 이 통합 결과에 포함하지 않는다.

## API 포함 재검증과 실제 서빙 연결의 남은 문제

상대 세션의 API/SDK 변경도 포함해 별도로 확인했다. `9224446` 고정 전체 검사는
1787pass/1skip/1fail이었다. 유일한 실패는 기존 export HTTP 테스트의 400 기대와 새
계약 422의 차이였고, 상대 세션이 `55787c0`에서 수정했다. 실패 receipt도 보존한다.

최종 `55787c0` 고정 전체는 **1788 passed, 2 skipped / 350.46초**, source 변경0,
stable_pass=True, Ruff lint/format279files다. 새 optional `typesafe_sdk`가 이 venv에
없어 SDK 통합 모듈이 건너뛰어졌으며 상대 세션의 SDK 설치 환경 검증과 구분한다.
`.dev/codex-jev-teacher-full-55787c0-20261004.json` 참조.

같은 소스에서 native v5 준비와 전체 audit도 다시 통과했다. 이전 `404fb90` control의
train_items/preparation/recipe/test_commitment와 bytes가 같았다. teacher0/optimizer0/
download0/native weights ready=False 조건은 그대로다.
`.dev/direct-native-swift-input-55787c0-cpu-20261004/receipt.json` 참조.

```text
bundle manifest: fa8e288b4f15853d30aef8fa2c6ed39a685e5de663ce0790e835056b8677ca7f
holdout manifest: 9e0921beef20cc4c93115028abfe1b563f5f32c9a1095a64cee890bdfe81c14b
```

추가 독립 리뷰에서 **실제 출하 경로가 아직 연결되지 않은 문제**를 확인했다.
체크포인트 metadata의 Swift input recipe를 production loader가 읽지 않아 기본
Decision은 legacy segmented 입력을 사용한다. 서버 QuestionSpec에는 original Choice
wire labels도 전달되지 않으며 standalone export는 입력 metadata를 버린다.
Trainer의 pre-encoded probe 재로딩 일치는 이 서빙 경로를 검사한 것이 아니다.

또 detached text backbone의 저장 LoRA key와 HFReader full LM의 key 경로가 다르다.
PEFT는 누락된 key를 경고하고 계속할 수 있으므로 실제 학습한 nonzero A/B tensor가
로드됐는지 검사해야 한다. adapter-only HFReader는 checkpoint의 type/length
calibration도 읽지 않는다. raw-reader 일치와 calibrated-serving 일치를 나눠 검증해야 한다.
actual local tiny Granite + nonzero LoRA + nonunit temperature로 Service의 입력/token/
확률 및 실제 HFReader 로드를 검사하고, 연결 전에는 silent legacy fallback을 거부해야 한다.
이 작업은 `.dev`로 공유했다. 준비 전체 완료나 새 모델의 성능 향상을 아직 주장하지 않는다.

## 실제 체크포인트 서빙·추론 연결 완료

위의 serving blocker는 `8b552b6`에서 수정했다. loader가 체크포인트의
`input_encoding/input_recipe/input_recipe_sha256`을 검증하고 실제 Decision에 전달한다.
Swift 직접 학습과 같은 renderer·chat template·canonical letter readout을 사용하며,
서버의 원래 Choice label, Noul false/true, 정수 Score의 원래 ID/ordinal 순서를 유지한다.
typed/length temperature도 학습 체크포인트에서 읽는다. Swift에서는 legacy prefix
header와 pointer shortlist를 쓰지 않으며, 실제 입력 토큰으로 usage를 계산한다.

부분 metadata·변경된 tokenizer/template·hash 불일치를 거부하고, 저장한 checkpoint에는
`input_contract_required`를 기록한다. 전체 metadata를 삭제해도 legacy 입력으로 조용히
복귀할 수 없다. 이전에 이 계약이 없던 공개 v1/legacy checkpoint 동작은 유지한다.
native Swift Score에서는 +2/+4 같은 정수 표현을 보존하며 fractional/중복 숫자 alias는
HTTP422로 거부한다. 입력 문맥 초과도422이고 tokenizer 계약 손상은 backend502다.

Swift recipe는 **`ayaka-swift-direct-inputs-2`**로 올렸다. 기존 backend/template SHA에
`tokenizer_config_sha256`을 추가해 BOS/EOS·special tokens·decode 설정까지 묶는다.
이전 Swift recipe1의 bundle/checkpoint metadata는 명시적으로 재준비해야 한다.
schema/version 문자열만 고쳐 통과시키지 않는다.

`ayaka/swift_continuation.py`는 같은 adapter로 풀이를 생성한 뒤 native 3-turn
template의 최종 판정을 같은 cache에서 이어간다. `on + high`는 직접 판정·auto
라우터를 호출하지 않고 시작하며 단순/높은 확신도/숫자 없는 질문에도 1,024 예산을
유지한다. EOS로 정상 종료하면 길이를 채우지 않는다. EOS는 생성량에 포함하되 final
template 이전에는 cache에 넣지 않아 Gemma sliding cache의 불필요한 rollback을 피한다.
빈 풀이·문맥 부족·decode 실패는 fallback 사유와 실제 생성량을 남긴다. 조용히 effort를
낮추지 않는다. `off`와 명시적0예산은 생성0회다.

실제 CPU random Granite/Gemma와 저장한 native LM·fast tokenizer·nonzero LoRA·nonunit
temperature로 Service의 입력 IDs와 보정 확률을 Trainer와 비교했다. 질문별/요청별
격리와 warmed cache, 실제 1,024토큰 강제 생성, Gemma sliding cache의 trace-final
logits/full-row 일치, EOS/실패 usage를 검사했다. 이 단위의 관련240tests가 통과했다.
현재 Swift 단일 판독은 질문당2–26후보다. 26개 초과는 명시적으로 거부하며 Claude의
grouping arm을 자동으로 사용하지 않는다. native template가 지원하지 않는 cache
rollback을 요구하면 명시적 fallback이다. 모든 모델 계열의 cache 지원을 주장하지 않는다.

## 학습한 LoRA의 native full-LM export 완료

`ee47993`의 `ayaka.adapter_export`는 detached text adapter를 full causal LM에 맞는
경로로 변환한다. cached pinned native config로 **meta architecture**만 구성해 실제
text stack과 PEFT target을 확인하며, 필요한 모든 A/B key·shape·유한성을 검증한다.
일부 layer의 A/B 쌍 전체가 빠진 경우도 거부한다. full native class의 `auto_mapping`을
기록해 explicit HFReader와 PEFT 자동 loader 모두 실제 adapter를 읽도록 한다.
source config/head/meta/adapter의 변경을 검사하고 새 디렉터리에만 내보낸다.

```bash
python -m ayaka.adapter_export \
  --checkpoint /path/to/recipe-bound-ayaka-checkpoint \
  --out /path/to/new-native-hf-adapter
```

이 artifact는 **raw canonical-letter readout**이다. 원본 type/length temperature는
sidecar에 기록하지만 적용하지 않으며 `calibration_applied=false`다. 보정된 서비스는
원래 Ayaka checkpoint로 실행한다. native reader의 raw 확률을 보정된 확률이라고
보고하지 않는다. 변환 자체는 native weights를 로드하거나 다운로드하지 않는다.

실제 local random Granite/Gemma에 `HFReader._load`와
`AutoPeftModelForCausalLM.from_pretrained`를 사용해 **모든 저장 A/B tensor가 정확히
로드됐고 missing-adapter 경고가 없으며 raw logits/확률이 일치함**을 확인했다.
loader를 mock한 자기 재로딩 검사가 아니다. full Gemma audio 포함 config의 meta
구성, deterministic export, source 변경/누락/NaN 거부, 실제 offline CLI도 검사했다.
export와 serving 두 모듈의 최종 관련40tests가 통과했다.

고정 Gemma12B revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`의 meta 조사에서는
text stack이 `model.language_model`이며 rank32 adapter는 **656 A/B tensor,
131,137,536 parameters**였다. 7개의 target pattern이 328개 module에 적용되는
구성이며 native weights/VRAM/품질 실측은 아니다.
`.dev/codex-native-adapter-layout-20261004.json` 참조.

## 최신 고정 통합 검사와 남은 학습 조건

`ee47993` Git archive에서 상대 세션의 matched public 2×2 진단 코드까지 포함한 전체
CPU 검사는 **1852 passed, 2 skipped / 461.69초**, source 변경0,
stable_pass=True다. Ruff lint/format289files도 통과했다.
`.dev/codex-native-serving-full-ee47993-20261004.json`에 실행/log hash를 기록했다.
archive SHA는 `0c27ab8ca2ba769bbbce88c5574b24aece79c54508e4bdc10aba9910a46e2c99`다.
두 skip에는 이 venv에 없는 optional `typesafe_sdk` 통합 모듈이 포함된다. 상대 세션의
SDK 설치 환경 검사와 구분한다. ignored calibration/dev fixture 두 파일만 SHA 확인 후
연결했고 private test/artifacts/deploy는 복사하지 않았다. GPU/유료API/다운로드0이다.

같은 고정 소스와 실제 cached Gemma tokenizer로 recipe2 development bundle을 다시
준비하고 전체 regeneration audit을 통과했다. authored480문항 중 train96행은 세 유형
각32개이며26source lineages, 3fixed batches, 각행1회, 10715입력토큰이다. test 원문은
별도 holdout에 있다. 이전 `55787c0`의 train96행과 비교하면 변경한
`direct_input_binding`을 제외한 **모든 필드가 동일**하다. 파일 전체 bytes가 같다는
주장은 아니다. `.dev/codex-native-input-migration-equivalence-ee47993-20261004.json`
및 `.dev/direct-native-swift-input-ee47993-cpu-20261004/receipt.json` 참조.

```text
bundle manifest: 956ad9c67bc7a4720d0888ee3be6d0b40c804b46730da5d5cf8546072a8bba1d
holdout manifest: 2d6dc429084dfd08ac08560dee1d0e2b4c77c6bb3715ac00b624ee3a60425c6c
```

이 control의 teacher/optimizer 실행은0이고 FA2/Liger는 **meta topology** 검사다.
native weights는 `native shard inventory differs from the declared weight index`로
ready=False다. 실제 pretrained 품질 향상이나 GPU 학습 시간을 측정하지 않았다.
CPU 준비 최적화의 과거 속도 관측을 이번 동시 실행 시간과 비교하지 않는다.

다음은 실제 native weights 준비, 자연 정책·다단계 품질 corpus와 실제 teacher 관측,
matched dev pilot 및 고정한 checkpoint/policy bytes의 독립 최종 평가, CUDA 성능과
잔액·가격에 묶인 전체 완주/회수/정리 경로다. evidence head의 feature-store/training
runner도 별도 절제군 준비 항목이다. Swift 공통 `token_input`의 BatchEncoding 경계와
`SwiftReadIndex`의 canonical dict 순서 문제는 Claude 담당자에게 수정 요청을 남겼다.
이번 bridge/converter의 방어를 공유 Swift 코드의 해결로 부르지 않는다.
**전체 학습 직전 준비 완료나 v2 off의 v1 reasoning 대비 향상은 아직 증명하지 못했다.**

## 학습 서버 시작 전 감사의 재사용과 데이터 재읽기 수정

`465c18f`의 `training.direct_audit`는 CPU에서 전체 gold/token/teacher/schedule
regeneration을 완료한 뒤 별도 감사 receipt를 만든다. receipt와 원래 bundle manifest의
**각각 외부에 고정한 raw-byte SHA256**을 모두 요구한다. 현재 파일에서 새로 계산한
hash만으로 원래 외부 anchor를 대체하지 않는다. 감사 중 payload·source·dependency·
tokenizer 변경은 저장 전에 실패한다. receipt는 bundle/default holdout 밖 새 파일에
저장하며 custom holdout 경로의 위치는 자동 추론하지 않는다.

서버의 빠른 경로는 모든 payload를 hash 확인한 같은 메모리 bytes에서 data-only
dataclass로 읽는다. pickle이나 전체 재토큰화, 원본 natural source archive 읽기는
하지 않는다. 현재 source·survey·meta architecture·tokenizer/backend/template/설정은
대조한다. 후보/정답/recipe mapping과 opaque holdout 격리를 검사하고 전체 inventory와
fixed schedule을 다시 계산한다. Decimal ordinal·teacher 분포·샘플 내 prefix 객체
공유를 유지하며 frozen replay는 나중에 native snapshot에 따로 묶는다.
이는 신뢰한 CPU regeneration 결과의 재사용이며 backend 실행 증명/동시 수정 격리나
성능 승격은 아니다. 기존 직접 bundle v5/schema와 Swift recipe2는 유지한다.

```bash
# GPU 할당 전에 같은 고정 소스에서 전체 CPU 감사를 수행한다.
python -m ayaka.training.direct_audit \
  --bundle /path/to/development-bundle \
  --expected-manifest-sha256 ORIGINAL_BUNDLE_SHA256 \
  --out /path/to/new-cpu-audit.json

# 반환한 receipt SHA를 bundle 밖에 고정한 후 빠른 CPU 검사를 사용할 수 있다.
python -m ayaka.training.run_direct audit \
  --bundle /path/to/development-bundle \
  --expected-bundle-sha256 ORIGINAL_BUNDLE_SHA256 \
  --audit-receipt /path/to/new-cpu-audit.json \
  --expected-audit-receipt-sha256 PINNED_AUDIT_RECEIPT_SHA256
```

`76ba69b`는 위 두 receipt flags를 `profile/train`에도 연결했다. 생략하면 full
regeneration이 기본이다. 두 경로 모두 감사한 split 메모리와 **같은 tokenizer 객체**를
Trainer/replay/calibration/dev에 전달해 검증 후 파일/토크나이저 재읽기를 없앤다.
profiler도 source ID 대신 original sample 객체로 행을 연결해 같은 출처 ID의 서로
다른 질문들이 덮어써지지 않는다. original test는 읽지 않는다.

논리적 감사 내용은 optimizer continuation에 묶고 외부 receipt-file digest/로드 mode는
로그로 남긴다. 같은 외부 bundle anchor를 쓰면 full→fast 재개가 가능하다. actual
native weight/header 검증, 명시적 CUDA 실행/전체 잔액 admission, 커널 probability/
loss/gradient parity 및 실측 전체 schedule/disk admission은 계속 학습 전에 수행한다.
실제 random native LoRA의 full/fast/interrupted-resumed 최종 trainable tensor와 dev
확률이 정확히 같았다. 관련92tests, teacher/CLI 후속4tests, runner38tests 및 실제
pipeline CLI1test가 통과했다. 독립 read-only 리뷰도 두 단위의 blocker를 찾지 못했다.

고정 `76ba69b`의 실제 cached Gemma12B tokenizer에서 같은96행의 warmed CPU
startup 감사는 **6.4074초 → 2.1252초 (3.015배)**, backend 직렬화6회→2회였다.
item 전체 bytes·inventory·split·논리 binding이 같았고 같은 tokenizer를 사용했다.
tokenizer load9.0153초는 별도다. 단일 CPU 관측이며 GPU/학습 throughput 측정이 아니다.
calibration/dev는 평가 때 여전히 입력을 렌더링한다. 전체 GPU 학습이3배 빨라졌다고
해석하지 않는다. `.dev/direct-native-swift-input-76ba69b-cpu-20261004/startup-comparison.json`
참조. 실제 native tokenizer prepare/regeneration audit도 통과했다.

```text
bundle manifest: 812dc760c864e0c898ae3b1ca4f7b09cb553e51a5ffa5ada9a572260b41da717
holdout manifest: d08397de36f04146af5b5522e6b668adf660db63275af91fb7be57f0debeea95
CPU audit receipt: 82a9fd28a4feb6781ff0d9d4eca8c0a0618a74632c967f04f3d7cd235a44db7f
```

새 control도 train96/10715tokens/26lineages/3completebatches, teacher/optimizer/GPU/
download0, native weights ready=False다. 독립 검토에서 **빈 HF 캐시의 서버에 arbitrary
`--snapshot-path`만 업로드하면 초기 metadata/tokenizer audit가 cache repo를 요구하는
문제**도 확인했다. native snapshot inventory는 현재 tokenizer 자산을 포함하지 않는다.
업로드만으로 즉시 실행하는 경로에는 검증된 local metadata 경로와 full config/tokenizer
binding이 추가로 필요하다. portable receipt의 transformers/tokenizers/peft 버전도 맞춰야
하며 source checkout의 approved survey file을 함께 전달해야 한다. 이 조건들과 앞의
실제 품질/가중치/CUDA 조건을 충족하기 전에는 학습 직전 전체 완료로 표시하지 않는다.

최종 고정 `76ba69b` Git archive 전체 검사는 **1900 passed, 2 skipped / 565.32초**,
source 변경0, stable_pass=True, Ruff lint/format292files로 통과했다.
`.dev/codex-direct-audit-full-76ba69b-20261004.json` 참조. archive SHA는
`f4f93ac8ddf52d1138697901225b3a07c8c7d6b7f47d5d974e5c496ff04dc824`다.
기존 optional SDK 미설치 skip 등 두 skip은 그대로이며 실제 CUDA kernel 검사를
CPU 결과로 대체하지 않는다. private test/artifacts/deploy는 복사하거나 변경하지 않았다.

### 로컬 업로드 경로와 native loader 바이트 검증 (2026-10-05)

`88722ae` / `378ff65`는 빈 HF 캐시 서버에서 업로드 폴더를 쓰는 경로를 구현했다.
준비·전체 감사·CPU 감사 재사용은 `--native-path`, runner는 `--snapshot-path`를
받는다. 실제 tokenizer/config와 최초 모델/체크포인트 reload 모두 같은 로컬
디렉터리를 사용한다. 선언한 repository/revision과 승인 survey는 그대로 검증하며,
명시한 폴더가 불완전하면 캐시로 우회하지 않는다. 로더는 local-only/remote-code-off다.

새 **direct bundle6**에는 native metadata inventory와 full config SHA가 들어간다.
같은 크기의 rope/scaling/dropout 변경도 AutoConfig/meta 생성 전에 거부한다.
새 **deployable snapshot2**는 safetensors index/header/모든 shard 바이트에 더해
fast tokenizer/config/template 자산을 포함한다. 기존 weights-only snapshot1 API는
유지하지만 native profile/train에는 쓸 수 없다. bundle5와 이전 CPU receipts는
같은 소스/메타데이터로 새로 준비·감사해야 하며, 체크섬만 고쳐서 재사용하지 않는다.
v1 모델/서빙이나 main의 변경은 없다.

설치된 HF의 실제 preferred directory(`additional_chat_templates`)와 versioned fast
files도 inventory에 넣는다. config redirect는 명시적으로 거부하며 canonical
`config.json`이 필요하다. implicit PEFT `adapter_config.json`/adapter payload는
fresh-base 목표와 충돌하므로 거부한다. 필수 파일 이름은 Linux에서 그대로 열릴
소문자 이름이어야 한다. 대소문자 충돌도 허용하지 않는다. snapshot/CPU receipt를
native 디렉터리 안에 만들면 loader inventory를 바꾸므로 출력 위치를 거부한다.

독립 read-only 검토에서 실제 preferred template 경로, embedded adapter 로딩,
감사 종료 후 metadata 변경, 긴 weight hash 도중 새 preferred 파일 추가를 잡았다.
수정 후 metadata 전체와 shard inventory를 성공 반환 전에 다시 검사한다.
이것은 검증한 로컬 입력을 묶는 기능이며 publisher 바이트의 독립 인증이나
악의적 concurrent writer를 막는 파일 시스템 격리라고 주장하지 않는다.

완전한 native 폴더와 bundle6를 준비한 뒤 CPU에서 다음처럼 검사할 수 있다.
기존 receipt와 manifest hash는 별도의 신뢰된 기록에 저장한다. source checkout의
`docs/experiments/v2_candidates.json`과 CPU receipt에 묶인 정확한 dependencies도
함께 필요하다. pip 설치와 가중치 폴더만 전달한 상태를 준비 완료로 보지 않는다.

```sh
python -m ayaka.training.native_snapshot \
  --repo PINNED_REPOSITORY --revision IMMUTABLE_REVISION \
  --path /srv/native --out /srv/receipts/native-snapshot.json
python -m ayaka.training.direct_audit \
  --bundle /srv/bundle --native-path /srv/native \
  --out /srv/receipts/cpu-audit.json \
  --expected-manifest-sha256 ORIGINAL_BUNDLE_SHA256
python -m ayaka.training.run_direct audit \
  --bundle /srv/bundle --snapshot-path /srv/native \
  --snapshot-record /srv/receipts/native-snapshot.json \
  --expected-bundle-sha256 ORIGINAL_BUNDLE_SHA256 \
  --audit-receipt /srv/receipts/cpu-audit.json \
  --expected-audit-receipt-sha256 PINNED_AUDIT_RECEIPT_SHA256
```

Runner audit는 snapshot record가 있으면 전체 weight bytes도 검사하고 그 여부를
`native_weight_bytes_verified`에 기록한다. tensor를 모델에 읽지는 않는다.
profile/train에서는 동일 snapshot2·bundle metadata 일치, 실제 weight 검증,
CUDA 실행과 전체 잔액 admission, 실제 probability/loss/gradient 커널 parity를
모두 통과해야 optimizer가 진행된다. native CPU mechanics에는 명시적으로
`--mechanics-only`를 붙이며 이는 production-weight/quality 증거가 아니다.

관련 **219 CPU tests / 335.19초**와 own8files Ruff lint/format 재검사가 통과했다.
실제로 저장한 pristine random Gemma/Granite sharded native LM+fast tokenizer를
빈/offline HF cache에서 prepare/audit/receipt/snapshot/runner CLI로 검사했다.
폴더 이동 후 item bytes·recipe·논리 audit binding이 같으며 full/fast/resumed
최종 trainable tensors와 dev 확률도 완전히 같고 checkpoint reload parity가 통과했다.
초기 새 fixture가 tuple/이미 detach한 head를 잘못 저장한 부분은 pristine native
constructor로 교체했다. strict missing-head loader나 gold 검증은 완화하지 않았다.

고정378ff65의 실제 cached Gemma12B tokenizer/config control도 bundle6로 재생성됐다.
기존 inventory/schedule hash와 96rows/10715tokens/26lineages/3completebatches는 같다.
FA2+Liger는 meta topology 검증이며, 실제 CUDA 측정은 아니다. control receipts는
`.dev/direct-native-swift-input-378ff65-cpu-20261005`에 있다.

```text
bundle manifest: 4889c4d43e922097f7c23777fc905c31744b3230f423d6b60fb5d27464431834
holdout manifest: ca373f23978fea61800ba54d4503491e22018253a5e691312ef827b2c08c4002
native metadata: 0db749370aa1641172bb189754c0a4c8fa914346d68049754e04eacb4dd82aa2
full native config: 478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9
```

새 control의 teacher/optimizer/download/GPU0, native weights ready=False다.
실제 quality corpus/teacher와 complete publisher weights/CUDA 검증, 같은 조건의
v1 reasoning 대 v2 off 품질 측정, 최종 독립 test 및 provider 회수/정리가 남았다.
이 변경은 업로드 준비 문제를 해결한 것으로, 성능 향상이나 학습 직전 전체 완료를
입증하지 않는다. Shared Swift의 token_input/SwiftReadIndex P1 상태는 `.dev`로
소유자에게 재확인 요청했다. Swift/serve/private test/artifacts/deploy는 수정하지 않았다.

최종 고정378ff65 전체 검사: **1963 passed, 2 skipped / 687.87초**, source 변경0,
stable_pass=True, exit0. Ruff lint/format295files도 통과했다. Git archive SHA는
`3799fab0d85d0f2c0ba528d0f3476f09b765b2ab5ba0853784a68a4793bb30b9`다.
`.dev/codex-native-upload-full-378ff65-20261005.json`과 log SHA
`edef0c050770f8ee83f991f01276485b2cdc789caea966eb5939213ff71a3a91`로 기록했다.
검사 child 환경만 Git Bash/OMP_NUM_THREADS=1/MKL_NUM_THREADS=1을 지정했다.
시스템 환경은 변경하지 않았고 전체 소스는 검사를 마칠 때까지 고정했다.

전체 pytest 종료 후, 겹치는 작업 없이 동일 warmed native tokenizer에서 새 schema6
CPU 감사 startup을 측정했다. full regeneration **6.2661초 → anchored reuse 2.0287초
(3.089배)**, backend 직렬화6→2회이며 전체 item bytes·inventory·splits·논리 audit
binding·사용한 tokenizer 객체가 같다. 별도 tokenizer load는7.3131초다.
새 CPU audit SHA는 `2854496f0c3179922a0eb1fa62e925682221730c27c06dee991f6a8d1cf7857d`다.
control의 `startup-comparison.json` 참조. 추가 raw metadata 검증을 포함한 단일 CPU
관측이며, 전체 GPU 학습 속도/품질 개선이나 CUDA FA2/Liger 성능으로 해석하지 않는다.
실험/test 프로세스는 모두 정상 종료했고 GPU/paid API/다운로드는 사용하지 않았다.

### 인간 라벨 분할·원본 검증과 실제 native 입력 준비 (2026-10-05)

`ayaka-v2-experiments`에서 세 코드 단위를 별도로 커밋했다. `main`의
`475bec37e183a021f52bf119408f57c2ac9de3e2`와 Swift/API 소유 범위는 유지했다.

- `2b2cc27`: 원래 질문·primary lineage·example ID·번역·파생·lineage aliases의
  연결 성분을 먼저 닫고, public/reserved 제외와 split/quota를 적용한다.
  HelpSteer의 다른 response/rating을 같은 prompt라는 이유로 중복 제거하던
  문제를 수정했다. 중복은 전체 state/question/gold가 동일한 경우로 한정한다.
  원래 raw primary ID를 보존하며, text의 공백은 정규화하고 media bytes/path의
  대소문자는 보존한다. Authored/natural의 교차 참조도 split 감사에 포함한다.
- 공개 평가 검사는 선택지 설명과 구조화된 state의 문자열도 검사한다.
  공백 없는 CJK에는 명시된 character/exact 규칙을 추가했다. Opaque holdout2는
  개별 identity의 해시를 동일 namespace에서 비교해 train의 `derived_from`이
  test의 `source_example_id`를 참조하는 경우도 원문 없이 거부한다.
- `a4672f2`: 고정 raw file SHA를 읽기 전/parse 후/최종 출력 전에 재검사한다.
  메모리의 rows/features/provenance/binding 변경도 감지한다. 준비와 전체 감사는
  같은 registry를 유지해 전체 corpus의 중복 parse를 제거한다. Full audit/receipt
  출력 직전 raw byte 검사도 수행하며, 검증 후 교체 시 출력은 생성하지 않는다.
  이미 외부 해시로 고정된 prepared-row load는 raw cache 없이 계속 동작한다.
- Teacher 변환 CLI는 `--native-path`로 local package를 받는다. Native metadata의
  입구/출구를 확인하고, cache fallback이나 mechanics tokenizer와의 동시 override를
  거부한다. Saved observations를 변환할 뿐 모델/생성 backend는 실행하지 않는다.
- `b029c74`: natural sample이 test에만 있을 때도 준비가 가능하게 했다. Test/final
  raw 검사는 outer registry를 유지하고, authored-only development에는 unused
  registry를 전달하지 않는다. Development 감사는 test 원본이나 raw cache를 열지 않는다.

독립 Codex 검토에서 cross-kind alias, 공백, media case, late-write,
mutable parsed payload 및 natural-only-test 경계를 발견해 수정했다.
Claude가 이 단위들을 승인했다는 뜻은 아니며, `.dev`에 결과와 검토 요청을 공유했다.

관련 CPU 검사는 source/holdout147pass(76.20초), raw/native/fast-load128pass
(178.52초), 최종 serialization/policy32pass(33.55초), holdout corner44pass
(23.30초)다. 서로 겹치는 검사이므로 독립 테스트 수로 합산하지 않는다.
마지막 고정 `b029c74` 전체 검사: **1992 passed, 2 skipped / 726.54초**,
source 변경0, stable_pass=True, exit0. Ruff lint/format **298files** 통과다.

```text
fixed archive SHA: c13ba2ca95b2424c9a5ebe28a3dbdc9f26d320ca829eefe7efce44d140ab58ba
full receipt: .dev/codex-raw-gold-full-b029c74-20261005.json
full log SHA: 3d779f42798ce2ddd85d7e1757b48d991fd3f892e185dbf963564893bcc3eb29
```

실제 캐시의 고정 인간 라벨 53,093개 raw row를 변환하고 **157,417개 질문 전부**에
raw gold verifier를 호출했다. HelpSteer101620, CommonsenseQA9741,
MASSIVEko23028/ja23028이다. 이 파일의 target은 모두 정수/one-hot이었다.
소수 mean을 adjacent ordinal에 보존하는 계약은 별도 fixture로 검증했으며,
이 corpus에서 fractional label을 실제 관측했다고 주장하지 않는다.

고정 `b029c74`에서 actual Gemma4-12B native config/tokenizer로 text-only
mixed direct control을 준비하고 전체 감사했다. 모델 weight·optimizer는 실행하지 않았다.

| 준비 내용 | 실제 결과 |
|---|---:|
| Train sample / question | 928 / 1280 |
| Human rehearsal / authored question | 896 / 384 |
| EN / KO / JA question | 1088 / 96 / 96 (영어85%) |
| Choice / Noul / Score | 512 / 320 / 448 |
| Router-train / dev / calibration / test | 각240문항 |
| 선택한 train corpus의 전체 epoch | 1회, 40배치 × 32문항 |
| 모든 prepared row의 실제 schedule 방문 수 | min=max=1, unique1280 |
| Train text token / 최대 native input | 267898 / 1670 |
| Reasoning/proposal/teacher/optimizer/GPU/download | 모두0 |

Whole epoch는 **선택한 train1280문항**에 대한 것이다. Raw157417문항 전체를
학습하거나 main training을 실행했다는 뜻은 아니다. Source/sample quota를 먼저
정했고 모든 질문을 유지했으며, 부족한 quota를 다른 split에서 채우지 않는다.
선택지 순열은 원 질문당 한 번만 적용했다. Test 원본은 별도 holdout directory에 둔다.

```text
control: .dev/direct-human-native-b029c74-cpu-20261005
plan SHA: 91f84e5ea408810a9947204bac1647af9bc569ddcfaa2abb323f5ad0c06c75d9
bundle manifest: 8673695562857743a229936d8d96f17939eb9feed31cd25152e2ca50479bfecb
holdout manifest: 5c8316b149a19a8076ab2077a0a850cac3ca06123234a438d35a34d1e508ea66
CPU audit: c3dd0aa7d8a47da10dae26c56c73559cef03a8be248d8b76cb7174f6b53ace87
inventory SHA: 9469eea610ced8bbec13d64d5e391846a4350410c47a69f9370543adb18a2afa
schedule SHA: d10d1c69b29eb4142dfd41295a5a1b8857d22a92783c96066945717283da6dbd
```

별도로 고정한 `a4672f2` control과 모든 bundle payload hash가 일치했다.
선택 단계에 tokenizer identity scope를 재사용해도 행·gold·native token·schedule은
동일했다. 재생 시 raw registry/gold verifier/token 재생성을 금지하고1280행을
복원했으며 모든 행1회 방문을 확인했다. 기능적인 준비·재사용 검증으로,
CPU timing이나 GPU throughput의 비교 결과는 아니다.

이 control에는 prior private reserved inventory를 공급하지 않았다.
Human rehearsal은 일반 능력 유지이며 natural policy reasoning corpus나
실측 teacher benefit을 대신하지 않는다. 실제 policy corpus/teacher,
complete native weights와CUDA parity, 전체 workload의 credit admission,
matched v1 reasoning 대v2 off 품질 측정·최종 독립test와provider 회수/정리가 남는다.
FA2/Liger는 이 준비에서는meta topology만 확인했으며 GPU 성능 실측은 아니다.
학습 직전의 모든 조건 완료·성능 향상·승격을 주장하지 않는다.

## 2026-10-05: 정식 direct corpus CLI와 실제 실행 계약

`.dev` human control의 설명을 학습 소비자가 검증하는 정식 계약으로 옮겼다.
코드 커밋은 네 단위로 나눴다.

- `94dc45f`: 입력 계획·샘플별 marker·source/type/언어 quota와 전체 epoch 검증.
- `d8f4f4e`: offline `direct_corpus plan/prepare` CLI와 입력/저장 경계 검증.
- `626a779`: CPU 정적 입력과 runtime frozen-base replay의 호환성.
- `4d00f56`: 실제 inventory 전체의 runtime 변조 거부.

독립 검토에서 실제 group 교체, 모순된 opaque test 언어 집계,
파일 저장까지 이어진 tokenizer 재사용 scope, plan 저장 전 tokenizer 누락,
authored replay와 inventory 분류를 함께 바꾸는 반례를 발견하고 수정했다.
각 반례에는 회귀 테스트가 있다. Claude의 최신 incoming 문서에는 이 변경에
대한 새 승인이나 성능 판단이 없으며, outgoing `.dev` 문서로 검토를 요청했다.
Swift/API의 기존 소유권을 지켰고 `main`은 `475bec37` 그대로다.

계획은 선택 **이전**의 설정과 실제 모델·tokenizer·native metadata·encoder·
원본 gold registry·공개/기존 평가 inventory를 고정한다. 선택/검증/제외 결과는
별도 report에 기록한다. 외부 `plan_file_sha256`는 파일 바이트를,
`corpus_plan_sha256`는 canonical 내용을 고정한다. Fast load에는 원본 human,
public, private 데이터 캐시를 다시 요구하지 않는다.

공통 검증기는 prepare/full audit/fast load를 연결한다. Opaque holdout2의
선택적 extension에는 source별 sample/type 집계와 plan/member digest만 둔다.
원래 test 입력·정답은 별도 저장소에 유지한다. 실제 inventory·prepared row/group
digest를 고정하고 `E×R % B == 0`, 정확한 steps/seed, 모든 `(sample,row)`의 E회
방문을 검사한다. 언어별 replacement sampling, 반복 tail, 질문 제거, 실제 group
교체와 inventory 변조를 거부한다. Plan 관련 필드가 모두 없는 legacy6 경로만
기존 계약으로 읽는다. 선택 membership 고정은 원본 전체 선택 알고리즘을
재생했다는 증명이나 미열람 test의 실행 증명이 아니다.

Tokenization 재사용 scope는 메모리 내 선택까지만 유지한다. Bundle·holdout·
receipt를 쓰기 전에 scope를 닫고 실제 tokenizer를 재직렬화한다. 최종 candidate
permutation **후** 원래 질문 전체를 native encoder로 검사하며 typed overflow만
whole-sample 제외로 처리한다. Authored overlap/overflow는 quota 오류로 실패한다.
입력 파일, 공개/private 목록과 바이트, raw annotation, parsed reserved payload,
native assets가 변하면 결과 저장을 거부한다.

Frozen-base replay는 유일한 runtime attachment인 `base_probs`만 정적 digest에서
분리했다. CPU 준비에는 해당 값이 없어야 한다. 학습 전에는 positive replay
weight의 natural rows 전체에 aligned/finite/normalized 분포를 요구하고 authored와
gold-only 주입은 거부한다. 기존 native/input/candidate header와 optimizer-state의
frozen-read digest는 유지한다. 실제 무작위 CPU LoRA 업데이트와 정확한 저장·복원도
통과했다. Pretrained 학습 결과나 성능 개선의 증거로 사용하지 않는다.

고정 `4d00f56` 전체 CPU 검증:

```text
Ruff lint/format: 303 files passed
pytest: 2046 passed, 2 skipped / 793.69 seconds
exit: 0; source changes: 0; stable_pass: true
archive SHA: c635fa33078736e470b97e49fcf6b4a0c3ef1cd2ef8cd047c0040e4159122fe6
receipt: .dev/codex-direct-corpus-full-20261005-final.json
```

실제 캐시된 Gemma4-12B-it metadata/tokenizer와 byte-pinned human 원본으로 정식
CLI를 실행했다. Config는 LM readout/r32/alpha64, max_seq_len4096/serve8192,
Swift canonical/labeled/compact/nonthinking이며 FA2/Liger는 meta preflight 설정이다.
기존 private 평가 목록은 공급되지 않아 **draft** scope를 명시적으로 보존했다.

- 원본 53,093 rows의 157,417 transformed questions를 전량 raw gold와 대조.
  이 고정 원본에서 fractional target은 0개이며, 보존 동작은 fixture에서 검증.
- Train928 samples/1280 questions: human896 + authored384.
  EN1088/KO96/JA96 (영어85%), Choice512/Noul320/Score448, source lineages614.
- Router/dev/calibration/test는 각각160 samples/240 questions.
  Train 전체1epoch40×32, 모든 row min=max1, 267898 text tokens, 최대1670tokens.
  선택한 corpus의 한 바퀴이며 전체 raw53,093rows의 한 바퀴는 아니다.
- 원본 registry/public/private/encoder 읽기를 금지한 actual portable load도
  1280rows/928groups/40batches와 모든 row1회 방문을 재확인했다.
- 기존 `b029c74` control과 train_items/teacher_reads 바이트, 모든 split의 실제
  context audit와 schedule SHA가 같다. 새 계약은 검증/재현성 변경이며,
  모델 출력·성능이나 throughput 개선의 근거가 아니다.

현재 `4d00f56` control은 `.dev/direct-corpus-native-4d00f56-20261005`에 있다.

```text
plan file SHA: 84f5d6e9f9b954e021b7f056cb01e8e80433f56b59e1cafc8f0c33ba69fb57da
canonical plan SHA: 87f39d60824b45f54b8d86d6739a2fddb5121afe4b2fcdbd5e6c59ebcf234df7
bundle SHA: 2a39864cc040b7d2f9e5d759b234718f0b92bd7de5bbfa0e1454b1dca003b52c
CPU audit SHA: 3ad826be69b8e4d525e8a8b65303c22a044e83f639dca04cf42dd1a25328b278
holdout SHA: 58a4285fb5554923564ed09bb34df6291a026d4868b9987969b285ce95fe0823
selection report SHA: 40f517f348fedb03e4ade41db7dfb233dd747a65cde8f9cf572727453eb1e6f1
portable check SHA: aa2f5bebb6124838e0d65ec8cb74c2cde5bd78aca5c858056784ba87923c1867
```

명령과 private manifest 계약은 [DIRECT_CORPUS_PREPARATION.md](DIRECT_CORPUS_PREPARATION.md),
예시 설정은 `direct_corpus_settings.json`에 기록했다. 모든 자체 process가 끝났고
production optimizer/teacher/GPU/download/paid API는0이다. 전체 CPU suite와 실제
control은 동시에 실행했으며, CPU/GPU 속도 비교 실험으로 사용하지 않는다.

남은 조건은 prior private inventory, 실제 자연 policy corpus/teacher의 이득,
complete native weights와CUDA parity/throughput, 전체 workflow credit admission,
matched v1 reasoning 대비 v2 off의 품질·최종 독립test 및 provider 회수/정리다.
학습 직전 전체 완료나 v2 off > v1 reasoning을 아직 주장하지 않는다.

## 2026-10-05: 원본 전체 계약 문서의 policy 감독 추가

일반 human rehearsal에 실제 문서의 규칙·예외·정보 부재 감독을 추가했다.
공식 ContractNLI **original train**만 사용하며, 423개 NDA 문서와 7,191개 인간
annotation을 immutable revision·ZIP/train/LICENSE SHA로 고정했다. 모든 문서에
원래 17개 hypothesis와 Entailment/Contradiction/NotMentioned를 보존한다.
Neutral을 false로 합치거나 evidence span으로 문서를 잘라 gold를 노출하지 않는다.
문서 ID·URL·파일명·annotation span은 ancestry/verifier에만 쓰며 모델 입력은
원본 전체 문서, hypothesis, 세 클래스 설명이다. 원본 dev/test와 raw PDF 내용을
열지 않았다. 원본 자료의 의미적·법적 정답을 독립적으로 재판정한 것은 아니다.

출처와 재현 명령은
[DIRECT_CORPUS_PREPARATION.md](DIRECT_CORPUS_PREPARATION.md#5-add-original-full-contract-policy-supervision)에
기록했다. 네 소스 plan1은 그대로 유지하고 선택적인 다섯 소스 plan2를 추가했다.
Plan2의 private state excerpt 검사는 공유 hypothesis/schema를 제외하고 실제 state
문구의 중복을 전체 lineage group 단위로 제거한다. 짧은 4–12단어 문구와 wrapper가
있는 연속 CJK 12문자도 검사한다. URL query는 유지하고 fragment만 제거한다.
문서가 context를 넘으면 17문항 전체를 제외하며, 문서/질문 quota가 부족하면 실패한다.
이는 literal matching이며 의미상 paraphrase나 목록에 없는 private 평가를 보장하지 않는다.

구현은 문제별 상세 본문을 갖는 별도 커밋으로 나눴다.

- `2c2b6b1`: 전체 계약 문서의 raw human gold registry와 train-only extractor.
- `c8a3819`: 해시를 확인한 **동일 buffer**를 파싱하도록 파일 교체 경계 수정.
- `b33445c`: composite registry, plan2, private excerpt/group 검사, CLI·teacher·runner 연결.
- `69675b2`: 공식 ZIP의 미사용 raw PDF 이름을 filesystem destination과 분리.
- `117967d`: frozen replay 테스트가 새 공개 registry 인자를 직접 사용하도록 수정.

공식 ZIP의 raw PDF 이름에는 `T:\\proc_notices\\...pdf` 형식이 실제로 들어 있어
첫 정식 추출은 실패했다. `69675b2`는 미사용 raw 이름을 opaque metadata로만
취급하고 정확한 train/LICENSE member, byte anchor, 중복 이름 검사와 unsafe non-raw
경로 거부를 유지한다. 이후 실제 공식 archive의 정식 추출·prepare가 성공했다.
Runtime에는 downloader를 추가하지 않았다. 이번 작업의 다운로드는 공개 데이터
archive 65,362,913바이트 1개이며 pretrained weight/GPU/paid API/production optimizer는0이다.

독립 Codex 검토에서 해시·파싱 buffer 불일치, runner registry 전달 누락,
wrapped CJK excerpt와 반복 generator 소비 반례를 발견하고 회귀 검사로 수정했다.
Claude의 incoming 문서에는 새 승인·성능 판단이 없으며 `.dev`로 결과와 검토 요청을
공유했다. Swift/API 소유 파일을 수정하지 않았고 `main`은 `475bec37` 그대로다.

실제 cached Gemma4-12B-it tokenizer/configuration으로 offline 정식 CLI를 실행했다.
Revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, LM readout/r32/alpha64,
train4096/serve8192, Swift canonical/labeled/compact/nonthinking 설정이다.
FA2/Liger는 meta preflight이며 실제 CUDA kernel 검증을 대신하지 않는다.

| 실제 준비 내용 | 결과 |
|---|---:|
| 전체 raw 질문의 gold 대조 | 164,608 |
| Train sample / question | 992 / 2,368 |
| Human / authored question | 1,984 / 384 |
| Policy document / question | 64 / 1,088 |
| EN / KO / JA question | 2,176 / 96 / 96 (영어91.89%) |
| Choice / Noul / Score | 1,600 / 320 / 448 |
| Router-train / dev / calibration / test | 각각168 samples / 376 questions |
| Train 전체 epoch / batch | 1회 / 74 × 32 |
| Prepared row 방문 | unique2,368 / min=max1 |
| Train source lineages | 678 |
| Train text tokens / 최대 입력 | 2,349,245 / 4,037 |

선택한 train corpus 한 바퀴이며 전체 raw164608문항을 학습한 것은 아니다.
Contract 원본423문서는 closure420groups, 최대2samples/group이다. 17개 hypothesis는
17개의 독립 문서가 아니다. Train의 클래스 분포는 E481/C139/N468이며,
context 때문에 전체 제외한 계약 문서는6개다. 이는 quota를 채우기 전 검사한 문서의
제외 수로, 원본423문서 전체의 overflow 비율을 측정한 것은 아니다.
각 heldout의 최대 입력은 router5270/dev4576/calibration6951/test3545다.
모든 선택 문항에 원본 full context를 보존했으며 raw soft label은 실제로0개다.

Teacher transport fixture는 무작위 tiny LM의 17개 hypothesis와51개 saved backend
calls를 검사했다. 실제 reasoned teacher나 학습 성능 향상의 증거는 아니다.
이번 native control은 gold-only이며 teacher/production optimizer 실행은0이다.
Prior private evaluation inventory가 공급되지 않아 **draft** scope를 계속 기록한다.
Report의 `reserved_samples=768`은 authored controls이며 이전 private 평가768개를
검사했다는 뜻이 아니다. 실제 test 입력은 별도 holdout directory에 보관한다.

Portable load에서 raw registry/gold verifier/encoder/public/private inventory entrypoint를
금지하고2,368rows/992groups/74batches와 모든 행1회 방문을 재확인했다.
첫 로컬 검사 helper는 editable checkout을 잘못 import해 LF snapshot으로 고정된
receipt와 source bytes가 달라 거부됐다. Immutable checkout을 명시한 뒤 통과했다.
`69675b2`와 최종 `117967d`의 ayaka Python147files는 바이트가 같으며 test만 바뀌었다.
기존 control의1,280개 train rows는 policy 추가 후에도 **모든 필드가 동일**하다.

비용 진단은 같은 frozen buffers와 실제 schedule을 사용했다. Micro budget8192에서
독립 full row는436chunks/2,349,245 unpadded/2,548,843 padded forward input tokens다.
Prefix 공유가 가능하고 activation checkpointing이 꺼진 계획은
233chunks/434,039 unpadded/501,012 padded tokens다. 모든2,368문항을 유지한다.
Shared chunks에는157개 prefix forward가 추가되므로 예정 backbone 호출은390회다
(independent436회). Chunk 수를 실제 호출 수나 wall-clock 감소율로 읽지 않는다.
이는 입력 처리량의 정적 계산이며 GPU 시간·VRAM·비용·v1 속도 비교가 아니다.
Shared token budget은 prefix+batch×suffix이며, 반복된 batch×prefix KV cache의
메모리 상한이 아니다. Backward 재연산·OOM retry 비용도 이 집계에는 없다.
OOM이 checkpointing을 켜거나 모델 cache sharing이 미지원이면 이 절감을 적용할 수 없다.
현재 LoRA dropout0.05에서는 같은 prefix를 공유한 질문들의 stochastic noise도
공유된다. Independent rows와 개별 loss/gradient의 결정적 일치를 주장하지 않는다.
Zero-dropout parity와 기대 목적함수·실제 학습 품질은 별개로 검증해야 한다.

```text
actual control: .dev/direct-policy-native-69675b2-20261005
plan file SHA: 825bd93bcb4918b3412f26e279db3632f1acbe23d2afa5efebbe0ff3a810f0ad
canonical plan SHA: c4021b68d814fb9a7cff393bf72d37c2e74e9f0f7103b37acfb11f8b139d54a1
bundle SHA: 6b2eca89f873f0d39315a131c522524e7682a7da6ea1da84c0412df8c0e89707
CPU audit SHA: 3e827fbec083144fa03eca973bc18ce9031d65ee84a93e801ff1a58ebe401813
holdout SHA: d820025097d93da7e702c260b02252adc5c525a2f6c6334655cf9ac93dc72dc3
selection SHA: af13afe870b2bff55291e86baa44e96c25850db86166f3ed36df2c444a39fb45
inventory SHA: cf090cf865159a48c983cf6f6dcb95f8ca2d62ccfc9d253df208b76918690ade
prepared groups SHA: 626d787b255515484ad845503ac6f60e6bc1cbc32ce30256eca9c4a13c3a9374
schedule SHA: 6a058dde98a84d54248bbb12cdc3fac1bc27c6db532ca8b25c0fa700a5ead669
portable check SHA: c01bbb3d68aea61ae63d7b86ad30beabbf2c4b662df3229ae3aa78935d5be0e3
workload shape check SHA: 8ee3b7f0ceab35825fb494f9f90a9ca58d10f39eaf684d86a94c9b9da52ac567
```

관련 검사는 Contract35pass, source/plan/teacher/replay120pass와 최종 replay32pass다.
서로 겹치는 suite이므로 합산하지 않는다. 수정 전 고정 `b33445c` 전체 결과는
2095pass/1fail/2skip/802.40초였다. Fail은 기존 테스트의 registry 이중 주입이며
새 인자를 사용한32개 검사에서 연속 실행·재개 실행의 최종 가중치 완전 일치를
다시 확인했다. 수정된 `117967d` 고정 전체 검사도 **2096pass/2skip/1warning,
649.53초, exit0/source변경0/stable_pass=True**로 끝났다.

```text
archive SHA: 29eb6b18552b22aa2c68c74a284a9185f66145963afaff75f4c094aa6c09fbc1
full receipt: .dev/codex-policy-corpus-full-117967d-20261005.json
full log SHA: 4a976664c7dc7b94ec061949b08cca8fec2186614c8e436e7977607dfc3339e9
```

비용 진단의 후속 독립 검토에서 실제 학습 목적함수 문제를 발견했다.
RPS와 missing은 eligible 문항만 평균 내는데 trainer가 모든 chunk의 질문 수로
다시 가중해, prefix sharing·micro token budget·checkpoint 분할이 손실 비중을
바꾸고 있었다. 동일 logits의 Score/다른 유형 혼합 반례에서 수정 전 gradient의
최대 상대 차이0.144344/절대 차이0.014017을 재현했다. 과거 v2 성능 저하의
확정 원인이나 실제 모델 정확도 차이를 측정한 것은 아니다.

`61f3c6b` 수정은 전체 optimizer batch의 eligible 문항 수로 RPS·missing을 정규화하고
KL/replay 진단값도 각 global subset 평균으로 집계한다. Teacher/replay 목적함수,
NLL/Brier/pointer와 trace/proposal CE의 기존 전체 질문 분모는 유지한다.
CPU row descriptor로 분모를 계산하므로 새 CUDA host sync를 추가하지 않는다.
Standalone decision_loss 의미와 설정 LossWeights는 바꾸지 않는다.
13개 격리 regression과 실제 nonzero LoRA B/dropout0 native tiny 모델을 포함한
14개 검사가 통과했다. 실제 LoRA에서는 공유·selective checkpoint 경로의 loss와
모든 trainable gradient가 한 번의 전체 배치 reference와 허용 오차 내 일치한다.
Nonzero dropout의 stochastic noise 상관은 앞서 기록한 별개 제한이다.
최종 관련109개 CPU 검사도36.95초에 통과했고 Ruff308 source files의 lint/format이
통과했다. 독립 reviewer는 추가 P1/P2가 없다고 보고했다. Claude 승인은 별개다.

Trainer가 바뀌었으므로 위 native receipt를 새 코드에 그대로 적용하지 않는다.
최종 코드의 전체 CPU 검사와 같은 입력 plan의 native corpus 재준비·portable load를
새 디렉터리와 SHA로 다시 기록한다. 이전 결과와 오류 기록은 보존한다.

고정 `61f3c6b`에서 같은 input plan으로 actual native prepare를 다시 완료했다.
원본164608문항 전량 확인, 모든 split의 실제 입력 길이 audit, train rows·전체 epoch
schedule은 그대로다. 이전 control의 train_items/teacher_reads 및 모든 선택 split
payload bytes가 완전히 같다. Holdout은 내용 파싱 없이 opaque bytes만 비교했다.
Source anchor가 바뀌므로 manifest/receipt는 아래 새 값으로 고정한다.
새 portable load도 원본 source/encoder entrypoint 재실행 없이2368rows/992groups/
74batches/everyrow1visit을 확인했다. Local tokenizer identity 검사는 유지한다.
새 workload shape 검사에서233chunks+157prefixcalls=390backbonecalls 및 같은 토큰
집계를 재확인했다. 두 제어 run은 새 모델의 출력·성능 비교가 아니다.

```text
latest actual control: .dev/direct-policy-native-61f3c6b-20261005
bundle SHA: 170ece7c5e1b41dadd98051b26e1260e889c8033cc91c6658a5a087a7438a624
CPU audit SHA: 2c960994d18a20624fb8ca4854624daa24b9f9e40ed21855348d68447e3e1b7c
holdout SHA: f86e5420047cf2916a65a3fe79989ad765c40d51bad8aed3af1b6fd034a36b5a
selection SHA: 73b8e56813c0370f992dc07a170dda60928940b2f09d9982f7543b98b210ed2a
portable SHA: 3e1b93c92eb6fef9f09501fe895b65d88a9529b7234b44bcedf2b8257762f81f
workload shape SHA: 6fb531d71189dc90c1019bfcf93129a46c59ef7a8073720709bf30d8fba171e5
archive SHA: ea1df85d663735e6df702b94bb62c7f07caaaf14cd5c8c3ce892df2704455516
```

최종 `61f3c6b` 고정 전체 CPU 검사: **2110 passed, 2 skipped, 1 warning /
614.84초**, exit0/source변경0/stable_pass=True. Warning은 중복 ZIP member를
만드는 공격 fixture의 예상 UserWarning이다. Ruff lint/format은308 source files와
이 단위의 두 Markdown 문서를 확인했다. 전체 검사와 native prepare는 동시에
진행했으므로 CPU/GPU 속도 비교에 사용하지 않는다. 모든 자체 검증 process는 종료됐다.

```text
full receipt: .dev/codex-policy-corpus-full-61f3c6b-20261005.json
full log SHA: c9ef00eb9cc0f33172b843473c884e5ff3d72259130497e32b85a5333e5efe8a
```

남은 필수 조건은 prior private inventory, 실제 reasoned-teacher의 paired 이득,
complete native weights/CUDA parity·throughput, 전체 workflow credit admission과
matched v1 reasoning 대비 v2 off·독립 test이다. Policy corpus 준비는 진행됐으나
학습 직전 모든 조건 완료, 품질 향상이나 승격을 주장하지 않는다.

## 2026-10-05 실제 full weights와 CPU 업로드 준비

위의 complete native weights 조건은 이번 단계에서 실제 다운로드와 검증으로
진행했다. `google/gemma-4-12B-it`의 고정 revision 전체 safetensors
23,919,549,408 bytes가 공식 LFS SHA256과 일치한다. 설치된 HF loader와 같은
key 변환으로 666개 BF16 text tensor가 persistent state 667개를 충족하며,
생략된 native output은 진짜 tied embedding alias다. 별도 untied head를 입력
embedding으로 대체하지 않는다. Full weights와 실제 LoRA·판독 모듈의 보수적
합은 12,104,772,662 parameters다. Header·meta 검사는 runtime proof가 아니다.

새 `scripts/direct_v2/native_layout.py`와 upload 도구는 `ea48f3d`, `94f6005`로
따로 커밋했다. 기존 감사 Ayaka source, main과 Swift/API 소유 파일은 바꾸지 않았다.
소스·bytecode import, snapshot 변경과 output alias, compressed stream hash,
tar buffer의 숨은 payload, command/anchor binding 반례를 독립 검토로 확인하고
회귀 검사로 닫았다. 관련 37개 CPU 검사 및 312개 source의 lint/format이 통과했다.

실제 최종 zstd level 3 archive는 169개 파일 / 18,736,900,543 bytes다.
압축 전 payload 23,977,145,314 bytes 대비 21.855% 줄었다. 검증된 source,
개발용 학습 입력, receipts와 full native files만 포함하며 원본 test 입력,
raw source 및 HF download cache는 제외했다. 전체 archive 검증 후 새 디렉터리에
복원한 실제 12B/control도 HF offline·빈 cache에서 CPU audit가 통과했다.
Native bytes verified true, pretrained tensors loaded false, optimizer 0이다.

```text
archive: .dev/direct-policy-native-upload-final-61f3c6b-20261005.tar.zst
archive SHA: 6781202a7c0521f6cdef57b35944e8df6ea42f94283d30211da49118e1b9c9f2
native record SHA: 9b0a408a6f4f198e4002e455ec233bd6ab4882e18784b3affccdad1c6f48273e
layout report SHA: afc1c3a4e76fb2ae2511d2717ba2b5ff0c58e5100bd7d51d29de147b840dc0d1
restored audit SHA: 7c4f58da635e01e2f97eb42011640f9b9b4f810e6a228c25edf31f7d70dfcafa
```

전체 입력·절차는 [DIRECT_UPLOAD_PREPARATION.md](DIRECT_UPLOAD_PREPARATION.md)에
기록했다. Linux/CUDA runtime은 아직 포함하지 않았고 default action은 CPU audit다.
Actual teacher 이득, prior private inventory, CUDA kernels/throughput와 전체 비용
검증 및 matched v1 reasoning 대비 v2 off의 독립 품질 비교는 계속 남아 있다.
이번 단계의 GPU 사용과 paid API 호출은 0이다. Claude incoming은 section 18이
마지막이며 새 결과를 `.dev`로 공유했지만 승인을 추정하지 않는다.

최종 고정 `94f6005` 전체 CPU 검사는 **2147 passed, 2 skipped, 1 warning /
649.63초**다. Source와 fixture의 전후 hash 동일, exit0/stable_pass=True다.
첫 실행은 Git archive에서 ignored 개발 fixture가 빠져 7개 실패했고, 이전
검증본의 calibration/dev JSONL 두 개만 동일 해시로 복사한 뒤 통과했다.
실패 receipt는 보존했으며 코드를 바꾸거나 원본 test 입력을 가져오지 않았다.
Warning은 중복 ZIP member 공격 fixture의 예상 경고다.

```text
full receipt: .dev/codex-native-upload-full-94f6005-fixtures-20261005.json
full log SHA: 5c5ee1291800b05cb8e6c887bdedf13d61cb29472aa2f494b2d0d1bdd9db671c
```

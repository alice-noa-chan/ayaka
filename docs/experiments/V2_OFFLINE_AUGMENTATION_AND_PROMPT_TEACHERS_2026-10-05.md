# v2: 반사실 데이터 검사와 선택적 프롬프트 teacher 준비

작업 브랜치는 `ayaka-v2-experiments`다. 이번 변경은 학습 전에 CPU에서 검증할 수 있는
입력·정답·teacher 계약을 구현한다. 새로운 학습 성능, JevBench 순위, v2 off의 v1 on
우위는 측정하지 않았다. 유료 GPU·API 호출과 본 학습은 실행하지 않았다.

## 실험 순서에 대한 Claude의 추가 판단

Claude의 [저장된 Noul 관측 분해](SWIFT_NOUL_DECOMPOSITION_2026-10-05.md)는 cygnet의
이득이 자체 제작 Noul에 집중돼 있다고 보고했다. 자연 HotpotQA Noul에서는 policy CC가
같고, verified 절차형에서는 낮았다. 이를 반영해 **자연 문서의 gold-only direct 기준선이
먼저**다. 프롬프트 KL은 별도 선택 실험으로 준비하되 이 결과만으로 GPU 실행을 승격하지 않는다.
기존 cygnet serving 불채택 결론도 유지한다.

AUC와 accuracy는 다른 측정이다. 두 값의 차이를 정확한 인과 기여도 비율이나 증류 이득의
수학적 상한으로 해석하지 않는다. 이미 사용한 calibration/dev/hard 관측을 train으로
재라벨하지 않는다. 향후 teacher 이득은 authored와 natural을 분리해 검증한다.

## 제한된 반사실 편집과 독립 정답 검증

코드: [authored_counterfactual.py](../../ayaka/data/authored_counterfactual.py).
커밋 `fef115e`는 기존 v3 authored 문법에서 다음 10개 편집만 지원한다.

| 편집 | 보존하는 것 | 독립적으로 확인하는 것 |
|---|---|---|
| 윤년의 연도 | 질문·선택지 | Gregorian 달력 결과 |
| 사건 날짜 | 규칙 개정일·한도 | 사건이 적용받는 규칙 |
| 예외 override | 다른 조건·규칙 | 예외 판정 |
| credential 수준 | 다른 조건·규칙 | 임계값·예외 판정 |
| 금액 | 반올림 규칙 | 정확한 decimal cents |
| 시간대 offset | 현지 벽시계 시간 | UTC 날짜 |
| 영업일 종료 날짜 | 시작일·휴일 조건 | 명시된 weekdays 수 |
| 빨강·파랑 개수 | 확률 질문·선택지 | 계산한 확률 |
| 완료된 checks | 명시된 mechanical rubric | 충족한 checks 수 |
| 후보 순열 | ID별 설명·ordinal·정답 | 같은 ID별 분포 |

`flip`은 실제 질문의 gold 분포가 바뀌어야 한다. `preserve`는 그 분포가 유지된다는
뜻이며 문서 전체의 의미 동등성을 주장하지 않는다. 새 답이 원래 후보 집합에 없으면
거부한다. 자연 문서의 자유 편집, human rating 변경, 임의 paraphrase는 지원하지 않는다.

원본과 편집본 모두 schema와 independent gold verifier를 통과해야 한다. 원본 전체
payload digest, 편집 종류와 값으로 편집본을 재구성해 실제 payload와 비교한다. 원본이
없는 편집본, 연쇄 편집, no-op, 거짓 relation, 변조한 후보·정답·계보는 거부한다.
부모의 trace·teacher·proposal supervision은 복사하지 않는다.

모든 기존 계보 alias와 **검증된 사실 본문 SHA**를 전이적으로 연결한 뒤 역할 누수와
component 크기를 검사한다. 사실이 같아도 train/dev wrapper가 다를 수 있어서 원문
state의 문자열 비교만으로는 충분하지 않았다. 원본 row와 공통 source grouping 코드는
수정하지 않고 auditor의 복사본에서만 사실 alias를 추가한다.

독립 검토에서는 Decimal의 기본 precision 때문에 `32.494999… × 100`이 먼저 반올림돼
3250 cents가 되는 결함도 찾았다. [독립 verifier](../../ayaka/data/direct_verification.py)는
충분한 **local** precision으로 곱셈과 half-up을 계산한다. 정확한 결과는 3249다.
호출자의 전역 Decimal 설정을 바꾸지 않으며 serving 계산기를 추가하지 않는다.

CLI는 한 번 읽은 입력 bytes를 외부 SHA와 비교한 뒤 동일 bytes를 파싱한다.
결과 파일은 새 파일로만 쓰며 overwrite하지 않는다.

```powershell
.venv/Scripts/python.exe -m ayaka.data.authored_counterfactual `
  NEW_AUTHORED_COHORT.jsonl --expected-sha256 EXTERNAL_BYTE_SHA256 `
  --output NEW_AUDIT.json --max-component-rows DECLARED_CAP `
  --require-operation year=DECLARED_MINIMUM
```

이번 무료 mechanics 예시는 원본 10개와 편집본 14개, 총 24개다. 10개 operation 모두
포함하고 facts/lineage closure 후 8개 component, 최대 8 rows를 확인했다. 입력 SHA는
`1a8900c0289319b2d6dabe6ad10485ef83e299747b6fc2bc0d82775f9266cb46`다.
예시와 receipt는 ignored `.dev/counterfactual-mechanics-fef115e-20261005/`에 있다.
이 예시를 실제 train에 자동 삽입하지 않았다.

Receipt의 component weights는 권고일 뿐 trainer에 적용하지 않았다.
`weights_applied`, `context_ready`, `trace_supervision_ready`, `promotable`은 모두 false다.
문법 검증은 원본의 역사적 출처나 좋은 augmentation 분포를 증명하지 않는다.

## 풀이 없는 teacher를 기존 direct 학습에 연결

코드: [prompt_teachers.py](../../ayaka/training/prompt_teachers.py),
[direct_distillation.py](../../ayaka/training/direct_distillation.py),
[teacher_artifacts.py](../../ayaka/training/teacher_artifacts.py).
커밋 `cbb0525`는 `ayaka-prompt-context-teacher-1 / prompt_context`를 추가한다.

| 경로 | teacher 입력 | 학생 입력 | 생성한 풀이 |
|---|---|---|---|
| 기존 reasoned teacher | 완료된 풀이와 typed 판정 문맥 | 원본 direct 입력 | 기존 계약의 1–1024 tokens |
| 새 prompt-context teacher | 원본 질문의 cygnet 프롬프트 | 동일 원본의 min 프롬프트 | 0 |
| gold-only control | teacher 없음 | min 프롬프트 | 0 |

새 경로는 짧은 min과 긴 cygnet의 **실제 저장된 raw canonical logits** 두 종류를
요구한다. 관측 양측의 native model/tokenizer revision, adapter와 runtime은 같아야
하고 prompt variant만 달라야 한다. 원본 state, 질문, 후보·soft gold, 계보와 native
messages/input IDs/canonical IDs를 양측에서 재계산한다. 다른 transport, 잘린 문맥,
순서가 바뀐 canonical map, 다른 model·adapter, trace가 포함된 read는 거부한다.
현재 native single-pass canonical readout의 2–26 후보 범위를 유지한다.

Preparation에서도 두 입력을 다시 만들어 envelope와 비교한다. `false == 0` 같은
Python 값 비교를 피하기 위해 canonical fingerprint로 payload를 비교하고 확률 분포는
복사해 저장한다. 호출자가 원래 teacher payload를 나중에 수정해도 준비된 학습 target과
receipt binding은 바뀌지 않는다. version/route를 함께 legacy reasoned로 바꾸고 가짜
trace 필드를 덧붙이는 downgrade도 거부한다.

후속 커밋 `bea1a08`은 한 번만 순회할 수 있는 sample iterator도 먼저 materialize한다.
자료 목록을 만드는 순회가 뒤의 gold/native 검증 입력을 소모해 빈 teacher를 성공으로
보고하던 Python API 결함을 고쳤다. 최종 teacher ID 집합도 선언한 전체 cohort와
일치해야 한다. CLI의 list 입력은 영향을 받지 않았고, list/iterator의 결과와 검증 호출이
동일한지 회귀 검사한다.

학생 입력은 원래 min tokens를 유지하고, 기존 gold NLL·설정된 typed loss에 candidate
forward KL을 더한다. 풀이 CE나 cygnet의 긴 prompt tokens를 학생 입력에 넣지 않는다.
기존 NLL 이득·Brier 비퇴보와
typed credit, Score nMAE/RPS 검사를 유지한다. 이 기준을 통과하지 못한 teacher는
gold-only replay로 남는다. 기존 reasoned CLI 기본값과 정상 legacy receipt는 유지한다.

```powershell
.venv/Scripts/python.exe -m ayaka.training.teacher_artifacts `
  --teacher-kind prompt_context `
  --train TRAIN.jsonl --expected-train-sha256 EXTERNAL_TRAIN_BYTE_SHA256 `
  --config CONFIG.json --input-encoding MIN_ENCODING.json `
  --teacher-input-encoding CYGNET_ENCODING.json --teacher-identity IDENTITY.json `
  --direct-reads MIN_TRAIN.reads.jsonl --expected-direct-sha256 EXTERNAL_MIN_CONTENT_SHA256 `
  --prompt-reads CYGNET_TRAIN.reads.jsonl --expected-prompt-sha256 EXTERNAL_CYGNET_CONTENT_SHA256 `
  --native-path COMPLETE_OFFLINE_NATIVE_PACKAGE --out NEW_EXPORT
```

Read SHA는 ordered JSON contents의 `fingerprint`이며 train SHA는 literal file bytes다.
둘을 혼동하지 않는다. 전체 선언 train 관측 cohort의 양쪽 read가 필요하며 bundle의
나머지 train rows는 teacher 없이 사용할 수 있다. Export된 `teacher_reads.json`은 기존
direct bundle 준비·외부 manifest anchor를 이용한 재감사에 연결된다.

Usage는 저장된 양쪽 direct read의 입력·판정 토큰과 호출 수를 집계하고
`reasoning_tokens=0`을 기록한다. 저장되지 않은 실패 호출 비용은 별도 collection
usage log가 필요하다. SHA와 native 재구성은 **content consistency**를 확인하며 명명한
weight가 실제 backend에서 실행됐다는 증명은 아니다. 모든 receipt의
`execution_attested`와 `promotable`은 false다.

이 준비는 OPCD의 on-policy generation/reverse KL 재현이 아니다. 긴 context의 효과를
학생에 옮겨보는 finite-label forward KL 절제다. 실제 train teacher 관측은 아직 없다.

## 검증과 남은 조건

각 코드 단위는 코드·테스트 수정 → Ruff lint 수정 → format → lint/format 재검사 →
관련 테스트 → 명시적 diff 검토 → 상세 본문이 있는 별도 commit 순으로 처리했다.

- `fef115e`: counterfactual, independent gold, source groups 관련 **97 passed**.
- `cbb0525`: 새 prompt teacher, 기존 distillation/converter/bundle 관련 **101 passed**.
- `bea1a08`: iterator 회귀 수정 후 같은 4개 모듈 **102 passed**.
- Independent source reviewer가 facts 누수·Decimal 경계와 teacher downgrade·bool/int
  혼동·mutable alias를 찾아 수정했다. 후속 iterator 결함도 수정했고, 독립 검토자는
  최종 source에서 남은 P1/P2는 없다고 보고했다.
- 새 teacher 테스트는 실제 random native CPU gather/backward, 비대칭 soft Score 좌표,
  양쪽 canonical-map 공격, 학생은 들어가지만 teacher만 넘는 문맥 경계, offline CLI,
  bundle 저장 후 재감사를 검사한다. Synthetic holdout의 지정 경로에 대한
  `Path.read_bytes` guard를 검사한 것이며 모든 파일 접근을 가로채는 증명은 아니다.
- 기본 시스템 Python에는 peft가 없었다. 기존 project `.venv`의 peft 0.21.2 /
  transformers 5.18.0으로 관련 검사를 다시 통과했다. 의존성 설치·변경은 하지 않았다.

최종 코드 source는 `bea1a0818cf68f3a6bfa75a1b78f5998bc870b0c`로 고정했다.
공유 working tree의 후속 변경이 검증 결과에 섞이지 않도록 `git archive`를 풀어 검사한다.
Archive SHA는 `8e9fd0d39a0089db8760f0272348c34aedd87fe8a14c81cf8db58b1900dd242a`다.
Linux에서는 관련 7개 모듈 **199 passed / 67.72s**를 확인했고 wrapper는 79.59s였다.
332개 source 파일의 실행 전후 SHA가 같고 archive와도 일치한다. 전체 Windows CPU 검사는
**2367 passed / 2 skipped / 958.76s**이며 wrapper는 961.51s다. Warning 1개는 중복 ZIP
member 거부 테스트가 만드는 의도된 fixture 경고다. Windows receipt의 335개 항목은
source 332개, `pyproject.toml`, 기존 development fixture 2개다. Source와 pyproject의
SHA는 archive에, fixture SHA는 별도 고정 anchor에 대조했다. 두 receipt의 실행 전후
hash와 log digest를 확인했고 모두 `stable_pass=true`다.

| 최종 source 검증 | receipt SHA256 | log SHA256 |
|---|---|---|
| Windows CPU 전체 | `3d677a13464e9678e5a2140459f5aa1c3d38110e9fe2919dbe9ccbcd0c80a949` | `921710430c5f55993ccdab7561b149bbd60ef48954ab4e67ed8ed8d8227f5add` |
| Linux CPU 7개 모듈 | `caa8d43cf81eae2226d3f1f0fe05253ce7086e16edf0f2d5d1f76f28baff0cb9` | `14cd148e7a716804faff853b8f8b37cd4910cffda65b1c8406ed57a60f9a06fa` |

로컬 receipt는 ignored `.dev/codex-pretraining-{full,linux}-bea1a08-20261005.json`과
각각 같은 이름의 `.log`에 보존한다. 앞선 `cbb0525` 전체 검사는 2366 passed /
2 skipped였지만, 이를 iterator 수정 뒤의 최종 source 검증으로 대체해 표기하지 않는다.
현재 private holdout 원본은 열거나 복사하지 않았다. 이전에 사용한 development fixture
2개의 anchor는 calibration `6f4c2ee35e8fd52ae58184cc8f7002ec26b76c7e563e4fbd05506129489dc1b9`,
dev `1cc485dfe13388040fd5ed966e51c2804d20e7e01bcc602711aa59f9fc1b258f`다.

남은 성능 조건은 실제 자연 gold-only 학습·같은 native readout의 adapter on/off 비교,
fresh 독립 평가, 실제 train teacher 수집을 선택할 근거, 그리고 같은 질문·환경의
**v1 on ↔ v2 off** 비교다. 오래된 private 평가 원본 inventory를 아직 찾지 못했으므로
역사적 평가 전체와의 무중복까지 증명하지 않는다. DSS rationale CE의 별도 task mask와
단계 검증, ICoT 다단계 학습도 이번 구현에 포함하지 않았다.

본 학습 시간·비용은 고정 corpus/schedule과 선택 GPU의 실제 throughput 관측으로
확정해야 한다. 이번 CPU fixtures의 시간이나 논문 결과를 본 학습 속도·성능으로
외삽하지 않는다. 공개 v1 main과 Claude 소유 Swift 파일은 수정하지 않았다.

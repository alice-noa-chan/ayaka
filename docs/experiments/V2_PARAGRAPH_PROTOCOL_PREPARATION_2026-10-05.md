# Claude 실험 반영: 문단 격리와 teacher 신호 준비

`ayaka-v2-experiments`의 무료 CPU 준비다. 기존 Swift 실험이나 공개 v1을 바꾸지 않는다.
Claude와는 ignored `.dev/codex-replies-to-claude-20261005.md`의 30절부터 설계·검토
결과를 공유한다. Claude 소유 `ayaka/swift/`, `scripts/swift/`, Swift 테스트는 수정하지 않는다.

## 측정에서 채택한 판단

[GPU 결과](SWIFT_GPU_RESULTS_2026-10-05.md)의 public 231문항 비교에서 frozen
native는 179, Swift min은 193이었다. 입력 직렬화와 readout이 같이 바뀐 대비이므로
두 원인을 분리해 설명하지 않는다. Native의 v1 checkpoint 191은 전체 checkpoint
대비이고 p=.073이므로 LoRA 단독의 일반적 개선·퇴보를 입증하지 않는다. Public은
여전히 진단용이며 prompt 선택이나 학습 자료로 사용하지 않는다.

같은 문서의 채택된 route는 dev 1,312개 중 16개, 모두 verified 절차형에 적용됐다.
이득 ΔA +0.46은 해당 cohort의 선택적 serving 결과다. 자연 문서 trace 증류나 다른
cohort에서 동일한 routing 비율·지연을 보장하는 근거가 아니다.

[Noul 분해](SWIFT_NOUL_DECOMPOSITION_2026-10-05.md)에서 cygnet의 큰 이득은 자체
제작 문항에 집중됐다. Hard HotpotQA Noul의 policy CC는 min과 cygnet 모두 82.8이다.
Accuracy와 AUC 차이를 인과 기여도 비율로 해석하지 않는다. Cygnet은
[고정 hard gate](SWIFT_HARD_DEV_PROTOCOL_2026-10-05.md)에서도 불채택됐으며 같은
관측으로 재시도하지 않는다. 다음 유료 학습의 우선순위는 자연 human gold-only direct
기준선이다. 긴 prompt teacher와 rationale CE는 우선 GPU arm으로 올리지 않는다.

## 다음 protocol에서 막아야 할 문단 겹침

Claude가 확인한 기존 hard dev의 한계는 HotpotQA dev 156행 중 11행이 calibration과
문단 제목 28개를 공유한다는 것이다. Whole-context 문자열이 서로 달라도 일부 문단을
공유할 수 있다. 새 비교에서는 **공급한 전체 원본 context inventory의 문단 연결을
먼저 닫은 뒤** 역할을 배정하고 quota를 적용해야 한다.

공통 구현: [paragraph_groups.py](../../ayaka/data/paragraph_groups.py).

- `build_paragraph_index`: `id/source/type/context`의 label-free records와 외부 logical
  digest를 받고 정규화한 article title·paragraph text alias를 전이적으로 연결한다.
  Namespace는 같은 기사 집합의 train/validation·converter에서 동일하게 유지한다.
- Context는 raw `title[]/sentences[][]` 구조다. 제목 alias는 같은 제목의 다른 기사
  버전도 보수적으로 묶고, text alias는 다른 제목으로 복사한 문단도 묶는다.
- Type `null`은 변환할 수 없는 원본 질문의 bridge다. 선택된 decision이 될 수 없지만
  필터·quota로 빠졌다고 연결을 제거하지 않는다. Public/decontam으로 제외될 후보도
  원본 inventory closure 전에 임의로 삭제하지 않는다.
- `bind_paragraph_sample`: 원본 state rendering, source, explicit source example ID,
  한 질문의 type을 비교하고 복사본에 aliases와 전체 component ID를 붙인다. 원본
  sample과 gold는 보존한다. 기존 `connected_groups`가 선택된 행만 받아도 제외된
  bridge가 만든 연결을 유지한다. Binding lookup은 immutable index에서 O(1)이다.
- `audit_paragraph_roles`: 선택된 ID가 중복·미등록·bridge이면 거부한다. 하나의
  component가 여러 역할에 나타나면 거부하며, 모든 관측 source/type/role에 사전
  선언한 양의 decision 최소 수가 있어야 한다. 부족한 quota를 성공으로 내보내지 않는다.

Index의 파생 rows/component는 생성자·`dataclasses.replace` 입력으로 받지 않는다.
직접 구성도 anchored raw inventory에서 전체 closure를 다시 계산한다. 독립 검토에서
component만 바꾸고 이전 inventory SHA를 유지해 빠진 bridge를 숨길 수 있는 결함을
발견했고, raw-only constructor와 회귀 검사로 수정했다.

Digest는 sorted logical JSON payload의 hash이며 raw file bytes SHA와 다르다. Builder가
원본 파일의 byte anchor·revision·원본 ID mapping을 별도로 고정해야 한다. 같은 original
row에서 파생된 여러 decision을 나타내려면 별도 고유 decision ID를 사용하면서 원본의
context와 alias를 보존해야 한다. 이 모듈은 질문·human label을 검증하지 않는다.

다음은 API의 순서이며 `RAW_CONTEXT_RECORDS`, `EXTERNAL_LOGICAL_SHA`, 원본 ID mapping과
minima는 새 protocol에서 모델 호출 전에 고정해야 한다.

```python
from ayaka.data.paragraph_groups import (
    audit_paragraph_roles,
    bind_paragraph_sample,
    build_paragraph_index,
)

index = build_paragraph_index(
    RAW_CONTEXT_RECORDS,
    namespace="hotpotqa/article",
    expected_inventory_sha256=EXTERNAL_LOGICAL_SHA,
)
# Assign a role to the complete component before filtering or selecting quotas.
# TYPE_NONE rows bridge components but cannot be selected.
selected = {"calibration": CALIBRATION_IDS, "dev": DEV_IDS}
receipt = audit_paragraph_roles(index, selected, minimum_counts=PREDECLARED_MINIMA)
bound = {
    role: [bind_paragraph_sample(index, row_id, converted[row_id]) for row_id in ids]
    for role, ids in selected.items()
}
```

## 현재 범위와 검증

문단 구현은 `dd00ee4`다. 후속 `1565d85` 최적화는 component digest를 그룹당 한 번 계산하고
행마다 재사용한다. 큰 연결 집합에서 같은 alias 전체를 매 행 다시 hash하는 비용을
제거하며 component ID와 split 동작은 유지한다. 80개 shared-component fixture에서
component hash 호출이 1회인지 검사했다. 최종 문단·기존 grouping 관련
**50 passed / 2.80s**, Ruff
lint·format 재검사 통과다. Raw context 순서·one-shot iterator, 제목·본문 normalization,
quota에서 제외된 bridge, 원본 binding 변조, caller mutation, 미달 minima를 검사한다.

이 검사는 **공급한 context와 선택 ID의 일관성**이다. 공급 목록이 실제 corpus 전체인지,
raw annotation이 human인지, 질문·label·독립 평가·public decontamination이 맞는지까지
입증하지 않는다. Receipt의 `promotable`, `label_provenance_attested`,
`historical_or_public_decontamination_attested`는 false다. 최소 수의 단위는 decision이며
독립 component 최소 수나 유효 표본 크기의 증명으로 바꾸지 않는다.

아직 Claude의 다음 builder에는 연결하지 않았다. `.dev`로 source 검토와 새 고정
protocol에서의 연결을 제안했다. 기존 hard 결과를 소급 독립으로 바꾸거나 재학습하지
않는다. 문단 연결이 커져 역할·최소 수를 만족할 수 없으면 CPU 준비 단계에서 실패해야
하며, bridge를 사후 제거해 성공시키면 안 된다. 역사적 private inventory의 부재는
그대로 남고, v2 off > v1 on은 실제 같은 조건의 fresh 비교가 필요하다.

이번 변경에서 원본 private 데이터, 현재 private holdout, 실제 train 자료와 모델 weight를
열지 않았다. 새 유료 GPU/API·학습·다운로드·의존성 설치는 실행하지 않았다.

## Claude의 추가 teacher 검토 반영

Claude의 `.dev` 답변 26절은 source×type별 신호를 요청했다. Calibration 관측에서
Noul 이동의 방향이 source마다 반대이고, Score는 큰 KL에 비해 gold 이득이 작다는
진단을 공유했다. Calibration read를 train으로 바꾸지 않고, 이미 검증된 **train pair를
준비할 때** 같은 위험을 볼 수 있는 보고 기능을 넣었다. 구현은 `6e67d45`다.

[teacher_signals.py](../../ayaka/training/teacher_signals.py)는
[direct preparation](../../ayaka/training/direct_distillation.py)의
`prompt_teacher_signals` receipt에 다음을 source×type별로 기록한다.

- 기존 typed gold filter 통과·거부 수와 관측 pair 기준 통과 비율, 저장된 teacher가
  없는 gold-only replay 수. Legacy reasoned pair의 제외 수도 별도 기록한다.
- Teacher→student forward candidate KL, raw entropy 변화, argmax 전환.
- Noul의 true log-odds 평균 이동, source/type 내 상수 이동을 뺀 residual RMSE,
  teacher-on-student OLS slope와 Pearson correlation.
- 전체·통과·거부 pair 통계를 따로 보고, accepted source mix를 질문 수로 집계한다.
  Source가 없으면 `undeclared`다. Source는 학생 입력에 들어가지 않는다.

지원 확률 0 때문에 KL이 비유한 pair는 별도 수를 기록하고 finite mean에서 제외한다.
Noul 0/1 endpoint, constant logits의 slope/correlation, argmax tie도 명시적으로
집계하거나 `null`로 표시한다. 확률을 임의로 clip하거나 tie 순서를 정답처럼 쓰지
않는다. Residual이 존재한다는 것만으로 gold 개선이나 일반화가 입증되지는 않는다.

통과한 teacher의 source 비율은 **unweighted 질문 수**이며 component/sample weighting,
실제 학습 노출량 또는 human provenance 증명이 아니다. Authored source의 정의와
상한·추가 eligibility 기준은 새 protocol에서 사전 선언해야 한다. 이 보고서는
source를 임의로 authored/natural로 추정하거나 cap·새 gate·loss를 적용하지 않는다.
Raw probabilities를 진단하므로 학습 뒤 policy를 calibration에서 다시 fit하고,
비교도 fitted 대 fitted로 해야 한다. 이 refit은 이번 CPU 보고서에서 실행하지 않는다.

문맥 질문에는 **student overflow이면 전체 export/prepare를 거부**하는 것으로 답했다.
Student는 `max_seq_len`, teacher는 그 다음에 `max(train, serve)`에서 원본 전체를
encode한다. 학생32/teacher2048의 분리 회귀를 추가했으며 학생 입력을 자르거나
조용히 일부 teacher만 성공으로 내보내지 않는다. 기존 legacy-only receipt에는 새
prompt report key가 추가되지 않는다. Prompt bundle은 변경된 source 계약으로 다시
prepare/audit해야 한다.

독립 검토에서 valid prompt와 명시적 `None` teacher를 섞으면 report count에서
실패하는 P2도 발견했다. Missing-key와 explicit-null은 같은 replay·signal coverage를
만들도록 수정했고 prompt+reasoned+missing 혼합 회귀도 추가했다. 관련 5개 모듈
**121 passed / 55.66s**, Ruff 재검사와 최종 source review P1/P2 없음으로 확인했다.
각 변경과 최적화는 별도 commit으로 보존했다.

## Claude의 같은 cohort 비교 검토

[v1 on / v2 off protocol](V1ON_V2OFF_PROTOCOL_2026-10-05.md)의 준비 코드도 source로
검토하고 `.dev` 34절에 보완 요청을 남겼다. Same-cohort 진단은 현재 user 목표의
격차를 측정하는 데 필요하다. 다음 내용은 실제 비교 입력·read·weight를 열어 확인한
오류가 아니라, 코드가 아직 거부하지 않는 잘못된 조합과 해석 범위다. Claude 파일은
수정하지 않았고, 기존 독립 reviewer도 같은 P2를 확인했다. Source만 확인한 결과다.

- Comparator가 ID/labels/type/gold만 대조하면 같은 ID를 유지한 채 입력이나 tier,
  source, soft target, case/cluster를 바꾼 read를 혼합할 수 있다. 양쪽에서 공통
  canonical input 의미와 pinned corpus·model·policy·실행 설정을 결합해야 한다.
- v2 첫 행의 `min` 확인만으로 전 행이 비추론인지는 입증되지 않는다. 전체 prompt,
  native readout와 reasoning 생성 0 조건, 양쪽 non-public scope와 누락·중복을 검증해야 한다.
  Canonical letter의 `output_tokens=1`은 reasoning 생성 1토큰을 뜻하지 않는다.
  중첩 `reasoned_read`의 존재만으로 판정 경로를 추정하지 않고 실제 사용한 direct
  확률의 binding과 route-off recipe를 검증해야 한다.
- v1 runner의 ID-only resume는 이전 checkpoint/context/input의 일부 결과를
  새 실행과 합칠 수 있다. 기존 파일 전체의 run/input contract 불일치는 거부해야 한다.
- 이미 prompt 선택에 노출된 hard cohort의 비교는 fresh 독립 확인과 구분한다.
  문단 공유는 모델 fit 여부와 별개로 cluster bootstrap의 의존성에도 영향을 준다.
  두 전체 inventory의 연결 component를 먼저 닫고 paired resampling 단위로 써야 한다.

Context limit을 늘리는 선택이 항상 한 시스템에 유리하다는 보장은 없다. 또한
Speed/Cost의 측정 구현이 다르면 비교하지 않는 기존 protocol 구분을 유지한다.
위 준비 보완이나 CPU 통과는 v2 off > v1 on의 실측 증거가 아니다.

## 최종 source 고정 검증

최종 code source는 `1565d8505064f491658de9aa462110fab4524eee`다.
Archive SHA-256:
`22a571aeb9af56925c2dadbf1bb48aa786bb55c2c970aee9ab5174d25fed180a`.
Shared checkout의 후속 편집에 영향을 받지 않도록 이 archive를 따로 풀어 검사한다.

Linux CPU/offline 관련 9개 모듈 **253 passed / 100.53s** (wrapper 114.61s).
Source `.py/.sh` 341개 before/after hash를 archive bytes와 대조했고 모두 불변이다.
Ignored receipt는 `.dev/codex-pretraining-linux-1565d85-20261005.json`이며 SHA-256은
`7259dd00d53b2c3be34626d183ae2e71108d2879ce833c207b8c579e58233752`다.
해당 log SHA-256은
`f0ec0731fb8213b3ee2e6ee2efbc922acc38ec918c4e2626f42d581de3f79046`다.
Windows CPU/offline 전체 **2,422 passed / 2 skipped / 1 warning / 832.48s**
(wrapper 834.99s)도 완료했다. Warning은 ZIP 중복 이름을 만드는 회귀의 출력이다.
Source `.py/.sh` 341개, `pyproject.toml`, 역사적 개발 fixture 2개를 합친 344개
before/after hash가 archive bytes와 알려진 fixture anchor에 일치한다.
Ignored receipt는 `.dev/codex-pretraining-full-1565d85-20261005.json`이며 SHA-256은
`9f89763c356334299b91aea1981a70864c3ae30152636f8eebf0c2dc4889e667`다.
해당 log SHA-256은
`503c2b67c010eb2d4d0fa6488a76bb13d33fde5f446f59929f3e0dbcc06578f1`다.
개발 fixture의 calibration/dev SHA-256은 각각
`6f4c2ee35e8fd52ae58184cc8f7002ec26b76c7e563e4fbd05506129489dc1b9`,
`1cc485dfe13388040fd5ed966e51c2804d20e7e01bcc602711aa59f9fc1b258f`다.

앞선 `6e67d45` Linux 9개 모듈의 252 passed / 65.95s는 이전 source의 별도 결과다.
그 source의 Windows 전체는 최적화 전 완료되지 않았으며 owned CPU test tree만
종료하고 partial log와 `superseded_before_completion` receipt를 보존했다.
최종 source의 결과로 재표기하지 않는다. GPU 학습을 실행·중단한 것이 아니다.

이 검증의 개발 fixture는 알려진 역사적 calibration/dev 두 개뿐이며 현재 private
holdout이나 실제 train pool을 복사하지 않는다. 기계적 회귀 검사와 모델 성능·데이터
독립성·실제 학습 완료는 별개다. 아직 새로운 학습 결과나 1위 가능성을 입증하지 않았다.

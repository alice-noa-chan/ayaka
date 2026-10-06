# v2 학습 전 조건을 실제 실행 경로에 연결

Claude 답변 27–28절과 앞선 교차 검토의 미해결 조건을 후속 구현했다.
작업 브랜치는 `ayaka-v2-experiments`이며 공개 v1/main은 보존한다.
협업 교환은 `.dev/codex-replies-to-claude-20261005.md` 44절 이후에 있다.
기존 Claude 소유 `ayaka/swift/`와 `scripts/swift/`를 수정하지 않고,
검증을 실제로 소비하는 새 `scripts/direct_v2/` 진입점을 제공한다.

## 해결한 구현 조건

| 조건 | 새 실행 경로의 검사 | 범위 |
|---|---|---|
| 채점 필드가 nested binding과 달라질 수 있음 | 전체 gold/soft gold/tier/source/type/labels/case와 원본 문항을 정확히 연결 | 고정 노출 cohort |
| frozen policy를 truthiness로 검사 | 전체 frozen v1 policy와 run contract를 fingerprint로 정확히 비교 | 값·형식 일치 |
| checkpoint 선언과 실제 소비가 다를 수 있음 | config/meta/head/전체 adapter 파일의 존재와 bytes를 외부 inventory와 비교하고 검증한 독립 복사본만 load | 실제 소비 snapshot |
| Swift record 내부의 logits/probabilities 불일치 | 기존 strict validator + canonical letter 순서·정렬·runtime·pass·messages·사용량 검사 | 저장 record의 내부 일관성 |
| 문맥 잘림 또는 재개 행의 문맥 변경 | 전체 v1 원문·extraction·실제 풀이가 붙은 판정을 별도 검사하고 재개 시 CPU 입력과 비교 | 조용한 clipping 거부 |
| GPU 시작 후 hard 행 누락·긴 procedural 입력 발견 | exact hard calibration/dev coverage, 전체 pending Swift 입력을 native tokenizer로 먼저 검사 | GPU 시작 전 CPU gate |
| 이전 결과·다른 서버를 새 실행으로 오인 | fresh OUT, 기존 port 점유 거부, 직접 띄운 process 감시, checked comparison 성공 뒤 DONE | 새 job 실행 순서 |
| 문단 helper가 실제 생성기에 미연결 | 전체 supplied raw context closure → component별 role → exclusion/input 검사 → quota → canonical 출력 | 새 opt-in Hotpot builder |
| 많은 종속 질문으로 독립 표본 수 부풀림 | decision과 distinct component minima를 함께 검사하며 component queue에서 교대로 선택 | 공급된 문단 관계의 의존 단위 |
| public 검사 데이터가 비거나 해석되지 않음 | 외부 byte-pinned public manifest, 정확한 정책·형식, 비어 있거나 효과 없는 corpus 거부 | 제공한 공개 inventory |
| 준비 중 tokenizer 변경 | entry/exit native serialization과 recipe 비교, 변경되면 출력 폐기 | 한 CPU 준비 scope |

첫 계약 단위는 `a4aeaff`, checkpoint/context 실행 단위는 `577380f`,
GPU 전 준비·job 연결은 `fedfbf4`, raw builder는 `ab45d7f`다.

## 같은 cohort의 v1 on / v2 off 비교

[원래 protocol](V1ON_V2OFF_PROTOCOL_2026-10-05.md)의 cohort·성공 기준·v1 8192
문맥·384-token gate 및 Swift min direct 설정을 바꾸지 않는다. Production policy는
외부 byte SHA로 고정하고 route만 제거한다. Fitting input/source SHA는 필수 선언이며,
실제 fitting을 재현했다는 증명이 아니다. 이 비교는 기존 선택에 노출된 문항을 사용하며
fresh 독립 성능 확인으로 승격하지 않는다. Speed/Cost 비교도 추가하지 않는다.

[prepare_matched.py](../../scripts/direct_v2/prepare_matched.py)는 완전한 consumed-file
inventory를 가진 로컬 checkpoint receipt, policy, fit 선언의 외부 SHA를 받아 protocol을
준비한다. 기존 receipt가 `meta.json` 등 실제 소비 파일을 빠뜨리면 실패한다. 누락된
해시를 추측하거나 이 문서의 예시 값으로 채우지 않는다. Protocol은 v1 관측 전에 고정한다.

[matched_preflight.py](../../scripts/direct_v2/matched_preflight.py)는 저장된 hard 두 파일이
원본 hard cohort 전체와 정확히 일치하는지 검사한다. 미수집 procedural 입력은 pinned
offline tokenizer의 실제 chat template, min/pretty/non-thinking, canonical answer prefix로
인코딩한다. 전체 입력과 판독 1 position이 16384를 초과하면 GPU 수집을 거부한다.
현재 collection 구현의 raw bytes도 기존 세 Swift source SHA와 일치해야 한다.
Windows checkout의 CRLF를 새 collection SHA로 간주하지 않는다. 배포에는 고정 Git
archive의 LF source를 사용한다.

[matched_v1.py](../../scripts/direct_v2/matched_v1.py)의 `--preflight-only`는 모델 weights를
load하거나 forward하지 않는다. Checkpoint 파일은 bytes 검증을 위해 읽고 복사하며,
그 복사본의 config/adapter/legacy input contract를 확인한다. 원문 판정의 전체 입력은
8192 안에, extraction 입력은 별도 12288 안에서 384-token reserve를 확보해야 한다.
실제 풀이가 붙은 판정도 모델 호출 직전 같은 전체 입력 검사를 거친다. 실행 시 model
loader는 검증한 snapshot만 `merge=False`, offline, strict loading으로 소비한다.

실제 채점은 [matched_compare.py](../../scripts/direct_v2/matched_compare.py)가 한다.
전체 checked execution receipt와 그 외부 byte SHA, 동일 protocol SHA, 전체 행의 논리
digest와 context preflight가 필요하다. 자체 서명한 row hash만으로는 성공하지 않는다.
질문 순서와 canonical letter 순서의 의미 좌표도 검사한다. 생성/성능/origin의 외부
인증은 이 digest 검사로 대신하지 않는다.

[matched_job.sh](../../scripts/direct_v2/matched_job.sh)는 미래 GPU 실행 순서를 연결한다.
사전 준비한 `IN`에 protocol/checkpoint/policy JSON, procedural/hard JSONL, 기존 hard reads를
둔다. 부모 폴더가 존재하는 새 `OUT`과 사전 고정한 `PROTOCOL_SHA256`을 사용한다.
CPU 두 gate → vLLM procedural 수집 → v1 checked 실행 → checked 비교 → DONE 순이다.
모든 model/tokenizer 접근은 offline이다. 새 source bytes가 기존 receipt와 다르거나 어떤
gate라도 실패하면 성공 DONE을 기록하지 않는다. **이 job을 이번 작업에서 실행하지 않았다.**

## 다음 원본 문단 기반 direct 데이터 준비

[paragraph_split.py](../../ayaka/data/paragraph_split.py)는 원본 structured Hotpot rows를
끝까지 읽어 context 연결을 먼저 닫는다. Unsupported span 답변·선택 안 된 bridge·입력
overflow 행의 문단 관계도 유지한다. Whole-component role은 seed와 component key로
정하고 정답이나 모델 결과를 사용하지 않는다. 사전 role/source/type quota의 decision
수와 서로 다른 component 수를 모두 채울 수 없으면 출력을 만들지 않는다.

원본 human yes/no, 또는 comparison 질문에서 두 entity title 중 하나를 답한 정답을
existing converter로 변환한다. 임의의 span 답을 supporting-title 과제로 바꾸지 않는다.
공개 overlap은 unsupported 질문까지 원본 질문·제목·본문으로 검사하고 전체 component를
제외한다. `blocked_raw_ids`는 공급한 전체 raw inventory 안에 실제로 존재해야 한다.
알려진 과거 ID의 원본 bridge context도 inventory에 포함해야 그 연결을 제외할 수 있다.
Direct encoder는 `ayaka_segmented` 또는 명시한 `swift_canonical`이며, 원문 전체가
선언한 context limit에 맞지 않으면 해당 decision을 제외한다. Gold-only direct 준비
파이프라인이 읽는 canonical Sample JSONL과 감사 manifest를 출력한다.

CLI 필수 입력은 `--raw/--raw-sha256`, `--plan/--plan-sha256`,
`--public-manifest/--public-manifest-sha256`, `--public-root`, 새 `--output`이다.
Plan은 version/seed/namespace/source/hf_split/license/roles/minimum_counts/
minimum_components/input_encoding/context_limit/blocked_raw_ids를 정확히 선언한다.
문항·component minima는 `role/source/type` 양의 정수이고 동일 셀을 포함해야 한다.
이 Hotpot converter의 출력 유형은 Choice/Noul이다. Score 등 다른 원본의 독립 split을
이 builder로 만들었다고 주장하지 않는다.

Public manifest version은 `ayaka-public-exclusion-1`, `format`은
`jevbench-singular-question`, `policy`는 `ayaka.data.decontam.POLICY` 전체 값,
`files_sha256`은 제공한 public JSONL 파일별 외부 SHA다. Manifest 자체의 byte SHA도
필수다. 각 public row는 state와 단일 question의 텍스트 instruction, 제공하면 텍스트
criteria를 가져야 한다. `questions[]` 형식이나 비텍스트 candidate object를 조용히
일부만 색인하지 않고 거부한다. 제공한 corpus가 실제 public 전체인지, sealed 문항이나
미상 과거 데이터까지 제외했는지는 이 형식·해시 검사로 증명되지 않는다.

Raw/plan/public/source 선언과 새 minima는 관측 뒤 성공하도록 바꿀 수 있는 선택 기준이
아니다. Receipt의 `promotable`, `historical_inventory_complete`,
`fresh_independence_attested`는 모두 false다. 새 원본 corpus·license의 실제 적절성,
누락된 과거 목록 복구와 별도 사례 의존 관계 검토가 필요하다. 기존 exposed cohort를
사후 재분할하거나 새 독립 test라고 이름을 바꾸지 않았다.

## 최종 검증과 남은 실제 증거

각 단위는 코드·회귀 수정 → Ruff fix → format → lint/format 재검사 → 관련 CPU
테스트 → 명시적 diff/독립 source review → 상세 본문 별도 commit 순으로 처리했다.
관련 7개 모듈은 **194 passed / 14.44s**, 마지막 public 형식 수정 뒤 paragraph/source
관련 3개 모듈은 **92 passed / 9.31s**였다. Bash syntax와 실제 Bash stub의 old OUT/
occupied endpoint 거부도 확인했다. 독립 최종 source/doc review의 P1/P2는 없었다.

마지막 기능 source는 `ab45d7f9a20537cecdb51b24dfedad1413be3d24`, archive SHA-256은
`ad027ac6bd058eab604742b37f0a36c96e5b4f869989a89c75914b7685e01f10`이다.
고정 source Linux CPU/offline 관련 **12개 모듈 371 passed / 71.59s**
(wrapper 82.97s)로 완료했고 source `.py/.sh` 354개 전후가 archive bytes와 일치했다.
Ignored receipt `.dev/codex-pretraining-linux-ab45d7f-20261006.json`의 SHA-256은
`edb28237c5398ceffc2e53157640366224ca4db905320b1c7fbce198b035787a`,
log SHA-256은 `c88ee281f6612d2105bdd7ee898c351629a05f003e69d3e5d83320accb1103da`다.

첫 Windows 전체 검사(ab45d7f)는 **2573 passed / 2 skipped / 2 warnings / 1050.21s**였다.
ZIP 중복 이름 거부 회귀의 예상 warning 외에, 새 occupied-endpoint 셸 테스트의 UTF-8
child 출력을 cp949로 읽는 reader-thread warning이 하나 있었다. 이를 숨기지 않고
test-only `ff4cc93`에서 child/parent 인코딩을 UTF-8로 일치시키고 거부 사유 assertion도
강화했다. 해당 모듈 **11 passed / 12.63s**를 reader-thread warning을 오류로 처리하며
재검증했고, 독립 review P1/P2 없음으로 별도 commit했다. 앞선 receipt/log는 보존했다.

최종 test 수정 source는 `ff4cc935d9917c9c02e6212fa6b50632753f42bc`, archive SHA-256은
`b2656c26aa08b344e0f1cacef97fc3cbbe8041fbecd4468528a60a718cdb9a03`이다.
동일 frozen source의 Linux 변경 모듈은 **11 passed / 10.11s** (wrapper 20.27s)로
완료했다. Reader-thread warning을 오류로 처리했고 354개 source before/after는 새
archive bytes와 일치한다. Receipt `.dev/codex-pretraining-linux-ff4cc93-20261006.json`
SHA-256은 `e48883bf0e03b1694f6634ecff0b2b1d433ed3d7b289665d9a557dd8e35c2372`,
log SHA-256은 `c727e05e3f1bed0eb88acab9316e43dc7e4586d7743bf78889aa740c0af44376`다.

최종 frozen ff4cc93 Windows CPU/offline 전체는 **2573 passed / 2 skipped / 1 warning /
1046.66s** (wrapper 1049.92s)로 완료했다. Reader-thread warning을 오류로 처리한
검사에서 인코딩 warning은 재발하지 않았다. 남은 하나는 중복 ZIP member를 만드는
기존 거부 회귀의 예상 출력이다. Source `.py/.sh` 354개와 `pyproject.toml`, 역사적
개발 fixture 두 개의 총 357개 before/after가 archive/fixture anchor와 일치한다.
Receipt `.dev/codex-pretraining-full-ff4cc93-20261006.json` SHA-256은
`805f046c71c8abc71a769b49537633333000db9157d2ecc696226775cc4dc14e`,
log SHA-256은 `009b7c7a4672cd01f6d5cf67f9f9534a008507fe2eb4d4795ec3a3dcf688601b`다.
최종 test-only 수정 전후 두 archive의 source 차이는 `tests/test_matched_preparation.py`
하나뿐임도 확인했다. 실제 모델·학습·비교 구현은 ab45d7f 검증본과 동일하다.

전체 검사에는 이전에 사용한 역사적 개발 calibration/dev fixture 두 개만 복사했다.
각 SHA-256은 `6f4c2ee35e8fd52ae58184cc8f7002ec26b76c7e563e4fbd05506129489dc1b9`,
`1cc485dfe13388040fd5ed966e51c2804d20e7e01bcc602711aa59f9fc1b258f`다.
이번 작업에서 실제 train/새 hard-dev/current private holdout/모델 원본을 열거나
새 GPU job·paid API·model download·의존성 설치·실제 모델 학습을 시작하지 않았다.
CPU 테스트의 작은 synthetic 모델·tokenizer·파일로 구현 동작을 확인한 범위다.

코드로 해결한 조건과 실제 실행으로만 확인할 조건을 구분한다. 아직 필요한 것은 실제
로컬 checkpoint의 완전한 inventory/policy fit 선언 준비, 새 원본 자료와 제공할 public
inventory의 고정 및 과거 사용 목록 복구, GPU native logits/cache parity, 비용 측정과
고정 protocol의 v1 on 대 v2 off 성능 측정이다. 과거 목록을 찾지 못하면 fresh 독립성
주장을 하지 않는다. 조건 검사 통과는 학습 개선이나 목표 달성의 증명이 아니다.
`v2 off > v1 on` 실측 결과, 본학습 완료 또는 JevBench 상위권 달성은 아직 없다.

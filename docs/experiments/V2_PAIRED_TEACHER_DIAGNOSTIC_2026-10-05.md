# V2 saved paired-teacher diagnostic — 2026-10-05

실제 frozen Gemma 4 12B 관측을 재검증했다. 합성·검증 문항에서는 추론이
개선되지만 자연 HelpSteer2 분포에서는 퇴보한다. **일부 문항에 적용한
보정된 serving router의 이득과 전체 raw teacher 분포의 이득은 별도로
판단해야 한다.** 이 문서는 학습 성능이나 JevBench 공식 순위 보고가 아니다.

## 도구와 관측 범위

모델은 `google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, prompt `min`이다. 기존 Claude
관측은 `runs/swift-gpu-20261005/attempt7/extracted/out/google_gemma-4-12B-it/min/`
아래에 있다. 이번 Codex 진단은 model/backend 호출·fitting·학습 없이 저장
파일만 검사했다. 새 독립 private test 원문은 열지 않았다.

`python -m scripts.direct_v2.paired_diagnostic`는 네 입력 파일의 외부 SHA와
두 canonical cohort SHA를 요구한다. Role별 옵션은 `--calibration-direct`,
`--calibration-paired`, `--dev-direct`, `--dev-paired`이고 각각
`--expected-<role>-<kind>-sha256`를 함께 준다. 두
`--expected-<role>-cohort-sha256`와 새로운 `--out` 경로도 필수다.
`--iterations` 기본값 2,000, `--seed` 20261005이며 기존 보고서를 덮어쓰지 않는다.

Direct record, candidate gather, 최종 메시지·recipe·사용량 binding을 검사한다.
Canonical 행의 `(source,id)` 정렬 hash는 adoption cohort와 일치해야 한다.
전체 direct/paired 논리 hash, 실제 파일 SHA, 진단/Swift 의존 소스의 실행
구간 전후 SHA를 기록한다. 순수 함수의 canonical anchor만으로 제외 행·paired
파일의 외부 byte anchor까지 검증됐다고 주장할 수 없다. CLI가 네 파일 전체를 검증한다.

전체 direct inventory에서 ancestry/exact-evidence 연결 성분을 만든 뒤 subset을
취한다. 추론 미관측 bridge도 유지한다. Explicit lineage는 source 간 같은
문자열도 보수적으로 연결하여 우연한 local-ID 충돌을 과하게 묶을 수 있다.
같은 relation으로 calibration/dev 격리도 검사한다. Paraphrase의 의미 독립성은
증명하지 않는다. Bootstrap은 사례별 delta 합계와 행 수를 함께 재표본화하여
question-weighted 평균을 유지한다. 사례 두 개 미만이면 CI를 만들지 않는다.

CI는 탐색용 percentile 95%이며 다중 비교 보정을 하지 않았다. EOS subset은
완료에 조건을 건 관측이고 전체 요청의 인과 효과가 아니다. Raw NLL은 log-space로
계산한다. Score RPS·normalized expected absolute error의 분모는 Score 행만이다.
Hard argmax 진단은 공식 Noul 기권 credit이나 Score 기대값 metric을 대체하지 않는다.

## Coverage

| Source | Canonical 분모 / split | Calibration paired | Dev paired |
|---|---:|---:|---:|
| repository-authored | 832 | 832 | 832 |
| ayaka-v2-verified | 96 | 96 | 96 |
| HelpSteer2 | 320 | 264 | 271 |
| CommonsenseQA | 64 | 6 | 4 |
| MASSIVE ko | 0; grouped 64 제외 | 0 | 0 |
| MASSIVE ja | 0; grouped 64 제외 | 0 | 0 |

전체 direct는 각각 1,440행, canonical은 1,312행이다. Calibration paired
1,198행 중 EOS 1,064 / cap 134, dev paired 1,203행 중 EOS 1,079 / cap 124다.
빈 풀이 0이며 생성 토큰은 각각 246,239 / 246,015다. Canonical 미관측은
114 / 109다. 이전 공유 문서의 dev 961은 전체 paired나 EOS와 맞지 않아
cohort 설명을 Claude에게 요청했다. Coverage는 generation eligibility에 따라
선택된 subset이며 source마다 다르다. ContractNLI 관측은 없다. 자연 정책
문서 전체·ko/ja·26개 초과 후보·미관측 문항으로 일반화하지 않는다.

## 실제 raw 결과

다음은 낮을수록 좋으며 temperature나 serving policy를 추가 적용하지 않았다.

| 관측 | NLL direct → reasoned | Brier direct → reasoned | Score RPS direct → reasoned |
|---|---:|---:|---:|
| Calibration 전체 | 1.79517 → 2.36551 | .53656 → .32805 | .16655 → .13319 |
| Dev 전체 | 1.57088 → 2.28905 | .47798 → .32549 | .16479 → .13682 |
| Dev HelpSteer2, 271행 / 63사례 | 3.18048 → 7.50357 | .86718 → 1.12883 | .18506 → .22843 |
| Dev HelpSteer2 EOS, 196행 / 57사례 | 3.17446 → 6.99258 | .78982 → .95509 | .16306 → .18612 |

전체 dev NLL delta +.71817의 CI는 [.39082, 1.06642], Brier −.15249는
[−.21325, −.09263]다. HelpSteer2 dev는 NLL +4.32310 [3.59227, 5.08388],
Brier +.26166 [.13438, .40174], RPS +.04337 [.01654, .07239]로 악화한다.
Calibration의 같은 source 점 추정도 악화한다. 다만 calibration RPS와
normalized expected absolute error의 CI는 0을 포함하므로 모든 metric에서
유의한 퇴보라고 표현하지 않는다. CQA dev는 4행이며 Brier/NLL CI도 0을 포함한다.

Dev Choice NLL .95433 → .45702는 개선하지만 Noul 1.18327 → 1.29199,
Score 2.28994 → 4.31567는 악화한다. Target 최대 질량 label 기준의 hard argmax
진단은 wrong→correct 210 / correct→wrong 72다. Soft target도 mode로만 비교하므로
공식 credit이 아니다. 더 많은 답을 맞추면서도 틀린 답에 높은 확신을 보여
NLL이 악화할 수 있다. 합성 source의 개선으로 자연 source 퇴보를 가리지 않는다.

Teacher 전체를 증류하거나 cal/dev 풀이를 train으로 바꾸지 않는다. 기존
train 전용 converter와 per-question 양의 gold-NLL 이득, nonworse Brier·Score
ordinal loss, 완결된 풀이 조건을 유지한다. 실패 문항은 gold-only replay에
남는다. 실제 독립적으로 검증된 train 관측과 student의 독립 평가 이득은 아직 필요하다.

## Claude GPU 결과와의 관계

[SWIFT_GPU_RESULTS_2026-10-05.md](SWIFT_GPU_RESULTS_2026-10-05.md)의 별도
calibration-only fitted router는 dev ΔA +.45694, CI [.32891, .64882]로
serving gate를 통과했다. 전체 observed subset을 reasoned로 바꾼 이 문서와
다른 비교다. 라우팅 비율 약 1.2%이며 .7960 → .8087초 p95는 수집 지연
분포 기반 projection이다. 별도 serial HTTP probe 통계와 구분한다.
Router 채택은 train 증류 이득을 증명하지 않는다.

권고는 검증된 `min + selective reasoning`을 유지하고 Cygnet을 독립 hard
dev에서 재검증하는 것이다. 기존 Cygnet ΔA CI는 0을 포함하며 public은 원인
진단에만 쓴다. Public 2×2의 native/Swift factor는 serialization·readout이
함께 바뀐다. Native v1 cell은 LoRA·head·gate·temperature를 포함한 전체
checkpoint이고 Swift cell은 adapter-only다. LoRA/포인터 head만의 효과가 아니다.
Native +5.2 percentage points는 p=.073이며 일반적인 .05 유의 수준을 넘는다.
전체 base 능력 퇴보 가설의 근거는 없지만 모든 영역의 원인을 확정할 수도 없다.
Historical 4,096-token 192와 current 8,192-token 191의 근접성은 이 cohort에서
큰 context 회복 이득을 보여주지 않는다. 과거 구현도 달라 context 단독
ablation이나 모든 작업의 truncation 부정으로 해석하지 않는다.
이 구분과 fitted router의 exact-byte artifact 보존 권고를 `.dev`로 공유했다.

## 실행 anchors와 검증

```text
code commit: 1ab32614436c92936757993ec67129d375d3ceb9
script SHA: c0599f59fd333682c2a1f1930644579eb411309bf5373c4a6568acf2c182a6b5
report: .dev/codex-actual-paired-source-diagnostic-20261005.json
report SHA: 42f8ec9ef67898ba3d93750c50286d2dcf6b955abb54b6285e8b8ddcc4c18b9f
calibration canonical: 4b0b38dbfa17a959e3617623f62e1f13e6bb20542bf5a1cefb480b6f059a7d8b
dev canonical: dde220be7bdc93ef29b20e4783200ab7868874ebd3eaac2820956d4630374551
reasoning adoption SHA: db9b5c5fdb428e38d30d6086f3da97589515e67deeed9195d59cf4968c3eb9be
matched 2x2 SHA: c9377abb868db00248cef128628c67fc67eeb41753672b078ed4518a8df91ac3
```

Report는 네 입력 파일의 SHA와 전체 source SHA도 담는다. 실제 진단은 CPU
12.50초, project `.venv` 관련 tests는 83 passed / 52.37초였다. 새 도구 회귀는
23개이며 Ruff lint/format은 추적 Python 321개에서 통과했다. 독립 read-only
reviewer의 남은 P1/P2는 없다. 초기 system Python suite는 `peft`가 없어
81 passed / 1 failed였고 project `.venv` 재검사로 통과했다. 최종 고정 commit의
전체 CPU suite는 **2,252 passed / 2 skipped / 1 expected duplicate-ZIP warning,
689.61초**로 완료했다. Wrapper 691.47초, exit 0 / stable_pass true, source와
두 development fixture의 전후 hash가 같다. 복원 Linux에서도 새 도구의
23 tests가 9.54초에 통과했고 wrapper 17.38초, source 전후 hash가 같다.

```text
full receipt: .dev/codex-paired-diagnostic-full-1ab3261-20261005.json
full receipt SHA: c4fb1685f8e8ac7c1e3e40f046b5349b252e8027ce0b23905a0c664385a07726
full log SHA: 5b2a5ec95426099d01ffb88325a5f925254b65f6a1feb01d996046499a4066d5
Linux receipt: .dev/direct-linux-paired-diagnostic-1ab3261-20261005.json
Linux receipt SHA: 904900d45f23d5d76c45c334b2ba697ba31b65f2feec541d75da342c73ef13ec
Linux log SHA: b75c459794475095f8047195c6191ed6135c501a7d5cd020dc39fa690e03b63e
```

과거 private inventory metadata 19개에서 서로 다른 평가 파일 SHA 27개를
찾았다. Recovery overlay reference 12개 모두 원래 prepared/reference-ready
manifest와 연결된다. Clean recovery는 해시가 달라 대체하지 않았다. 이 단계는
완전한 이력·semantic decontamination의 증명이 아니다. Metadata-only
기록 SHA는 `8c642b27bc9992d88bdc777642a1d84ebf29a870c86407bb24811402bbd02d3e`다.
후속으로 과거 JSONL은 hash와 nonempty record 수만 stream 검사했다. 처음
26개 원본이 일치했고, 남은 Qwen pilot의 실제 이름 `pilot_dev.jsonl`을 찾아
원래 metadata SHA로 검증했다. 모든 27개 원본의 bytes가 일치한다. 과거
question/gold를 해석·내보내거나 새 train에 넣지 않았다. 새 private 원문 inode와
alias에는 별도 audit hook/descriptor guard를 적용해 open attempts 0과 최종
identity 동일을 확인했다. 이 recovered subset을 완전한 private inventory로
선언하거나 draft corpus의 scope/selection을 변경하지 않았다.

```text
historical bytes receipt: .dev/codex-private-input-bytes-recovery-with-pilot-20261005.json
historical bytes receipt SHA: cc900897f1243a24756d7e4baf232a3e6e0309e5acf25073383bc9e7adaab8ab
verified files: 27; 122082672 bytes; 0.162 seconds
```

이 Codex 단위의 유료 GPU/API 실행은 0이다. Claude의 별도 승인 GPU 측정을
계정 사용량 0으로 표현하지 않는다. Swift/API·main·artifacts/deploy는 변경하지
않았다. 이미 고정·검증한 native upload의 147-source snapshot과 payload를
유지하여 기존 archive anchors는 유효하다. 현재 repository의 새 Swift 커밋이나
이 진단 도구가 이전 upload archive에 포함됐다고 주장하지 않는다.
실제 train teacher·전체 prior-private inventory·CUDA parity/throughput·전체
credit admission·독립 v1 reasoning 대비 v2 off 품질 이득은 여전히 필요하다.

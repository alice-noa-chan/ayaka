# V2 paired temperature diagnostic — 2026-10-05

저장된 Gemma 4 12B 관측의 과신을 calibration-only 온도로 보정했다.
**전체 평균의 보정 이득과 자연 HelpSteer2에서 남는 상대 퇴보가 동시에 관측된다.**
이는 새 학습·serving 승격·v2 off 우위의 증거가 아니다. 기존
[raw paired diagnostic](V2_PAIRED_TEACHER_DIAGNOSTIC_2026-10-05.md)의 후속 분석이다.

## 방법과 범위

`8d1716b`의 `python -m scripts.direct_v2.paired_calibration`은 기존 raw 도구와 같은
네 파일 외부 SHA와 두 canonical cohort SHA를 먼저 요구한다. 두 role의 모든
binding, case/evidence 격리, 동일 모델·토크나이저·runtime·prompt·trace recipe를
검증한 뒤 fit한다. 새 private test와 train 원문은 읽지 않는다.

모델은 `google/gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, prompt `min`이다. 입력은 기존
attempt7의 calibration/dev direct·reasoned 관측이다. Calibration의 완결된
nonempty EOS **동일 1,064쌍**에서 각 경로·유형의 gold NLL 온도를 별도로 맞춘다.
Choice 343 / Noul 346 / Score 375쌍이며 Score에 HelpSteer2 191쌍이 포함된다.
Gold로 이득이 난 문항만 골라 fit하지 않는다. Source별 fit과 dev fit은 없다.

기존 `ayaka.swift.fit.fit_temperature`의 [.25,10] bounded log-temperature search를
재사용한다. Type에 완료 관측이 없으면 T=1과 명시적인 미관측 상태를 기록한다.
Fitter convergence를 새로 입증했다고 주장하지 않는다. Fit membership SHA,
source/type/case 수, 실제 derived log-mass 행 hash와 경계 도달 여부를 기록한다.

| Type | Direct T | Reasoned T | 경계 도달 |
|---|---:|---:|---|
| Choice | 3.95154 | 3.99893 | 없음 |
| Noul | 6.27700 | 6.01283 | 없음 |
| Score | 5.30560 | 7.29266 | 없음 |

온도를 고정한 후 dev에서 원본/원본, 직접 경로만 보정, 추론 경로만 보정,
양쪽 보정, 균등 확률 기준선을 모두 보고한다. Underflow로 0이 된 확률에 floor를
넣지 않고 실제 finite log masses를 보정한다. 원본 관측과 binding은 수정하지 않는다.
양의 온도는 argmax 순서를 유지하지만 Score 기대값과 Noul 결정 경계는 바꿀 수 있다.
HelpSteer2의 선언된 rating target을 human uncertainty 분포로 해석하지 않는다.

이미 관측한 dev에서 세운 가설에 대한 **post-hoc 탐색**이다. CI는 fit을 고정한
전체 direct inventory의 connected-case bootstrap, 2,000회/seed20261005다.
Question-weighted delta를 쓰고 source/type/EOS를 고른 뒤 cluster를 새로 만들지 않는다.
Fit uncertainty와 다중 비교를 보정하지 않았으며 독립 확인 실험이 아니다.

## 실제 dev 결과

아래는 낮을수록 좋고 두 경로 모두 위 calibration 온도를 적용했다.
전체 dev 관측은 1,203쌍, EOS 완료 subset은 1,079쌍이다. Score RPS의 전체 분모는
각각 495 / 391 Score쌍이며 다른 metric과 다르다. HelpSteer2는 모두 Score다.

| Subset | NLL direct → reasoned | Brier direct → reasoned | Score RPS direct → reasoned |
|---|---:|---:|---:|
| 모든 관측, 1,203쌍 | .70818 → .55218 | .34085 → .22652 | .10416 → .09577 |
| EOS, 1,079쌍 | .62884 → .42491 | .31334 → .17076 | .09607 → .07488 |
| HelpSteer2 모든 관측, 271쌍/63사례 | 1.26303 → 1.49522 | .64578 → .75574 | .13245 → .15704 |
| HelpSteer2 EOS, 196쌍/57사례 | 1.18898 → 1.35932 | .60730 → .68103 | .12211 → .13618 |

전체 관측 NLL delta −.15600 CI [−.21369,−.09886], Brier −.11432
[−.14870,−.08142]다. 전체 Score RPS delta −.00839의 CI [−.02051,.00337]은
0을 포함한다. EOS 전체에서는 NLL/Brier/RPS 세 delta CI 모두 0보다 작다.
이는 완료에 조건을 건 비교이며 전체 요청의 추론 효과가 아니다.

HelpSteer2 전체의 상대 NLL delta +.23219 CI [.13038,.34508], Brier +.10996
[.05309,.17053], RPS +.02459 [.01143,.03880]는 양수다. EOS에서도 NLL +.17034
[.04001,.30257], Brier +.07374 [.00106,.14720]는 양수이나 RPS +.01407
[−.00027,.02854]는 0을 포함한다. 모든 subset의 모든 지표가 유의하다고 쓰지 않는다.

균등 분포의 HelpSteer2 NLL은 ln(5)=1.60944다. 보정한 두 경로 모두 이 기준보다
낮지만 추론 경로는 보정한 직접 경로보다 높다. Raw argmax target-mode 일치는
HelpSteer2 전체에서 direct49.08% / reasoned43.17%, EOS에서56.12% /52.04%이며
보정 뒤에도 같다. 이 hard argmax 진단은 공식 Noul 기권·Score 기대값 지표가 아니다.

추론 쪽만 보정하면 HelpSteer2 NLL은 raw direct3.18048보다 낮은1.49522가 되어
좋아 보인다. 그러나 직접 경로도 같은 기준으로 보정하면1.26303이므로 상대 퇴보가
남는다. **과신 감소만으로 자연 source의 teacher 이득을 선언하면 안 된다.**
같은 이유로 pooled 개선을 모든 train 문항에 대한 증류 근거로 쓰지 않는다.
기존 train 전용 per-question gold-NLL 이득·Brier/ordinal 비퇴보·완결 조건을 유지한다.
실제 train 관측과 student 독립 평가가 필요하다. 이 분석으로 converter에 온도를
자동 주입하거나 serving gate/현재 `min + adopted route`를 바꾸지 않았다.

## 실행과 재현

CLI 옵션은 raw 도구와 동일한 `--calibration-direct/paired`, `--dev-direct/paired`,
네 `--expected-<role>-<kind>-sha256`, 두 `--expected-<role>-cohort-sha256`, 새로운
`--out`, `--iterations`, `--seed`다. 이미 존재하는 report는 덮어쓰지 않는다.
Fitter/loss를 포함한 Swift 소스와 두 진단 도구를 실행 전후 SHA로 묶는다.

```text
code: 8d1716bf88abf03de93776843c5e8353f33e4be2
script SHA: f449f49632b8c83cf972c01a181a69f55c512f359bb91182f65d058ce5a2e5c1
report: .dev/codex-actual-paired-calibration-diagnostic-stable-20261005.json
report SHA: e2a0e2b2bfd329fad08b0110b5015166fdf97d37a4b73ef77c2500c632034544
execution receipt SHA: e002f42b6a8a4f2477936fc5c8a8a4a1f3f2dc215f2545ac6f621ff94b3b91a4
fit membership SHA: 3374fac9265bf91c0b80fb75937ff52343f8466cef20dd143ef8f1e9172589af
```

실제 anchored 분석은 CPU23.65초, project `.venv` 관련59 tests는38.75초에 통과했다.
새 회귀13개가 cal-only/동일 membership/soft gold/underflow/입력 불변/빈 완료/line
permutation/외부 anchors/출력 보존/source drift를 확인한다. Ruff lint/format 통과,
독립 read-only reviewer의 남은 P1/P2는 없다. 원본 raw bootstrap을 재사용하고
cluster 입력을 정렬해 같은 seed의 결과를 입력 파일 행 순서와 무관하게 만든다.
초기 반복 계산 버전의 report는 보존했으며 최종 stable report가 위 anchors다.
초기 helper의 요약 출력은 source key 대소문자로 실패했으나 실제 CLI/report는 성공했다.
표시를 고쳐 새 report/receipt에서 CLI와 wrapper 모두 exit0을 확인했다.

고정8d1716b의 전체 CPU suite는 **2,266 passed / 2 skipped / 1 warning**, pytest
764.59초에 통과했다(wrapper766.62초). 경고는 중복 archive member를 거부하는
기존 테스트의 의도한 zipfile 경고다. 실행 전후 source SHA가 같았으며 receipt는
`.dev/codex-paired-calibration-full-8d1716b-20261005.json`이고 SHA는
`422ce1a57889c966af480f24664e89c72fe4a743a9f9c6ad36804079ba705011`이다. 고정 snapshot에는
검증에 필요한 기존 개발용 calibration/dev fixture 두 개만 복사했으며 새 holdout은
복사하거나 읽지 않았다. 클로드의 이후 untracked hard-dev 초안은 포함하지 않는다.

복원 Linux runtime에서도 두 diagnostic test 파일 **36 passed / 16.87초**를
확인했다. `.dev/direct-linux-paired-calibration-8d1716b-20261005.json`은 반환된
tool output을 기록한 receipt이며 wall-clock 측정을 별도로 보증하지 않는다.
기록 시 tested source가 고정 archive와 같음을 검증했고 receipt SHA는
`6b23cc24d2ce872fc9470c865537e87358e82f9603b317846bc9dfc9f219e9b8`이다.

이 단위의 유료 GPU/API·실제 학습은0이며 기존 frozen native upload·holdout·main·
Swift 소유 코드를 변경하지 않았다. 실제 train teacher 관측, CUDA backward/throughput,
과거 private 평가 inventory의 완전한 격리와 student 독립 성능 승격은 남아 있다.

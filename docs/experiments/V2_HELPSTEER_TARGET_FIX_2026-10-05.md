# HelpSteer ordinal target correction — 2026-10-05

공통 `helpsteer2_scores`가 소수점 rating을 `int()`로 낮추는 문제를 수정했다.
예를 들어 원본2.5는 이전에는2점 one-hot이었다. 이제2점과3점에 각 .5를 주어
원본 평균2.5를 보존한다. 이는 선언된 평균 점수의 adjacent-level 표현이며
실제 사람들의 표 수나 불확실성 분포를 추정한 것이 아니다.

## 적용 범위와 호환성

`ayaka.data.transforms.helpsteer2_target`을 공통 helper로 사용한다.
이미 소수점 평균을 보존하던 verified-direct converter의 `_ordinal_target`도
이 helper의 alias여서 기존 import와 정답 검증 의미는 유지된다. Generic loader와
verified-direct가 서로 다른 원평균을 학습·평가 정답으로 만드는 경로를 없앴다.

- 정수0..4의 one-hot, 후보·질문 순서, state와 metadata 값은 그대로다.
- 누락되거나 `None`인 attribute는 이전처럼 건너뛴다.
- 존재하는 rating은 finite numeric `int/float`의0..4여야 한다. 범위를 벗어난 값,
  bool·문자열·NaN·inf는 `ValueError`다. 이전의 int coercion/silent skip에 의존한
  caller에는 의도적인 계약 변경이다. 큰 정수도 같은 오류로 거부한다.
- JSON 왕복과 canonical Swift read의 soft-gold 변환 뒤에도 원평균을 유지한다.

기존 frozen corpus나 저장 관측을 덮어쓰거나 자동 재채점하지 않았다. 새 관측에는
수정한 converter로 corpus를 다시 만들고 원본 revision·file bytes·변환 소스와
unique example/question IDs를 수집 전에 고정해야 한다. 이 수정이 이미 발행된
390/396 hard-dev corpus에 반영됐다고 주장하지 않는다.

## 검증

코드·관련 회귀 작성 → Ruff fix → format → lint/format 재검사 → 관련 테스트 →
diff 검토 순으로 진행했다. 여섯 test 파일109개가7.71초에 통과했다. 새29개
회귀는 정수 호환, 원평균, JSON·wire·scorer 연결, shared verifier, 잘못된 rating,
누락 attribute를 검사한다. 실제 source 데이터는 다운로드하거나 열지 않았다.

코드는 `2fa760e437aedda71069f178ff6879e5e40a2122`로 별도 커밋했고 read-only 독립
검토의 남은 P1/P2는 없다. 고정 snapshot의324개 Python 파일 Ruff lint/format도 통과했다.
복원 Linux runtime의 같은 여섯 test 파일도109개/16.03초에 통과했다. CPU/offline
환경을 명시했고 검증 source의 실행 전후 SHA가 같았다. Linux receipt는
`.dev/direct-linux-helpsteer-targets-2fa760e-20261005.json`, SHA는
`28332508f0dec316cb14590fd93d69f7337fa515de01d3a0c8f8c2d6c7c9cd1a`다.

고정 `2fa760e`의 전체 CPU 통합은 **2,298 passed / 2 skipped / 1 warning**, pytest
1,006.39초에 통과했다(wrapper1,009.19초). 경고는 기존 duplicate archive member
거부 회귀에서 의도한 zipfile 경고다. 실행 전후 소스는 불변이며 모든 tested source가
git archive의 commit bytes와 같음을 확인했다. 알려진 개발용 calibration/dev fixture
두 개만 복사했고 현재 private holdout 원문은 복사하거나 읽지 않았다.

```text
full receipt: .dev/codex-helpsteer-target-full-2fa760e-20261005.json
receipt SHA: 7946b669676811964905ad6e4a9e49c3ab7838c5e3ef7bc71ad63bac369762a1
log SHA: f4a0e178a4a0f6e96496cb82bf05887c44854e54ece1f0e0df75c2f2db83ab74
archive SHA: 735eeaa43c57b429a7eda6227a5a68f9e0f2fb402d450de704ef78e1b0fb76ec
```

Swift 파일은 클로드 소유로 유지하며 이 수정은 공통 데이터 변환기 두 파일과 별도
회귀만 변경한다. Snapshot은 `eb88c34`의 hard-dev source도 포함하지만 전체 테스트
통과가 그 protocol의 ID/binding/격리·수집 실패 처리 검토 항목을 해결한 것은 아니다.

## 결과 해석

이는 재현 가능한 데이터 계약 버그의 수정이다. 실제 validation에서 fractional row가
얼마나 있는지 측정하지 않았고 이전 v2 실패의 원인이라고 입증하지 않았다.
[공정한 paired calibration 진단](V2_PAIRED_CALIBRATION_DIAGNOSTIC_2026-10-05.md)에서
남은 HelpSteer2 추론의 상대 성능 저하가 이 수정으로 해결됐다는 증거도 없다.
유효한 train teacher 관측과 student의 독립 평가는 여전히 필요하다.

이 단계의 유료 GPU/API 사용과 실제 학습은0이다. Main·현재 private holdout·기존
native upload를 변경하지 않았다. Hard-dev builder의 ID/원본 binding/문서 격리·
최소 cohort 검증은 별도 소스 검토 항목이며 `.dev`로 클로드에게 전달했다.

# Component coverage와 teacher 꼬리값 보고

Claude의 `.dev/claude-to-codex-20261004.md` 27–28절을 반영한 CPU 준비 변경이다.
의견·검토·담당 범위는 `.dev/codex-replies-to-claude-20261005.md` 37절 이후에 공유한다.
GPU 실행은 보류 중이다. 기존 선택 결과와 공개 v1/main은 바꾸지 않는다.

## 문항 수와 문단 연결 집합 수

[앞선 paragraph helper](V2_PARAGRAPH_PROTOCOL_PREPARATION_2026-10-05.md)는 공급한
전체 원본 context의 제목·본문 연결을 먼저 닫고 역할 겹침과 decision minima를
검사한다. 문항 8개가 같은 문단을 공유하면 문항 quota 8을 채워도 연결 집합은 하나다.
새 `minimum_components` 옵션은 다음 protocol에서 사전 선언한 집합 최소 수도 검사한다.

구현은 [paragraph_groups.py](../../ayaka/data/paragraph_groups.py), commit `28d1430`이다.

```python
receipt = audit_paragraph_roles(
    index,
    selected_ids_by_role,
    minimum_counts=PREDECLARED_DECISION_MINIMA,
    minimum_components=PREDECLARED_COMPONENT_MINIMA,
)
mapping = receipt["component_audit"]["membership"]
mapping_sha256 = receipt["component_audit"]["membership_sha256"]
```

Component minima는 decision minima와 **정확히 같은 role/source/type 셀**에 양의 정수로
선언해야 한다. 빠진 셀, 추가 셀, 0·음수·bool·소수는 거부한다. 각 셀에서 서로 다른
full-inventory component ID 수를 센다. 같은 component가 여러 source/type 셀에
걸려 있으면 각 셀에 나타나지만 role 전체에서는 한 번만 센다. 셀 수를 더해 독립 표본
수라고 부르면 안 된다.

Receipt는 정렬한 선택 ID→component mapping, 그 digest, 셀별·role별 집합 수를 낸다.
선택되지 않은 bridge의 연결도 유지하며, 기존 ID/역할/bridge/decision minima 검사를
우회하지 않는다. 생략하거나 `None`이면 이전 receipt shape와 동작이 그대로다.
기존 inventory logical digest, component ID, split key와 normalization은 바뀌지 않는다.

집합은 **공급한 문단 관계의 의존 단위**다. 실제 corpus completeness, 질문·human label,
사례/템플릿 등 다른 의존 관계, 통계적 독립성이나 유효 표본 크기의 증명이 아니다.
`independence_attested`와 `effective_sample_size_estimated`는 false다. 기존
`promotable`·historical/public decontamination attest도 false다. 원본 file byte anchor,
source mapping과 추가 case 관계 검증은 별도로 유지해야 한다.

관련 paragraph/source grouping **66 passed / 13.34s**, Ruff fix→format→재검사,
diff review와 독립 source review P1/P2 없음으로 확인했다. 실제 평가 데이터의 적절한
minima를 선택하거나 기존 노출 cohort를 사후 필터링한 것은 아니다. 다음 builder에
연결하는 작업은 Claude 담당으로 요청했으며 아직 연결 완료를 주장하지 않는다.

## 극소 확률이 평균을 지배하는지 확인

Claude 28절은 유한한 극소 확률이 평균 log-odds 이동·KL·OLS를 지배할 수 있다고
지적했다. Claude의 과거 표는 `1e-9` clipping을 사용했으므로 기존 raw 보고서 평균과
같은 수치라고 비교하지 않는다. 확률을 바꾸지 않고 다음 요약을 함께 기록한다.

[teacher_signals.py](../../ayaka/training/teacher_signals.py)의 report version은
`ayaka-prompt-teacher-signals-2`다. 원래 raw 평균·OLS·상관·endpoint/nonfinite 수를
유지하고, 전체·통과·거부 source/type slice 각각에 아래 통계를 추가한다.

- 유한한 teacher→student KL과 entropy 변화의 min, p10, median, p90, max, finite n.
- Noul student/teacher log-odds 및 teacher−student 이동의 같은 요약.
- `probability_clipping_applied: false`와 재현 가능한 quantile 방법.

Quantile은 정렬한 값의 `(n−1)q` 위치에서 선형 보간한다. 단일 값이면 모든 분위수가
같고, 유한한 관측이 없으면 n=0, 나머지는 `null`이다. KL 무한값과 Noul zero-support
endpoint는 원래 별도 count를 유지하고 finite-only 요약에서 제외한다. Entropy가
유한한 endpoint pair는 entropy 변화 요약에는 포함된다.

기계적 회귀에서는 10개의 이동 0과 하나의 `1e-20` 학생 확률을 섞었다. Raw 이동의
최댓값 약 46.05와 평균 약 4.19는 유지되고 중앙값·p90은 0이다. 이는 소수 꼬리값을
드러내는 toy 결과이며 실제 train/calibration 개선이나 distillation 채택 근거가 아니다.
Records/teacher target/gold filter/loss를 바꾸지 않았고 새로운 eligibility threshold,
authored cap 또는 serving policy를 적용하지 않는다. Policy refit과 fitted 대 fitted 비교
요구는 유지한다. 새 prompt 준비 보고서는 version 2로 다시 생성해야 한다.

관련 5개 모듈 **112 passed / 58.09s**, Ruff fix→format→재검사와 diff review를 통과했다.
독립 최종 source review P1/P2 없음으로 확인한 뒤 `fa962b6`으로 별도 commit했다.
최종 통합 수치는 아래 고정 source 검증 범위에 기록했다.

## Claude의 비교 source 교차 검토

Claude의 `8d783f7` 수정은 expected cohort membership, all-row recipe, resume binding,
exposed scope와 case/title closure를 보강했다. 동일 commit의 source-only 독립 검토에서
남은 P2를 `.dev` 39절로 전달했다.

- Nested input binding과 실제 top-level scoring/provenance 필드를 연결해야 한다.
- Frozen v1 policy를 truthiness가 아닌 전체 exact contract로 검증해야 한다.
- Checkpoint receipt의 선언 SHA를 실제 소비 파일 bytes와 model load 전에 대조해야 한다.
- 저장된 Swift native record/runtime/pass/logits/masses/probabilities의 기존 strict
  validator를 comparator에서 적용해야 한다.

실제 원본 cohort/read/checkpoint bytes를 열어 오류를 발견한 것은 아니다. 수정은
Claude 담당이며 이 문서가 향후 commit까지 같은 결함이 남는다고 보장하지 않는다.
고정 노출 cohort의 exact title closure와 다음 원본 structured inventory의 정규화·본문·
제외 bridge closure는 범위가 다르다. 새 builder에서 두 결과의 차이를 기록해야 한다.

문맥 8192와 절차형 standard tier는 사전 protocol로 유지하고, 유리한 방향을 가정하거나
관측 후 재선택하지 않는다고 답했다. 문맥 입력 overflow/clipping은 별도 preflight와
실행 기록으로 확인해야 한다. 새로운 v1 on / v2 off 성능 측정은 아직 없다.

## 최종 검증 범위

각 코드 변경은 관련 검사와 검토 후 별도 commit했다. 마지막 code source는
`fa962b666a741b18410d8c4d95ca45c694297bd4`, archive SHA-256은
`f86af7c5cd981ac3932728649765a46f331b7767170a6f786d62f7a65c5e8258`이다.
Linux CPU/offline 관련 9개 모듈은 **272 passed / 134.43s** (wrapper 150.30s)로 완료했다.
Source `.py/.sh` 342개 before/after hash가 archive bytes와 일치하고 log anchor도 확인했다.
Ignored receipt `.dev/codex-pretraining-linux-fa962b6-20261006.json`의 SHA-256은
`e9d2126aa8da4140d6db4abbaf60e7aa88e3f0469344c18b2bbd6ec9f83adc1d`, log SHA-256은
`f63888d7321e2bd64269695b336c915c5adf73dfe3a69d3f4f3cfd843144ce2c`다.
Windows CPU/offline 전체는 **2,454 passed / 2 skipped / 1 warning / 1,142.45s**
(wrapper 1,146.17s)로 완료했다. Warning은 ZIP 중복 이름을 만드는 거부 회귀의 출력이다.
Source `.py/.sh` 342개, `pyproject.toml`, 역사적 개발 fixture 두 개의 총 345개
before/after hash가 archive와 알려진 fixture anchor에 일치한다.
Ignored receipt `.dev/codex-pretraining-full-fa962b6-20261006.json`의 SHA-256은
`6062541d8020c2efcff2d88627797ba7bdfea200244ab6d6ffc50e83b6265466`, log SHA-256은
`9d11a6b160d7e2c0d8ed36f544d9e6beee62181cef4f4f8d73179e802a82409d`다.
개발 calibration/dev fixture의 SHA-256은 각각
`6f4c2ee35e8fd52ae58184cc8f7002ec26b76c7e563e4fbd05506129489dc1b9`,
`1cc485dfe13388040fd5ed966e51c2804d20e7e01bcc602711aa59f9fc1b258f`다.
이번 변경에서 현재 private holdout과 실제 train/model 원본을 열지 않았다. 전체 검사에는 알려진
역사적 calibration/dev fixture 두 개만 사용하며, 원본 corpus 독립성·GPU parity·
학습 성능·사용자 목표의 우위를 대신 입증하지 않는다. 새 paid GPU/API/학습/다운로드/
의존성 설치는 실행하지 않았다.

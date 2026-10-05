# Ayaka v2: literature review before another paid training run

검토일: 2026-10-05. 대상 브랜치: `ayaka-v2-experiments`.
목표는 **v2 비추론 판정이 v1 추론 판정보다 좋아지는 것**이다. 아래 우선순위는
논문·공식 구현과 현재 소스를 대조한 설계 판단이며, Ayaka에서 측정한 학습 이득이 아니다.
이 검토에서는 GPU, 유료 모델 API, 학습, 데이터셋 다운로드를 실행하지 않았다.

## 현재 관측이 요구하는 연구 방향

- 입력 직렬화와 프롬프트가 중요했다. 다만 이것만으로 현재 v2 off가 v1 on을 넘었다고
  할 수 없다. 역사적인 public 결과는 같은 환경의 paired 비교도 아니다.
  [기존 GPU 결과](SWIFT_GPU_RESULTS_2026-10-05.md).
- 같은 유형별 calibration-only 온도를 양쪽 경로에 적용해도 HelpSteer2의 reasoned
  판정은 direct보다 NLL/Brier가 나빴다. 이것은 이 cohort·보정 방식의 상대 퇴보이며
  모든 원인이나 source별 최적 보정 결과를 밝힌 것은 아니다.
  [공정 calibration 진단](V2_PAIRED_CALIBRATION_DIAGNOSTIC_2026-10-05.md).
- Claude의 최신 보고는 새 hard protocol에서도 cygnet을 채택하지 않았다고 한다.
  긴 프롬프트의 Intelligence 이득과 입력 비용을 분리할 동기가 있다. 해당 보고의
  source/job 검토 지적이 모두 해소됐다는 뜻으로 인용하지 않는다.
  [Claude가 기록한 결과](SWIFT_HARD_DEV_PROTOCOL_2026-10-05.md#result-2026-10-05-run-once).

따라서 추론을 더 길게 생성하는 일보다, **원문만 보는 판정 경로를 직접 감독하고
유용한 학습 신호를 그 경로로 옮기는 방법**을 먼저 비교할 필요가 있다.

## 논문에서 확인한 방법과 적용 범위

| 연구 / 출처 | 실제 방법 | Ayaka 판단과 한계 |
|---|---|---|
| [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/), Findings ACL 2023; [공식 구현](https://github.com/google-research/distilling-step-by-step) | 원문→label과 원문→rationale를 task prefix로 나눈 공동 학습. 판정 시 rationale 입력이 필요 없다. | 비추론 목표와 직접 맞는다. 논문은 T5 및 네 NLP 과제이며 Gemma decision head의 개선을 증명하지 않는다. 공개 PaLM rationale 자료를 가져오지 않고 검증된 자체 풀이 또는 허용된 사람 주석을 사용한다. |
| [From Explicit CoT to Implicit CoT](https://arxiv.org/html/2405.14838v1), 2024; [공식 구현](https://github.com/da03/Internalize_CoT_Step_by_Step) | CoT 앞부분을 점진적으로 제거해 최종적으로 원문→답을 학습한다. 제거 변화에 optimizer reset과 removal smoothing을 적용한다. | 직접 판정으로 수렴하는 curriculum 후보. 여러 단계의 비용·안정성이 필요해 첫 실험보다 뒤에 둔다. 단순 무작위 trace 삭제는 이 논문의 재현이 아니다. |
| [On-Policy Context Distillation](https://arxiv.org/html/2602.12275v2), 2026 preprint | context 없는 student가 생성한 궤적에서 context 있는 teacher와 token-level reverse KL을 학습한다. 최적화된 system prompt를 가중치로 옮기는 실험도 있다. | 긴 고정 지시문→짧은 입력이라는 동기가 유용하다. 우리 finite-candidate forward KL 실험은 OPCD 자체가 아니다. 원문의 rollout 결과·성능·시간을 Ayaka에 외삽하지 않는다. |
| [HelpSteer2-Preference](https://arxiv.org/html/2410.01257v2), ICLR 2025 | 기존 사람 rating에 별도의 사람 preference와 justification을 추가하고 regression/Bradley–Terry 및 결합 방식을 비교한다. | 자연 Score 학습을 일반 수학 CoT와 구분할 근거. 사람의 원래 pair 주석과 criterion을 확인한 경우만 보조 preference 학습을 고려한다. 평균 rating 차이를 원래 human preference라 부르지 않는다. |
| [Prometheus 2](https://aclanthology.org/2024.emnlp-main.248/), EMNLP 2024 | 사용자 rubric에 따른 direct assessment와 pairwise ranking을 따로 학습한 모델의 weight merging을 비교한다. | rubric별 판정 감독을 참고한다. ordinal 확률의 NLL/RPS 개선을 보여준 논문은 아니다. GPT-4로 만든 feedback 자료를 현재 학습 자료에 편입하지 않으며, 두 모델 학습·merge를 첫 실험에 추가하지 않는다. |
| [Number Cookbook](https://arxiv.org/abs/2411.03766v3), ICLR 2025; [공식 구현](https://github.com/GraphPKU/number_cookbook) | 네 수 표현과 17 작업의 41개 조합, 길이 범위별 평가. 사전학습부터 유용한 수 표현·tokenizer 기법이 기존 모델 fine-tuning에는 항상 유효하지 않았다. | 기존 tokenizer를 유지하고 작업×표현×자리수×경계의 coverage를 검사한다. 날짜 계산 전이는 별도 가설이다. GPL-3.0 저장소 소스를 Ayaka 코드에 복사하지 않고 원리만 참고한다. |
| [Coconut](https://arxiv.org/html/2412.06769v2), 2024; [공식 구현](https://github.com/facebookresearch/coconut) | 마지막 hidden state를 다음 input embedding으로 돌려 연속 latent thought를 계산한다. | 텍스트 풀이가 없어도 순차 계산이 추가된다. 현재 원문 한 번으로 판정하는 off와 다른 경로이므로 첫 비용 제한 실험에서 제외한다. |
| [Rethinking On-Policy Self-Distillation](https://arxiv.org/html/2607.05184v1), 2026 preprint | privileged context의 distillation이 일부 thinking 모델의 긴 수학 추론을 악화시킨 경우와 token-level correction 신호를 분석한다. | teacher에게 정보를 더 주면 반드시 좋아진다는 전제를 반박한다. Ayaka의 finite-label 퇴보 원인을 증명한 것은 아니다. 원래 gold와 직접 경로를 유지하고 teacher의 실제 개선·악화를 함께 검사한다. |

Cloudflare의 [Clef 공식 설명](https://blog.cloudflare.com/clef-decision-models/)도 다시 확인했다.
LoRA+head 학습, CE+Brier 및 입력 순열 증강은 참고할 수 있지만 CE+Brier와 여러
순열·reference 보존 도구는 이미 Ayaka에 있다. 이들을 새 아이디어라고 다시 셈하지 않는다.
기존 [Clef 분석](V2_CLEF_EVIDENCE_2026-10-04.md)에 구현과 범위가 기록돼 있다.
Decision Index 성적을 JevBench 성적으로 대체하지 않는다.

## 현재 코드와 논문의 차이

`ayaka/training/reasoning.py::reasoning_items`는 기본적으로 원문 direct item을 포함하고,
검증 풀이가 있으면 풀이→판정 item과 rationale CE positions를 추가한다. 따라서
**직접 판정 감독이 전혀 없다**고 설명하면 틀리다. 하지만 label-after-trace 손실도
함께 있으므로 DSS의 별도 label/rationale 두 과제와 같은 실험이라고 부를 수 없다.
어느 손실이 원문 판정에 도움이 됐는지는 분리된 절제가 필요하다.

`ayaka/training/direct_distillation.py::prepare_direct_distillation`는 원문만 입력하는
direct item에 gold NLL과 승인된 teacher forward KL을 유지한다. 준비 보고의
`reasoning_training_tokens`는 0이다. 이 경로에는 rationale CE가 아직 연결돼 있지 않다.
현재 teacher 형식은 완결된 nonempty reasoned trace를 요구하므로 prompt-only teacher를
빈 풀이·가짜 풀이로 넣어 계약을 우회해서는 안 된다.

Score의 Brier/RPS와 frozen native reference replay도 기존 구현에 있다.
이 손실과 reference를 유지한 상태에서 추가 학습 신호의 기여를 비교한다.

## 비용을 고려한 제안

먼저 같은 native readout/adapter와 같은 자연 train 자료를 쓰는 **gold-only direct
제어군**을 확보한다. 모든 실험에서 원문 label loss, 자연 replay, 후보·rubric 의미를
유지한다. train/dev/calibration/test의 role을 바꾸거나 이미 본 hard dev를 새 독립
확인용으로 사용하지 않는다.

### 첫 비교 후보: 고정 지시문의 증류

고정된 긴 지시문을 쓰는 teacher의 후보 분포를, 짧은 `min` 입력을 쓰는 student로 옮긴다.
매번 긴 프롬프트를 보내는 비용을 피하면서 그 이득을 유지할 수 있는지 검증한다.
이것은 OPCD에서 착안한 **finite-label context distillation** 제안이다. rollout이나
reverse KL을 추가하지 않고 gold에 고정한 기존 forward KL 계약을 비교 출발점으로 삼는다.

- 학습 전에 teacher의 긴 입력과 student의 짧은 입력을 각각 바인딩한다. 같은 state,
  질문, criterion, gold, 후보 순서와 lineage를 공통 semantic record에 묶는다.
  Model/revision/tokenizer/recipe, 실제 native token IDs, 후보별 mass/probability와
  saved payload의 외부 SHA도 묶는다.
- 내부화 대상은 고정 일반 지시문이다. 요청마다 바뀌는 문서·사용자 criterion·증거를
  student에서 숨기면 정당한 원문 판정 문제가 아니므로 허용하지 않는다.
- 별도의 `prompt_context` teacher kind와 train-only 수집·검증이 필요하다. 현재
  reasoned teacher artifact를 자동 재사용할 수 있는 준비 완료 상태가 아니다.
- teacher의 gold 대비 유효성 및 gold-only 제어군 대비 student 변화가 판단 근거다.
  같은 train 원문의 min 분포 대비 NLL/Brier 및 Score RPS·nMAE acceptance 기준을
  수집 전에 선언한다.
  train에서 teacher가 나은 행만 쓰면 그 선택 yield와 제외된 행도 보고한다.
- cygnet의 serving 채택 실패를 뒤집는 절차가 아니다. 그 관측은 동기만 제공하며
  학습 자료와 확인 cohort는 새로 격리해야 한다.

### 다음 비교 후보: 원문 판정 + 짧은 검증 풀이의 보조 과제

DSS를 참고해 **원문→분포**와 **원문→검증 풀이**를 구분해 같은 adapter를 학습한다.
첫 절제는 trace-conditioned label loss를 빼고 direct gold loss+rationale CE를 비교한다.
기존 trace-conditioned 판정이 필요한 `on` 경로를 없애는 API 변경은 하지 않는다.

현재 joint loss와 비교할 때 label/trace item 수로 평균낸 숨은 가중치 차이를 없앤다.
과제별 명시 mask와 reduction으로 CE-only 행이 direct loss에 들어가거나 그 분모를
희석하지 않게 한다. CE-only 행은 decision loss에서 제외하고, 제어군과 초기 모델·
optimizer schedule·direct question 방문 수·direct loss 분모를 맞춘다.
풀이 길이와 가중치를 고정하고 실제 처리 토큰과
GPU시간을 기록한다. 풀이를 추가한 만큼 gold-only 처리량이 줄면 그것도 비용이다.
gold-only·rationale 보조·기존 joint를 모두 본 학습 크기로 동시에 돌릴 예산은 가정하지 않는다.

날짜·수치의 풀이 검증은 학습 준비에만 사용한다. 이미 월말·윤년·영업일·시간대·반올림
grammar와 독립 verifier가 있으므로, 없는 기능이라고 다시 구현하는 대신 수 표현과
자리수·규칙 조합·문서 표현의 격리 및 coverage를 먼저 CPU에서 검사한다.
Carry/borrow·rounding 경계를 포함하고 같은 수의 표현 alias를 같은 split component에
묶는다. 길이 holdout과 후보 위치 균형을 새 관측 전에 고정한다.

### 자연 Score와 curriculum의 순서

자연 Score는 원래 사람 rating/rubric을 쓰는 direct replay와 기존 proper-score 손실을
우선 유지한다. 모든 reasoned 분포를 좋은 teacher로 간주하지 않는다. human pair 보조는
원본 pairing·criterion·동점/무효 계약을 확보한 뒤 별도 작은 절제로 비교한다.
같은 prompt의 두 response·기존 rating·추가 preference/justification도 한 ancestry
component로 묶어 학습과 평가 사이에 흩어지지 않게 한다. Prometheus의 합성 주석은
HelpSteer2의 원래 human pair 주석과 구분한다.
점진적 CoT 제거는 보조 rationale가 원문 판정에 도움이 된다는 관측과 충분한 단계별
예산이 있을 때 다음 후보로 검토한다. 지금 두 방법을 합쳐 재현 실패 원인을 늘리지 않는다.

## 승격에 필요한 결과

1. 같은 입력 cohort에서 frozen 공개 v1 + frozen worked-steps 정책과 v2 `off`를 비교한다.
   min-vs-cygnet gate와 전체 평균만으로 이 목표를 달성했다고 하지 않는다.
2. Choice/Noul/Score별 이득과 자연 Score 퇴보, 날짜·수치의 OOD 결과를 함께 기록한다.
   NLL/Brier/RPS, Score 기대값·nMAE, abstention, 한국어·일본어 회귀를 유지한다.
3. 제어군과 같은 데이터 조건의 비교 및 실제 총 학습 토큰/GPU시간을 제시한다.
   teacher 수집, 추가 rationale CE, 평가, 다운로드·회수를 모두 비용에 포함한다.
4. 방법 선택은 dev에만, 최종 확인은 새로 고정한 independent test에만 한다.
   public 결과나 이미 본 hard 관측으로 다시 고른 방법은 독립 검증이라고 부르지 않는다.

논문으로 **시도할 이유**는 찾았지만, v1 추론 초과·JevBench 상위권·기존 크레딧 내
완주를 증명하지는 못했다. 다음 결정은 gold-only 기준에 대해 위 두 학습 신호 중
무엇이 실제 direct 이득을 주는지
분리해서 확인하는 것이다. 구체 GPU 견적은 대상 장치의 측정 처리량과 전체 데이터
토큰 수가 필요하다. 이 문서는 새 GPU 실행 승인이나 학습 완료 보고가 아니다.

## 독립 검토

기존 코드 리뷰 에이전트가 논문 원문·저자 구현과 현재 두 학습 경로를 별도로 대조했다.
Gold-only 다음 prompt-only 증류를 가장 작은 보조 후보로 두고, rationale CE와 한 arm에
동시에 넣지 않는 순서에 동의했다. Direct loss 분모/문항 방문 수와 human/synthetic
preference provenance 지적을 위 설계에 반영했다. Claude에는 `.dev` outbox로
동일 자료와 판단 질문을 전달했으며, 이 설계에 대한 Claude의 답변은 아직 수신하지 않았다.

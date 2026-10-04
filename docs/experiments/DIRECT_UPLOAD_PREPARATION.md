# Direct-student CPU 준비와 업로드

이 도구는 검증된 direct 학습 입력을 옮기는 용도다. Cloud instance 생성,
GPU 학습, 결제 또는 패키지 설치를 수행하지 않는다. 성공 결과도
`production_ready=false`이며 Linux/CUDA 실행 환경은 별도로 준비해야 한다.

## 실제 native 가중치 점검

`scripts/direct_v2/native_layout.py`는 외부 SHA256으로 고정한 snapshot record를
받고 전체 loader 파일을 확인한다. 설치된 HF loader와 같은 이름 변환을 적용해
meta text model의 모든 parameter와 persistent buffer를 실제 safetensors header와
비교한다. 임의로 입력 embedding을 출력 head로 대신하지 않는다.

실제로 공유하는 parameter의 identity로 생략 가능한 tied alias를 확인한다.
이름 충돌, 누락, 크기 불일치, 알 수 없는 text tensor, tensor 내용 변환이 필요한
loader와 중복 저장된 tied alias는 명시적으로 실패한다. 정상 종료 전에 파일을
다시 확인해 header를 유지한 데이터 변경도 감지한다. Report는 validator SHA와
라이브러리 버전을 기록하며 pretrained tensor를 읽거나 GPU를 초기화하지 않는다.

2026-10-05에 준비한 `google/gemma-4-12B-it`의 immutable revision은
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`이다. 실제 다운로드한 단일 weight 파일은
23,919,549,408 bytes이며 공식 LFS SHA256과 일치했다:

```text
5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d
```

전체 677개 tensor 중 BF16 text 666개가 text state 667개를 충족한다.
생략된 `lm_head.weight`는 실제로 공유하는 `model.embed_tokens.weight`의 alias다.
나머지 11개는 사용하지 않는 audio/vision 항목으로 보고한다. 공식 full weight와
실제 LoRA·판독 모듈을 포함한 보수적 parameter 수는 12,104,772,662다.
이는 저장 파일과 meta 구조의 검사 결과이며 실제 loading·CUDA·품질 결과가 아니다.

```powershell
.venv/Scripts/python.exe -m scripts.direct_v2.native_layout `
  --snapshot-record .dev/native-gemma4-12b-snapshot-20261005.json `
  --expected-snapshot-record-sha256 9b0a408a6f4f198e4002e455ec233bd6ab4882e18784b3affccdad1c6f48273e `
  --snapshot-path .dev/native-gemma4-12b-20261005 `
  --out .dev/native-layout-new.json
```

Report는 새 파일에만 쓴다. 기존 record를 바꾸고 그 해시를 새로 계산해서
기존 감사의 외부 anchor를 대체하지 않는다.

## 업로드 묶음

`scripts/direct_v2/package.py build`는 CPU audit receipt의 외부 SHA256과
고정 source inventory를 표준 라이브러리로 먼저 확인한 후 해당 코드를 import한다.
기존 `.pyc`가 검증한 `.py`를 대신 실행하지 않도록 빈 bytecode cache를 사용한다.
Production 준비는 새 Python process에서 실행한다. Preloaded source 예외와
`--allow-tiny`는 외부 anchor로 확인한 실제 tiny recipe에만 허용된다.

묶음에는 다음 파일만 들어간다.

- 감사한 Ayaka Python source, LICENSE, pyproject와 고정 모델 survey.
- 준비된 train/router/dev/calibration, encoded rows와 별도 test의 opaque commitment.
- CPU audit receipt, snapshot record와 record가 고정한 native loader 파일.
- 실제 실행한 layout validator와 upload 도구, hash inventory와 CPU audit 명령.

원본 test 입력, raw 데이터와 native 디렉터리의 HF download cache는 포함하지 않는다.
파일마다 복사한 buffer의 길이와 SHA256을 검사한다. 실패 시 새로 만들던 partial
archive만 지운다. 압축은 zstd level 3, checksum 사용, worker 2개로 고정한다.

```powershell
.venv/Scripts/python.exe -m scripts.direct_v2.package build `
  --source-root .dev/frozen-policy-corpus-61f3c6b-20261005/source `
  --bundle .dev/direct-policy-native-61f3c6b-20261005/bundle `
  --audit-receipt .dev/direct-policy-native-61f3c6b-20261005/cpu-audit.json `
  --expected-bundle-sha256 170ece7c5e1b41dadd98051b26e1260e889c8033cc91c6658a5a087a7438a624 `
  --expected-audit-receipt-sha256 2c960994d18a20624fb8ca4854624daa24b9f9e40ed21855348d68447e3e1b7c `
  --snapshot-record .dev/native-gemma4-12b-snapshot-20261005.json `
  --expected-snapshot-record-sha256 9b0a408a6f4f198e4002e455ec233bd6ab4882e18784b3affccdad1c6f48273e `
  --snapshot-path .dev/native-gemma4-12b-20261005 `
  --out .dev/direct-upload-new.tar.zst
```

검증에는 build가 반환한 archive SHA256을 별도로 저장해 전달한다.
Verifier는 실제 읽은 compressed stream을 hash하고 모든 member의 inventory와
bytes를 대조한다. Path traversal, symlink, duplicate, manifest 이후 member,
tar EOF 이후 숨겨진 내용과 zstd checksum 불일치는 실패한다.
Anchor는 포함한 receipt/manifest/record의 digest와 일치해야 하며 실행 명령은
허용한 CPU `audit`의 인자 목록과 완전히 일치해야 한다.

```powershell
.venv/Scripts/python.exe -m scripts.direct_v2.package verify `
  --archive .dev/direct-upload-new.tar.zst `
  --expected-sha256 <build에서별도로저장한SHA256>
```

검증한 archive는 빈 디렉터리에 풀고 그 root에서 manifest의 CPU audit 명령을
실행한다. CPU receipt는 transformers/tokenizers/peft의 정확한 버전 일치가 필요하다.
이번 준비 환경은 5.18.0 / 0.23.2 / 0.21.2다. Torch의 CPU/CUDA build 차이와
Flash Attention/Liger의 실제 동작은 별도로 검사하며 이 archive가 증명하지 않는다.

2026-10-05 실제 최종 묶음은 169개 파일, 압축 전 23,977,145,314 bytes에서
18,736,900,543 bytes로 줄었다(21.855% 절감). 147개 Ayaka Python 파일은
감사한 `61f3c6b` source와 동일하다. Archive SHA256은 다음과 같다.

```text
6781202a7c0521f6cdef57b35944e8df6ea42f94283d30211da49118e1b9c9f2
```

모든 member와 압축 stream의 검증이 통과했다. 별도 새 디렉터리에 전체 파일을
다시 복원한 뒤 HF offline·빈 cache의 production `run_direct audit`도 통과했다.
Bundle/CPU receipt/snapshot anchor를 유지했고, actual full native weight bytes를
검사했으며 `model_weights_loaded=false`, `optimizer_steps=0`으로 종료했다.
실제 복원 감사 receipt의 SHA256은
`7c4f58da635e01e2f97eb42011640f9b9b4f810e6a228c25edf31f7d70dfcafa`다.

관련 CPU regression 37개가 72.35초에 통과했다. 소스·bytecode import 경계,
실제 tiny 예외, streaming 변경, 출력 head/shape/alias, 숨은 tar/zstd 내용,
command/anchor binding과 relocation을 포함한다. 독립 검토의 발견 사항을
수정한 뒤 추가 P1/P2가 없다는 결과를 받았다. 이는 Claude 승인과 별개다.
312개 source file의 Ruff lint와 format 재검사가 통과했다.

최종 `94f6005` 고정 source의 전체 CPU 검사는 **2,147 passed, 2 skipped,
1 warning / 649.63초**로 통과했다. Source와 개발 fixture의 전후 SHA256이
동일하며 exit 0, `stable_pass=true`다. 첫 실행은 Git archive에서 ignored
개발 fixture가 빠져 7개가 실패했다. 이전 검증본의 calibration/dev JSONL
두 개를 동일 해시로 복사한 뒤 재검사했다. Source 수정으로 우회하지 않았고
실패 receipt도 보존했다. Warning은 중복 ZIP member 공격 fixture의 예상 경고다.

```text
full receipt: .dev/codex-native-upload-full-94f6005-fixtures-20261005.json
full log SHA: 5c5ee1291800b05cb8e6c887bdedf13d61cb29472aa2f494b2d0d1bdd9db671c
```

## 남은 학습 조건

실제 reasoned teacher의 paired 이득, prior private evaluation inventory,
native CUDA kernel parity와 전체 schedule의 throughput, setup/test/export/download/
cleanup/recovery를 포함한 보유 credit 내 완주 예측이 필요하다. 실측으로 선택한
후보만 독립 test에서 v1 reasoning과 비교한다. CPU 준비나 header 일치에서
품질 향상, v2 off의 우위, JevBench 순위 또는 본학습 완료를 주장하지 않는다.

# Direct-student CPU 준비와 업로드

이 도구는 검증된 direct 학습 입력을 옮기는 용도다. Cloud instance 생성,
GPU 학습, 결제 또는 패키지 설치를 수행하지 않는다. 성공 결과도
`production_ready=false`다. Linux 실행 환경은 별도 고정 runtime으로 준비하며,
CPU import 성공은 목표 GPU의 CUDA 실행 성공과 구분한다.

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

현재 실제 24GB 파일의 준비는 아래 Linux 명령을 사용한다. 경로는 준비한
runtime prefix로 바꾼다. Windows의 별도 페이징 한도 실패는 뒤에 기록했다.

```bash
/absolute/prepared/ayaka-runtime/bin/python3.11 -m scripts.direct_v2.package build \
  --source-root .dev/frozen-direct-runtime-20261005/source \
  --bundle .dev/direct-policy-native-9fe2434-20261005/bundle \
  --audit-receipt .dev/direct-policy-native-9fe2434-20261005/cpu-audit.json \
  --expected-bundle-sha256 0ceeb878f60ed6877c7d15ba5eda9455f24e8289ebe39f7ec7e6948bea8f17dc \
  --expected-audit-receipt-sha256 eb693eb47c62a52a8d8ce34d4f04841c042dd48848ed6bc22456b1a2f3900836 \
  --snapshot-record .dev/native-gemma4-12b-snapshot-20261005.json \
  --expected-snapshot-record-sha256 9b0a408a6f4f198e4002e455ec233bd6ab4882e18784b3affccdad1c6f48273e \
  --snapshot-path .dev/native-gemma4-12b-20261005 \
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

2026-10-05의 이전 `61f3c6b` 묶음은 169개 파일, 압축 전 23,977,145,314 bytes에서
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

## Linux 실행 환경과 실제로 발견한 준비 실패

`5c55f8d`는 `scripts/direct_v2/requirements.lock.txt`, `prepare_runtime.sh`,
`runtime.py`와 22개 CPU 회귀 검사를 추가했다. Python 3.11.14 / Linux x86_64의
60개 distribution을 wheel SHA256으로 고정하고 binary-only 설치한다.
실제 CPU receipt와 일치하는 transformers 5.18.0, tokenizers 0.23.2,
peft 0.21.2를 유지한다. Torch 2.8.0+cu128, Flash Attention 2.8.3.post1,
Liger 0.8.4를 쓴다. 기존 배포 환경의 다른 dependency pin이나
flash-linear-attention은 이 FA2 실행 환경을 대신하지 않는다.

[공식 PyTorch CUDA wheel](https://download.pytorch.org/whl/cu128/torch/),
[공식 Flash Attention release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1),
[Liger 소스](https://github.com/linkedin/Liger-Kernel)를 기준으로 고정했다.
FA2 wheel은 cp311 / torch2.8 / cu12 / CXX11 ABI TRUE 빌드다.
설치된 distribution version은 filename의 build suffix와 달리 `2.8.3.post1`이다.

```text
lock SHA: a7537375091f574f6a703f6a540e1859e5b600eb2ac80a5521aebd43c1de8ba9
FA2 wheel SHA: 3807faf34d263ad0c3dff6f07c7ebdd27aef4a2614ef124db2afee4578ef7232
```

CPU에서 새 Linux prefix를 만드는 명령은 다음과 같다. `uv`를 PATH에 넣고
대소문자를 구분하는 Linux 파일 시스템의 새 절대 경로를 사용한다.

```bash
bash scripts/direct_v2/prepare_runtime.sh \
  /absolute/new/ayaka-runtime \
  a7537375091f574f6a703f6a540e1859e5b600eb2ac80a5521aebd43c1de8ba9
```

실제 local Ubuntu 26.04에서 60개 dependency metadata, prefix 안의 module origin,
FA2 확장 `.so` import, normal/varlen 함수와 padding helper, HF FA2 wrapper와
Liger RMSNorm/GEGLU의 인자를 검사했다. CUDA를 초기화하거나 실행하지 않았다.
원래 경로를 없앤 relocation 후 Python interpreter로 실행한 import 검사도 통과했다.
CLI script의 기존 shebang 전체를 재작성했다는 뜻은 아니므로 prefix의
`bin/python3.11`로 필요한 모듈을 실행한다.

실제 준비 중 세 가지 실패를 확인하고 기존 기록을 보존했다.

- Windows mount의 case collision으로 CPython terminfo 복사가 실패했다.
  복사 전에 목적지의 대소문자 구분을 검사하도록 바꿨다.
- 정상 kernel import 후 production audit가 `PIL` import에서 실패했다.
  Pillow 11.3.0을 lock에 추가하고 깨끗한 prefix를 다시 만들었다.
- 이전 `safe_open(framework="pt")`는 header 검사에서도 전체 24GB shard의
  writable TorchStorage를 만들어 7.5GiB RAM / 2GiB swap의 Linux host에서
  실패했다. `0bb3500`은 header 검사만 NumPy backend로 바꿨다. Tensor value를
  읽지 않으며 native pretrained loader는 바꾸지 않았다. 실제 같은 host에서
  677개 header와 666/667 text layout 검사가 통과했다. 동작 근거는
  [safetensors 0.8.0 구현](https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/bindings/python/src/lib.rs)이다.

새 header 검사도 Windows의 별도 페이징 한도 오류를 해결했다고 주장하지 않는다.
동시 전체 테스트 중 실제 Windows packaging은 OS error 1455로 실패했고,
새 upload package는 검증된 Linux 환경에서 준비한다. OS 설정을 바꾸지 않았다.

```text
Linux native layout: .dev/direct-linux-native-layout-numpy-20261005.json
layout SHA: 5f6770f540336bd57a1eb58629e31792b9bc7182d9ba1f45708aa38c2ba5ae2f
```

Header 변경 후 기존 source-bound receipt를 그대로 사용하지 않았다.
`9fe2434` source에서 전체 CPU gold/render/schedule audit를 다시 수행했다.
기존 prepared payload와 opaque test commitment는 byte 단위로 동일하다.
2,368개 준비 row를 74 × 32로 한 번씩 방문하는 전체 schedule을 유지한다.
새 source에서 selection을 다시 실행했다거나 teacher 이득을 확보했다는 뜻은 아니다.

```text
bundle SHA: 0ceeb878f60ed6877c7d15ba5eda9455f24e8289ebe39f7ec7e6948bea8f17dc
CPU receipt SHA: eb693eb47c62a52a8d8ce34d4f04841c042dd48848ed6bc22456b1a2f3900836
test commitment SHA: f0d695cc6ca226f3d2378c02ad65a880a295fd2b445ae16a383d617c001ea4bc
```

기존 private holdout manifest는 이전 development manifest SHA를 참조한다.
Opaque commitment가 같아도 그 연결이 자동으로 바뀌지 않는다. 원본 test를
열지 않는 metadata migration과 별도 외부 anchor가 필요하다. 아래 `f2c510f`
단계에서 실제 연결을 갱신했으며 기존 기록은 보존했다.

`9fe2434`는 Linux Torch 2.8의 CPU attention dispatch 차이도 검사에 반영했다.
FP32 GQA/반복 KV 경로 사이 rounding을 semantic 오류와 구분하기 위해 독립
FP64 matmul·mask·softmax reference를 사용하고 별도 FP32 12개 case를 추가했다.
기존 production kernel이나 허용 오차를 바꾸지 않았다. Windows kernel 관련
35개와 Linux native/head/objective/resume 관련 96개가 통과했다.

고정 Linux prefix는 zstd level 3 / worker 2 / checksum으로 별도 압축했다.
7,975,697,724 bytes의 파일 payload가 3,558,478,728 bytes의 archive가 됐다.
Archive stream SHA와 24,827개 파일의 byte hash, 1,048개 내부 symbolic link를
확인하고 새 Linux 경로에 실제 복원했다. 복원된 interpreter의 CPU import도
통과했다. Runtime archive는 pretrained/model data archive와 별개다.
복원된 interpreter에서 새 source-bound bundle과 실제 24GB 가중치를 사용한
production `run_direct audit`도 HF offline·빈 cache로 통과했다. Native bytes를
확인했고 model weight loading과 optimizer step은 0이다. 248.72초는 DrvFS의
CPU 파일 감사 시간이며 CUDA 학습 속도로 사용하지 않는다.

```text
runtime archive: .dev/direct-linux-runtime-9fe2434-20261005.tar.zst
archive SHA: c1667a4fe1e18c4159b3b914b4fa60fe6686493a4746c91e301377b0458b72c3
inventory SHA: b58e127129ffc981a0d283b65f3564baa3e694f72c8b6247ac03488f9287c6c2
restored runtime/audit receipt: .dev/direct-linux-runtime-restored-9fe2434-20261005.json
receipt SHA: 8ee1cdd932677b687209119925e64711d4ff5dee6c6d81f02c6af2e0b4e4da89
```

이 결과는 local Ubuntu 26.04 / glibc 2.43에서 확인했다. 더 오래된 target host의
glibc, 실제 GPU와 FA2 binary의 지원 범위, CUDA forward/backward는 별도 조건이다.
Torch가 보고하는 `sm120` compile target을 FA2의 Blackwell 실행 증거로 사용하지 않는다.

수정된 `9fe2434` Ayaka source와 새 CPU receipt의 model/data archive는 Linux에서
정상 생성됐다. zstd level 3 / 169개 파일 / 18,736,893,392 bytes다. 전체 compressed
stream과 각 member의 bytes, inventory, anchors, CPU audit command binding을
검증해 exit 0으로 통과했다. 이 새 archive를 full native files와 함께 다시 복원한
검사는 수행하지 않았다. 위의 새 source/bundle/native CPU audit와 archive 전체
byte 검증을 각각 증거로 기록하며 이전 full restore 결과와 구분한다.

```text
model/data archive: .dev/direct-policy-native-upload-9fe2434-20261005.tar.zst
archive SHA: 107527d4159269032b3bf9442beeaa80abc9db993ab78aca2b82e0716d1e4f8f
verify record: .dev/direct-policy-native-upload-9fe2434-20261005.verify.json
```

마지막 고정 `9fe2434` 전체 CPU 검사는 **2,182 passed, 2 skipped, 1 warning /
935.74초**로 통과했다. Source와 개발 fixture의 전후 hash 동일,
exit 0 / `stable_pass=true`다. Wrapper는 child의 OMP/MKL/OpenBLAS thread를
각각 1로 고정하고 UTF-8 출력을 사용했다. GPU throughput 개선 근거로 사용하지 않는다.
314개 Python file의 Ruff lint/format 검사와 문서 anchor/whitespace 검사가 통과했다.

첫 전체 검사는 2,181 passed / 1 failed였다. Corpus replay의 checkpoint directory
rename이 Windows `WinError 5`로 실패했다. 관련 27개와 최종 전체 재검사가 source
수정 없이 통과했다. 원인을 특정하거나 publication 오류를 코드로 해결했다고
주장하지 않는다. 첫 wrapper의 cp949 출력 오류는 receipt 저장 후 발생했고,
새 wrapper의 UTF-8 출력으로 해결했다. 실패 log/receipt를 보존했다.
Warning은 중복 ZIP member 공격 fixture의 예상 경고다.

```text
full receipt: .dev/codex-direct-runtime-full-9fe2434-single-thread-20261005.json
full receipt SHA: 167468c4614bcab4c7bd94fda02eb332ff7918bdc83a262772c1b8f6d9749059
full log SHA: ea590c6b156089e201da1a5a25e62264a2184118ca89e6b2a6ee947d981467ce
failed receipt: .dev/codex-direct-runtime-full-9fe2434-20261005.json
```

## 남은 학습 조건

실제 reasoned teacher의 paired 이득, prior private evaluation inventory,
native CUDA kernel parity와 전체 schedule의 throughput, setup/test/export/download/
cleanup/recovery를 포함한 보유 credit 내 완주 예측, private holdout의 새
model/policy dev 선택 고정과 독립 평가가 필요하다. Development manifest의
metadata 연결은 아래에서 완료했다. 실측으로 선택한
후보만 독립 test에서 v1 reasoning과 비교한다. CPU 준비나 header 일치에서
품질 향상, v2 off의 우위, JevBench 순위 또는 본학습 완료를 주장하지 않는다.

## 2026-10-05 원문 내용을 읽지 않는 holdout 연결 갱신

`f2c510f`의 [rebind_holdout.py](../../scripts/direct_v2/rebind_holdout.py)는
서로 다른 source revision에서 full CPU audit로 재검증한 동일 frozen payload
bundle을 연결한다. Payload selection과 preparation을 다시 수행했다는 뜻은 아니다.
기존·새 bundle manifest와 CPU audit, 기존 holdout manifest의 SHA 다섯 개를
외부에서 고정해야 한다. 아홉 development payload의 실제 bytes와 opaque
commitment, source/manifest SHA를 제외한 두 complete CPU receipt가 동일해야 한다.
Source inventory의 digest 변경·추가·삭제를 양쪽 값으로 기록한다.

원문 `test.jsonl`의 내용은 읽지 않고 파일 식별자만 확인한다. 같은 filesystem의
새 별도 holdout directory에 hard link를 만들며 기존 manifest를 덮어쓰지 않는다.
Hard link는 동일 파일을 가리키므로 bytes의 불변 snapshot이나 접근 통제가 아니다.
두 경로의 device/inode/size/mtime를 검증하고, metadata와 payload를 다시 확인한다.
읽기 전에 path와 실제 열린 descriptor를 검사해 원문의 hard-link alias와 reparse
file을 거부한다. 원문 내용을 SHA로 검증했다고 주장하지 않는다.
Path 검사 직후 파일이 교체된 실패 경로에서는 descriptor가 먼저 열릴 수 있으나,
내용을 읽기 전 identity 검사에서 중단한다. 외부 동시 변경을 차단하는 기능은 아니다.

출력은 exclusive create·short-write 검사·flush·fsync를 거친다. Manifest를 마지막에
쓰고, 저장된 두 metadata 파일을 안전한 reader로 다시 hash 검사한 뒤 성공 anchor를
반환한다. 오류 시 부분 directory를 진단용으로 보존하며 자동 삭제·덮어쓰기·copy
fallback은 없다. Atomic directory transaction이나 동시 변경에 대한 접근 통제로
확대하지 않는다. 실패한 partial directory에 있는 manifest를 성공 receipt 없이
사용하지 말고 새 output 경로로 진단 후 실행한다.

CLI는 표준 라이브러리만 사용한다. 원문 test나 학습 모델을 준비 과정에서 열지 않는다.
다음 경로와 anchor는 이번 local 61f3c6b → 9fe2434 갱신의 재현 예시다. 이미 생성된
output directory에서는 재실행을 거부하므로 재현에는 별도 새 `--out`이 필요하다.

```sh
python -m scripts.direct_v2.rebind_holdout \
  --old-bundle .dev/direct-policy-native-61f3c6b-20261005/bundle \
  --new-bundle .dev/direct-policy-native-9fe2434-20261005/bundle \
  --old-audit .dev/direct-policy-native-61f3c6b-20261005/cpu-audit.json \
  --new-audit .dev/direct-policy-native-9fe2434-20261005/cpu-audit.json \
  --holdout .dev/direct-policy-native-61f3c6b-20261005/bundle-holdout \
  --out .dev/direct-policy-native-9fe2434-20261005/bundle-holdout \
  --expected-old-bundle-sha256 170ece7c5e1b41dadd98051b26e1260e889c8033cc91c6658a5a087a7438a624 \
  --expected-new-bundle-sha256 0ceeb878f60ed6877c7d15ba5eda9455f24e8289ebe39f7ec7e6948bea8f17dc \
  --expected-old-audit-sha256 2c960994d18a20624fb8ca4854624daa24b9f9e40ed21855348d68447e3e1b7c \
  --expected-new-audit-sha256 eb693eb47c62a52a8d8ce34d4f04841c042dd48848ed6bc22456b1a2f3900836 \
  --expected-holdout-manifest-sha256 f86e5420047cf2916a65a3fe79989ad765c40d51bad8aed3af1b6fd034a36b5a
```

이번 실제 갱신은 Windows NTFS에서 0.418초에 통과했다. 별도 Python audit hook으로
원문과 그 inode alias의 `open`을 금지했다. 원문 열기 시도 0, 같은 파일 식별자와
두 저장 SHA의 일치를 확인했다. Torch를 import하거나 모델·GPU·optimizer를
실행하지 않았다. Source 차이는 `training/native_snapshot.py` 하나다.

```text
new holdout: .dev/direct-policy-native-9fe2434-20261005/bundle-holdout
holdout manifest SHA: 39237aaeccfea35cc9976976503c2c5ec8d1f3201ecc5f6d0e2d25b1d3fc5135
migration SHA: e5b0c38ca99c10e5aef23b3fb22fc2cde70355722b60e6c83f8598a56da2239b
execution receipt: .dev/direct-native-holdout-rebinding-20261005.json
execution receipt SHA: c6f7b3bba4f39f39be4c1eacf4fee6cc5633403b7ef7b564ce649507ede7e017
```

기존 holdout-v2 manifest 계약과 `open_holdout`은 유지했다. 예전 model/policy
selection은 복사하지 않았다. 새 development manifest에 맞춘 독립 dev 선택을
외부 SHA로 고정한 뒤 evaluator가 원문 bytes와 membership을 검증해야 한다.
이번 migration 자체는 새 모델 선택·품질 결과·학습 실행 증거가 아니다.

Related Windows CPU tests는 **60 passed / 36.84초**다. 잘못된 anchor/payload/receipt,
원문 alias와 descriptor race, 출력 손상·sync 실패, source 추가·삭제, 기존 selection
거부와 나중 원문 검증을 검사했다. 독립 read-only reviewer는 남은 P1/P2를 찾지
못했다. 복원된 Linux runtime의 같은 테스트도 **60 passed / 121.22초**로 통과했다.
Linux wrapper의 전체 실행 시간은 128.32초이며 migration tool/test source의 전후
hash가 같다.

```text
Linux receipt: .dev/direct-linux-holdout-rebinding-f2c510f-20261005.json
Linux receipt SHA: 4d510338b84bf6238018152a0b3f8f4391d3ae2286f17d9b7fadd92231c15a16
Linux log SHA: 3148a517b14632e98974edcb226e19ff82e5e5b69d729c1bfa48474e4e8be9af
```

고정 `f2c510f` 전체 CPU integration은 **2,221 passed / 2 skipped / 1 warning,
743.84초**로 통과했다. Wrapper duration은 746.04초다. Exit 0 / stable_pass true,
source·개발 fixture의 전후 hash 동일이다. Child CPU thread는 각각 1로 고정했다.
Warning은 중복 ZIP member fixture의 예상 경고다. 이 elapsed time을 CUDA 학습
속도 개선으로 해석하지 않는다. Ruff lint/format은 추적 중인 Python 파일 319개에서
통과했다.

```text
full receipt: .dev/codex-holdout-rebinding-full-f2c510f-20261005.json
full receipt SHA: 123d8e20368dcc567656d43fb466268e46da8c872ae708c37936743e642841d4
full log SHA: 99cad99e7db959d9ea182a29b6708f2698996b81487246145420b178d68f4a84
```

Ayaka의 audited 147개 core source와 준비 payload는 이 변경에서 바뀌지 않았다.
따라서 위 기존 model/data·runtime archive와 CPU audit anchor는 유지된다. 새
migration helper는 별도 utility로 제공하며 이전 169-file native archive에 포함됐다고
주장하지 않는다. Private holdout은 training upload와 분리해서 보관한다.

## 2026-10-05 실제 teacher 관측의 추가 진단

저장된 12B calibration/dev 추론을 source별로 검사했다. Raw 분포는 일부
합성 문항에서 개선되지만 자연 HelpSteer2에서 퇴보한다. 별도의 보정된 sparse
serving router가 dev gate를 통과한 결과를 train teacher 이득으로 대체하지
않는다. 기존 train-only/per-question nonregression 조건을 유지하고, 실제
독립 train 관측과 student 품질 검사는 계속 필요하다.

새 standalone 진단과 관련 tests83, frozen full tests2252, restored-Linux
tests23은 통과했다. 과거 평가 원본27개의 metadata/bytes도 복원·확인했지만
전체 이력이나 semantic decontamination까지 증명한 것은 아니다. 기존
고정 upload의 source/data는 변경하지 않았고 새 진단은 archive에 포함되지
않는다. 재현 anchors와 범위는
[V2_PAIRED_TEACHER_DIAGNOSTIC_2026-10-05.md](V2_PAIRED_TEACHER_DIAGNOSTIC_2026-10-05.md)에 있다.

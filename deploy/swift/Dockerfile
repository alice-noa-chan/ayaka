# Ayaka Swift: frozen google/gemma-4-12B-it, one-token canonical letter readout, fitted decision policy.
#   docker build -f deploy/swift/Dockerfile -t ayaka-swift .
#   docker run --gpus all -p 8009:8009 -e HOST=0.0.0.0 -v $HF_HOME:/root/.cache/huggingface ayaka-swift
FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90

WORKDIR /opt/ayaka
COPY pyproject.toml README.md LICENSE ./
COPY ayaka ./ayaka
COPY deploy/swift ./deploy/swift
# The Swift server and readers use the standard library plus the tokenizer from the
# image's transformers; Ayaka's training dependencies are not needed.
RUN pip install --no-cache-dir --no-deps -e . && chmod +x deploy/swift/serve.sh

ENV POLICY=/opt/ayaka/deploy/swift/policy.json HF_HUB_OFFLINE=0
EXPOSE 8009
ENTRYPOINT ["bash", "deploy/swift/serve.sh"]

# Reproducible serving image for Ayaka: the exact base image (by digest) and
# package versions used for every reported number (see constraints.txt).
#   docker build -t ayaka .
#   docker run --gpus all -p 8000:8000 ayaka \
#     --ckpt alice-noa-chan/ayaka-large --revision <hf-revision> --reasoning
# runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# (torch 2.8.0.dev20250319+cu128, CUDA 12.8; needs an NVIDIA driver >= 570)
FROM runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853
WORKDIR /opt/ayaka
COPY pyproject.toml constraints.txt README.md LICENSE ./
COPY ayaka ./ayaka
RUN python -m pip install --no-cache-dir -c constraints.txt -e .
ENV HF_HOME=/models
EXPOSE 8000
ENTRYPOINT ["python", "-m", "ayaka.serve", "--device", "cuda", "--host", "0.0.0.0", "--port", "8000"]

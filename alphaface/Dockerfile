# AlphaFace realtime face swap — self-contained image.
#
# CUDA is NOT taken from a CUDA base image: the pinned torch/onnxruntime-gpu
# wheels ship their own cu124 libraries, and mixing them with a system CUDA is
# what caused CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on the original box.
# A plain Ubuntu + the nvidia container runtime (driver only) is what works.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      git curl ca-certificates \
      libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks its python as externally-managed; a venv is the clean way in.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
# insightface 0.7.3 publishes no cp312 wheel, so it compiles a Cython extension
# from source and needs a toolchain. Installed and removed in one layer so the
# ~250 MB of build tools never lands in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential python3-dev \
 && pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt \
 && apt-get purge -y build-essential python3-dev \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# The AlphaFace model code (~65 MB). Weights are NOT baked in — they are ~2 GB
# and live in a volume, fetched by the entrypoint on first run.
ARG ALPHAFACE_REPO=https://github.com/andrewyu90/Alphaface_Official.git
RUN git clone --depth 1 "$ALPHAFACE_REPO" /opt/Alphaface_Official

ENV ALPHA_DIR=/opt/Alphaface_Official \
    FF_MODELS=/models/onnx \
    VM_MODELS=/models/onnx \
    HOST=0.0.0.0 \
    PORT=8001 \
    INSIGHTFACE_HOME=/models/insightface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY rt_alphaface_server.py /opt/rt_alphaface_server.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/opt/entrypoint.sh"]

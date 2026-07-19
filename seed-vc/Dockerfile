# seed-vc realtime voice conversion — self-contained image.
#
# Same reasoning as the AlphaFace image: no CUDA base image, the pinned torch
# cu124 wheels bring their own CUDA libs and the nvidia container runtime
# supplies the driver.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      git curl ca-certificates \
      ffmpeg libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# seed-vc source. Pinned to the commit the server was built against — upstream
# changes real-time-gui.py's internals, which vc_server.py imports by file path.
ARG SEEDVC_REPO=https://github.com/Plachtaa/seed-vc.git
ARG SEEDVC_COMMIT=51383ef
RUN git clone "$SEEDVC_REPO" /app \
 && cd /app && git checkout "$SEEDVC_COMMIT"

WORKDIR /app

# vc_server.py loads real-time-gui.py by path from its own directory, so it has
# to sit inside the checkout rather than somewhere on PYTHONPATH.
COPY vc_server.py /app/vc_server.py
COPY vc_index.html /app/vc_index.html
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Model weights (~4.4 GB) are fetched from HuggingFace on first run into these
# cache dirs — both are volumes in docker-compose.yml.
ENV HF_HOME=/models/hf \
    HF_HUB_CACHE=/models/hf/hub \
    SEEDVC_CHECKPOINTS=/models/checkpoints \
    HOST=0.0.0.0 \
    PORT=17494 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 17494
ENTRYPOINT ["/opt/entrypoint.sh"]

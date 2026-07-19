#!/bin/bash
# Fetches everything AlphaFace needs into /models (a volume, so it happens once)
# and then starts the server. Safe to re-run: every step is skipped if the file
# is already there.
set -euo pipefail

MODELS=/models
ONNX="$MODELS/onnx"
mkdir -p "$ONNX" "$MODELS/insightface"

# insightface 0.7.3 ignores INSIGHTFACE_HOME and hardcodes ~/.insightface, so
# point that at the volume — otherwise buffalo_l (~300 MB) is re-downloaded
# every time the container is recreated.
if [ ! -L /root/.insightface ]; then
  rm -rf /root/.insightface
  ln -sfn "$MODELS/insightface" /root/.insightface
fi

# --- AlphaFace checkpoints (Google Drive) --------------------------------------
# ~2 GB total. gdown handles Drive's large-file confirm page, which plain curl
# does not. If Drive rate-limits you, drop the files into the volume by hand:
#   docker compose cp <file> alphaface:/models/...
ALPHA_CKPT="$MODELS/alphaface_demo.pt"
ARCFACE_CKPT="$MODELS/arcface_w600k_r50_pytorch.pt"

# gdown 6.x takes the id positionally — the old `--id` flag was removed.
if [ ! -s "$ALPHA_CKPT" ]; then
  echo "[init] downloading alphaface_demo.pt (1.8 GB, first run only)..."
  gdown "${ALPHAFACE_DEMO_GDRIVE_ID:-18ZOQB3WmIFnMwi1GqBroFFEOuSNKWpZQ}" -O "$ALPHA_CKPT"
fi
if [ ! -s "$ARCFACE_CKPT" ]; then
  echo "[init] downloading arcface_w600k_r50_pytorch.pt (167 MB)..."
  gdown "${ARCFACE_GDRIVE_ID:-1qc4s6eRQPluma72WFibUnw74GPMAYRtY}" -O "$ARCFACE_CKPT"
fi

# The model code loads these by relative path from ALPHA_DIR, so link them in
# rather than copying 2 GB around.
ln -sfn "$ALPHA_CKPT" "$ALPHA_DIR/alphaface_demo.pt"
mkdir -p "$ALPHA_DIR/Models"
ln -sfn "$ARCFACE_CKPT" "$ALPHA_DIR/Models/arcface_w600k_r50_pytorch.pt"

# --- ONNX masks & restorers ----------------------------------------------------
# bisenet/xseg come from facefusion-assets, GPEN/occluder from visomaster-assets.
# gfpgan is only a fallback for GPEN but the server loads it unconditionally.
FF_BASE=https://github.com/facefusion/facefusion-assets/releases/download
VM_BASE=https://github.com/visomaster/visomaster-assets/releases/download

fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then return 0; fi
  echo "[init] downloading $(basename "$dest")..."
  # -sS: no progress bar (it makes `docker compose logs` unreadable) but still
  # print real errors. GitHub release downloads do time out now and then.
  curl -fL -sS --retry 5 --retry-delay 3 --retry-all-errors -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

fetch "$FF_BASE/models-3.0.0/bisenet_resnet_34.onnx" "$ONNX/bisenet_resnet_34.onnx"
fetch "$FF_BASE/models-3.1.0/xseg_1.onnx"            "$ONNX/xseg_1.onnx"
fetch "$FF_BASE/models-3.0.0/gfpgan_1.4.onnx"        "$ONNX/gfpgan_1.4.onnx"
fetch "$VM_BASE/v0.1.0/GPEN-BFR-512.onnx"            "$ONNX/GPEN-BFR-512.onnx"
fetch "$VM_BASE/v0.1.0/occluder.onnx"                "$ONNX/occluder.onnx"

# insightface buffalo_l (~300 MB) downloads itself into INSIGHTFACE_HOME.

echo "[init] models ready in $MODELS"
exec python /opt/rt_alphaface_server.py

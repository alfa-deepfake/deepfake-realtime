#!/bin/bash
# seed-vc downloads its own weights from HuggingFace on first run (see
# hf_utils.py): Plachta/Seed-VC, funasr/campplus, FunAudioLLM/CosyVoice-300M,
# plus wav2vec2-xls-r-300m and bigvgan pulled by transformers. ~4.4 GB total.
# Nothing to fetch by hand — this script only makes sure it lands in the volume.
set -euo pipefail

CKPT_DIR="${SEEDVC_CHECKPOINTS:-/models/checkpoints}"
mkdir -p "$CKPT_DIR" "${HF_HOME:-/models/hf}"

# hf_utils.py hardcodes cache_dir="./checkpoints" relative to the working dir,
# so point that at the volume instead of letting it fill the container layer.
if [ ! -L /app/checkpoints ]; then
  rm -rf /app/checkpoints
  ln -sfn "$CKPT_DIR" /app/checkpoints
fi

# Writable dirs the server creates next to itself for recordings/references.
mkdir -p /app/recordings /app/references

echo "[init] starting seed-vc on ${HOST:-0.0.0.0}:${PORT:-17494} (first run downloads ~4.4 GB)"
exec python vc_server.py --host "${HOST:-0.0.0.0}" --port "${PORT:-17494}"

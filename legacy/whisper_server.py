"""Minimal Whisper ASR server over HTTP, backed by the local faster-whisper-medium.

POST /asr  (raw audio bytes or multipart 'file')  ->  {"text", "language", "seconds"}
GET  /health
"""
import os
import io
import time
import argparse
import tempfile

from aiohttp import web
from faster_whisper import WhisperModel

MODEL_DIR = os.environ.get(
    "WHISPER_MODEL",
    os.path.expanduser("~/work/alfa-deepfake/riskapi_and_ml_service/models/asr/faster-whisper-medium"),
)

MODEL = None


def load():
    global MODEL
    dev = os.environ.get("WHISPER_DEVICE", "cuda")
    ct = os.environ.get("WHISPER_COMPUTE", "float16" if dev == "cuda" else "int8")
    try:
        MODEL = WhisperModel(MODEL_DIR, device=dev, compute_type=ct)
        print(f"[whisper] loaded {MODEL_DIR} on {dev}/{ct}", flush=True)
    except Exception as e:
        print(f"[whisper] cuda load failed ({e}); falling back to CPU", flush=True)
        MODEL = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")
        print(f"[whisper] loaded on cpu/int8", flush=True)


async def h_health(request):
    return web.json_response({"ok": True, "model": os.path.basename(MODEL_DIR)})


async def h_asr(request):
    lang = request.query.get("lang")  # e.g. ru; None = autodetect
    data = None
    if request.content_type and request.content_type.startswith("multipart"):
        reader = await request.multipart()
        field = await reader.next()
        data = await field.read()
    else:
        data = await request.read()
    if not data:
        return web.json_response({"error": "no audio"}, status=400)

    fd, path = tempfile.mkstemp(suffix=".audio")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    try:
        t0 = time.perf_counter()
        segments, info = MODEL.transcribe(path, language=lang, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        dt = time.perf_counter() - t0
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return web.json_response({
        "text": text,
        "language": info.language,
        "duration": round(info.duration, 2),
        "seconds": round(dt, 2),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    load()
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.router.add_get("/health", h_health)
    app.router.add_post("/asr", h_asr)
    web.run_app(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()

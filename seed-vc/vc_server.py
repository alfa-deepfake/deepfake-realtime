"""Realtime voice-conversion WS server on top of seed-vc.

Browser streams mic PCM -> server converts chunk-by-chunk (SOLA-stitched) ->
converted audio is accumulated server-side only. Nothing is streamed back live
(no self-monitoring by design). On stop, the recording is muxed to WAV and the
client gets a URL to play it back.
"""
import os
import sys
import io
import time
import uuid
import wave
import asyncio
import argparse
import threading
import importlib.util

os.environ["OMP_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import torchaudio.transforms as tat
from aiohttp import web, WSMsgType

# real-time-gui.py isn't a valid module name; load it by path. Everything above
# its __main__ guard (load_models / custom_infer) is what we need.
_spec = importlib.util.spec_from_file_location("rtg", os.path.join(HERE, "real-time-gui.py"))
rtg = importlib.util.module_from_spec(_spec)
sys.modules["rtg"] = rtg
_spec.loader.exec_module(rtg)

DEVICE = torch.device("cuda")
rtg.device = DEVICE

REC_DIR = os.path.join(HERE, "recordings")
REF_DIR = os.path.join(HERE, "references")
os.makedirs(REC_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

BROWSER_SR = 48000

# Measured, not inherited from real-time-gui.py's defaults (block 0.3 / ce 2.5),
# which were worse on BOTH axes. Inference cost is near-constant per block (~130ms)
# because the CE context is re-encoded every time, so the block size is almost pure
# latency. Measured spk-similarity / latency / RTF:
#   0.30 + ce2.5 -> 0.877 / 430ms / 0.43   (old default)
#   0.20 + ce2.5 -> 0.898 / 330ms / 0.65
#   0.20 + ce1.5 -> 0.901 / 330ms / 0.65   <- here: faster AND closer to target
# RTF 0.65 leaves headroom for AlphaFace sharing the GPU.
BLOCK_TIME = 0.2
CROSSFADE_TIME = 0.04
EXTRA_TIME_CE = 1.5
EXTRA_TIME = 0.5
EXTRA_TIME_RIGHT = 0.02
DIFFUSION_STEPS = 10
INFERENCE_CFG_RATE = 0.7
# Only the first prompt_len seconds of the reference define the target voice.
# Longer is NOT better: measured on russian speech through this server,
#   prompt   WER    spk-sim
#     6s     5.0%   0.821
#    10s     2.5%   0.837
#    15s     2.5%   0.837
#    29s    10.0%   0.815   <- and seed-vc starts warning "max value is 1.0001"
# So feeding the whole reference degrades both intelligibility and similarity.
# Cost is nearly flat (130ms/block at 3s, 145ms at 29s), so this is purely a
# quality choice.
PROMPT_DEFAULT = 12.0
PROMPT_HARD_CAP = 30.0

MODEL_SET = None
MODEL_SR = None
# custom_infer caches the reference prompt in module globals, so only one
# conversion may be in flight at a time.
GPU_LOCK = threading.Lock()


def load():
    global MODEL_SET, MODEL_SR
    ns = argparse.Namespace(
        checkpoint_path=None, config_path=None, fp16=True, gpu=0,
        f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,
    )
    MODEL_SET = rtg.load_models(ns)
    MODEL_SR = MODEL_SET[-1]["sampling_rate"]
    print(f"[vc] model loaded, sr={MODEL_SR}", flush=True)


class Session:
    """Ports GUI.start_vc buffer setup + GUI.audio_callback SOLA loop, driven by
    WS chunks instead of a sounddevice callback."""

    def __init__(self, reference_path, diffusion_steps=DIFFUSION_STEPS,
                 prompt_len=None):
        self.sr = MODEL_SR
        ref, _ = librosa.load(reference_path, sr=self.sr)
        # Leading silence/breath would otherwise eat into the prompt window and
        # leave the model building the target voice out of near-nothing.
        trimmed, _ = librosa.effects.trim(ref, top_db=30)
        self.reference_wav = trimmed if trimmed.size >= self.sr else ref
        full = len(self.reference_wav) / self.sr
        self.prompt_len = float(prompt_len) if prompt_len else min(full, PROMPT_DEFAULT)
        # custom_infer caches its prompt on this name; vary it so a re-tuned
        # prompt length can't silently reuse the previous session's prompt.
        self.reference_path = f"{reference_path}#p{self.prompt_len}"
        # More steps = cleaner timbre but slower; the realtime model needs this
        # tunable to trade quality against keeping up with the mic.
        self.diffusion_steps = int(diffusion_steps)
        self.infer_ms = 0.0

        self.zc = self.sr // 50
        self.block_frame = int(np.round(BLOCK_TIME * self.sr / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(CROSSFADE_TIME * self.sr / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(EXTRA_TIME_CE * self.sr / self.zc)) * self.zc
        self.extra_frame_right = int(np.round(EXTRA_TIME_RIGHT * self.sr / self.zc)) * self.zc

        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame
            + self.block_frame + self.extra_frame_right,
            device=DEVICE, dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc, device=DEVICE, dtype=torch.float32
        )
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=DEVICE, dtype=torch.float32)
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame,
                                         device=DEVICE, dtype=torch.float32)
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window

        self.resampler_in = tat.Resample(orig_freq=BROWSER_SR, new_freq=self.sr,
                                         dtype=torch.float32).to(DEVICE)
        self.pending = np.zeros(0, dtype=np.float32)
        self.out_chunks = []
        self.warmup()

    def warmup(self):
        """Build the reference prompt up front.

        custom_infer computes it lazily on its first call, which costs ~1.8s —
        long enough to starve the client's jitter buffer at the very moment it
        is trying to fill, so the whole stream starts choppy. Pay it here, before
        the client is told the session is ready.
        """
        t0 = time.perf_counter()
        self._process(np.zeros(self.block_frame, dtype=np.float32))
        # discard warmup buffers so the real stream starts clean
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.sola_buffer.zero_()
        self.out_chunks = []
        print(f"[vc] prompt warmed in {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

    def feed(self, pcm48: np.ndarray):
        """Accept client PCM @48k, return the list of converted blocks produced.

        Blocks are also retained in self.out_chunks so a session can be dumped
        to WAV regardless of whether the client is also playing them live.
        """
        produced = []
        if pcm48.size:
            res = self.resampler_in(torch.from_numpy(pcm48).to(DEVICE)).cpu().numpy()
            self.pending = np.concatenate([self.pending, res])
        while self.pending.shape[0] >= self.block_frame:
            block = self.pending[: self.block_frame]
            self.pending = self.pending[self.block_frame:]
            out = self._process(block)
            self.out_chunks.append(out)
            produced.append(out)
        return produced

    def _process(self, indata: np.ndarray) -> np.ndarray:
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(DEVICE)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1):] = torch.from_numpy(
            librosa.resample(
                self.input_wav[-indata.shape[0] - 2 * self.zc:].cpu().numpy(),
                orig_sr=self.sr, target_sr=16000,
            )[320:]
        ).to(DEVICE)

        t0 = time.perf_counter()
        with GPU_LOCK:
            infer_wav = rtg.custom_infer(
                MODEL_SET,
                self.reference_wav,
                self.reference_path,
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                self.diffusion_steps,
                INFERENCE_CFG_RATE,
                self.prompt_len,
                EXTRA_TIME_CE - EXTRA_TIME,
            )
        self.infer_ms = (time.perf_counter() - t0) * 1000

        # SOLA stitch (from DDSP-SVC, as in real-time-gui.py)
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input ** 2,
                     torch.ones(1, 1, self.sola_buffer_frame, device=DEVICE)) + 1e-8
        )
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        sola_offset = torch.argmax(tensor, dim=0).item() if tensor.numel() > 1 else int(tensor.item())

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame: self.block_frame + self.sola_buffer_frame]
        return infer_wav[: self.block_frame].cpu().numpy()

    def to_wav(self) -> bytes:
        audio = np.concatenate(self.out_chunks) if self.out_chunks else np.zeros(0, dtype=np.float32)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(pcm16.tobytes())
        return buf.getvalue()


async def handle_index(request):
    return web.FileResponse(os.path.join(HERE, "vc_index.html"))


async def handle_reference(request):
    reader = await request.multipart()
    field = await reader.next()
    ref_id = uuid.uuid4().hex[:12]
    path = os.path.join(REF_DIR, f"{ref_id}.audio")
    with open(path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    try:
        y, _ = librosa.load(path, sr=MODEL_SR or 22050)
    except Exception as e:
        os.remove(path)
        return web.json_response({"error": f"не удалось прочитать аудио: {e}"}, status=400)
    sr = MODEL_SR or 22050
    dur = len(y) / sr
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    speech = len(trimmed) / sr
    if speech < 2:
        os.remove(path)
        return web.json_response(
            {"error": f"в файле всего {speech:.1f}с речи — нужно минимум 2с (лучше 10+)"},
            status=400)
    # Only the prompt window shapes the target voice, so report what actually
    # gets used rather than the file's length.
    used = min(speech, PROMPT_DEFAULT)
    warn = None
    if speech < 8:
        warn = (f"речи всего {speech:.1f}с — для похожего тембра нужно "
                f"хотя бы 10с чистой речи.")
    return web.json_response({"id": ref_id, "duration": round(dur, 1),
                              "speech": round(speech, 1), "used": round(used, 1),
                              "warn": warn})


async def handle_ws(request):
    ref_id = request.query.get("ref", "")
    # live=1: stream converted blocks straight back (desktop app -> virtual mic).
    # Otherwise the audio is only kept server-side and handed over as WAV on stop.
    live = request.query.get("live") == "1"
    ref_path = os.path.join(REF_DIR, f"{ref_id}.audio")
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)
    if not os.path.exists(ref_path):
        await ws.send_json({"type": "error", "message": "референс не найден"})
        await ws.close()
        return ws

    try:
        steps = int(request.query.get("steps", DIFFUSION_STEPS))
    except ValueError:
        steps = DIFFUSION_STEPS
    steps = max(1, min(50, steps))
    try:
        plen = float(request.query["prompt"]) if "prompt" in request.query else None
    except ValueError:
        plen = None
    if plen is not None:
        plen = max(2.0, min(PROMPT_HARD_CAP, plen))

    loop = asyncio.get_running_loop()
    sess = await loop.run_in_executor(None, Session, ref_path, steps, plen)
    await ws.send_json({"type": "ready", "sr": sess.sr, "steps": steps,
                        "prompt": sess.prompt_len,
                        "ref_used": round(min(len(sess.reference_wav) / sess.sr, sess.prompt_len), 1)})

    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            pcm = np.frombuffer(msg.data, dtype=np.float32).copy()
            try:
                produced = await loop.run_in_executor(None, sess.feed, pcm)
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                break
            if live:
                for out in produced:
                    await ws.send_bytes(out.astype(np.float32).tobytes())
        elif msg.type == WSMsgType.TEXT and msg.data == "stop":
            wav = await loop.run_in_executor(None, sess.to_wav)
            rec_id = uuid.uuid4().hex[:12]
            with open(os.path.join(REC_DIR, f"{rec_id}.wav"), "wb") as f:
                f.write(wav)
            await ws.send_json({"type": "recording", "url": f"/rec/{rec_id}.wav"})
            break
    await ws.close()
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    a = ap.parse_args()
    load()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/", handle_index)
    app.router.add_post("/reference", handle_reference)
    app.router.add_get("/ws", handle_ws)
    app.router.add_static("/rec/", REC_DIR)
    web.run_app(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()

import os, sys, asyncio, time, tempfile
import numpy as np, cv2, torch
import onnxruntime as ort
from aiohttp import web, WSMsgType
from insightface.app import FaceAnalysis
from insightface.utils import face_align

ALPHA_DIR = os.environ.get("ALPHA_DIR", "/home/master/deepfake-realtime/Alphaface_Official")
FF_MODELS = os.environ.get("FF_MODELS", "/home/master/facefusion/.assets/models")
os.chdir(ALPHA_DIR)
sys.path.insert(0, ALPHA_DIR)
from Models.Swapper_AlphaFace import build_AlphaFace  # noqa: E402

PROV = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DEV = "cuda"

print("[init] loading insightface detectors ...", flush=True)
# Full analyzer: detection + landmarks, used to align the SOURCE face (needs kps).
full_app = FaceAnalysis(name="buffalo_l", providers=PROV)
full_app.prepare(ctx_id=0, det_size=(640, 640))
# Light detector for per-frame target detection at 320x320 (faster).
det_app = FaceAnalysis(name="buffalo_l", providers=PROV, allowed_modules=["detection"])
det_app.prepare(ctx_id=0, det_size=(320, 320))

print("[init] building AlphaFace ...", flush=True)
model = build_AlphaFace(config=None, adv_train=False, new_id_model=False)
ckpt = torch.load(os.path.join(ALPHA_DIR, "alphaface_demo.pt"), map_location=DEV)
model.Swapper.load_state_dict(ckpt["swapper"])
model = model.cuda()               # loads cudnn into the process -> ORT below gets CUDA too
model.Swapper.eval()
model.Id_encoder.eval()
print("[init] AlphaFace ready on", DEV, flush=True)

# --- quality helpers: face parsing (bisenet) + face restore (gfpgan) --------------
# ORT CUDA works here only because torch already pulled cudnn into the process
# (standalone ORT gets empty LD_LIBRARY_PATH -> CPU). Load lazily-safe.
def _load_ort(path):
    try:
        s = ort.InferenceSession(path, providers=PROV)
        print("[init] loaded %s on %s" % (os.path.basename(path), s.get_providers()[0]), flush=True)
        return s
    except Exception as e:
        print("[init] FAILED to load %s: %s" % (path, e), flush=True)
        return None

bisenet = _load_ort(os.path.join(FF_MODELS, "bisenet_resnet_34.onnx"))
gfpgan = _load_ort(os.path.join(FF_MODELS, "gfpgan_1.4.onnx"))
# XSeg (DFL) is the primary swap mask: hugs the face, excludes hair/occluders, and stays
# clear of the crop border -> no black square rim. bisenet parse is the fallback.
xseg = _load_ort(os.path.join(FF_MODELS, "xseg_1.onnx"))
# VisoMaster restorers/masks: GPEN gives realistic skin (far less "plastic" than GFPGAN);
# occluder removes hands/objects that pass in front of the face.
VM_MODELS = os.environ.get("VM_MODELS", "/home/master/codex_visomaster_test/VisoMaster/model_assets")
gpen = _load_ort(os.path.join(VM_MODELS, "GPEN-BFR-512.onnx"))
occluder = _load_ort(os.path.join(VM_MODELS, "occluder.onnx"))

BISENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
BISENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
# CelebAMask-HQ classes to keep as the swap region: skin, brows, eyes, nose, mouth, lips.
# Excludes background(0), glasses(6), ears(7,8,9), neck(14,15), cloth(16), hair(17), hat(18).
PARSE_KEEP = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13], np.int64)

STATE = {"source_id": None, "enabled": True,
         "parse": True, "color": True, "enhance": True}   # enhance=GPEN on by default (quality)
_last = {"t": time.time(), "n": 0, "fps": 0.0}
_dbg = {"i": 0}


def _to_tensor_rgb01(bgr):
    """BGR uint8 -> [1,3,H,W] float RGB in [0,1] on cuda."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return t.to(DEV, non_blocking=True)


@torch.no_grad()
def _encode_face(bgr, require_face):
    """BGR frame -> arcface id embedding [1,512] (L2-normalized inside encoder).
    require_face=True: return None when no face is detected (used for video sampling,
    so blank/blurred frames are skipped). require_face=False: fall back to a plain
    112 resize (matches AlphaFace eval for an already-cropped source image)."""
    faces = full_app.get(bgr)
    if faces:
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        aimg = face_align.norm_crop(bgr, faces[0].kps, image_size=112)  # arcface-aligned 112
    elif require_face:
        return None
    else:
        aimg = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA)
    t = _to_tensor_rgb01(aimg)          # [0,1]
    t = t * 2.0 - 1.0                   # -> [-1,1]  (== (img*255)/127.5 - 1)
    return model.Id_encoder(t).detach()  # [1,512]


@torch.no_grad()
def compute_source_id(bgr_img):
    STATE["source_id"] = _encode_face(bgr_img, require_face=False)
    return True, "ok"


@torch.no_grad()
def compute_source_id_video(path, max_samples=48):
    """Sample up to max_samples frames evenly across the video, embed each detected
    face, and store the L2-renormalized mean id. Averaging over pose/lighting gives a
    more stable identity than a single still (quality, not speed -- per-frame swap cost
    is unchanged since the id is still precomputed once)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, "video open failed", 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 0:
        step = max(1, total // max_samples)
        idxs = list(range(0, total, step))[:max_samples]
    else:
        idxs = list(range(max_samples))  # unknown length: read first frames sequentially
    embs = []
    for i in idxs:
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        e = _encode_face(frame, require_face=True)
        if e is not None:
            embs.append(e)
    cap.release()
    if not embs:
        return False, "no face found in video", 0
    sid = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)          # [1,512]
    sid = sid / sid.norm(dim=-1, keepdim=True).clamp_min(1e-8)      # re-normalize
    STATE["source_id"] = sid.detach()
    return True, "ok", len(embs)


BORDER_MARGIN = 6  # px: force the mask to 0 near the crop edge (kills the black square rim)


def face_mask_xseg(bgr256):
    """DFL XSeg soft face mask [256,256] in 0..1 (BGR/255, NHWC in and out). None if off."""
    if xseg is None:
        return None
    x = (cv2.resize(bgr256, (256, 256)).astype(np.float32) / 255.0)[None]  # [1,256,256,3] BGR
    m = xseg.run(None, {"input": x})[0][0][..., 0]
    return np.clip(m, 0.0, 1.0)


def occ_mask(bgr256):
    """Occluder mask [256,256] in 0..1: ~1 on the visible face, ~0 on hands/objects that
    pass in front of it. None if the model is unavailable."""
    if occluder is None:
        return None
    x = cv2.cvtColor(cv2.resize(bgr256, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    o = occluder.run(None, {"img": x.transpose(2, 0, 1)[None]})[0][0][0]
    return 1.0 / (1.0 + np.exp(-o))  # logits -> prob


def build_face_mask(bgr256):
    """High-quality swap mask: XSeg -> bisenet parse -> ellipse fallback, minus occluders,
    softly feathered and forced to 0 near the crop border so no square edge bleeds in."""
    m = face_mask_xseg(bgr256)
    if m is None:
        m = face_parse_mask(bgr256)
    if m is None:
        m = _ellipse_mask()
    o = occ_mask(bgr256)
    if o is not None:
        # lenient: keep the full face (prob~1 -> 1), only carve out confident occluders
        m = np.clip(m, 0.0, 1.0) * np.clip(o * 2.0, 0.0, 1.0)
    m8 = (np.clip(m, 0.0, 1.0) * 255).astype(np.uint8)
    m8 = cv2.erode(m8, np.ones((5, 5), np.uint8), iterations=1)   # pull the seam in a touch
    m8 = cv2.GaussianBlur(m8, (0, 0), sigmaX=5)                    # smooth feather
    m = m8.astype(np.float32) / 255.0
    b = BORDER_MARGIN
    m[:b, :] = 0.0; m[-b:, :] = 0.0; m[:, :b] = 0.0; m[:, -b:] = 0.0
    return m


def face_parse_mask(bgr256):
    """bisenet face-parsing -> soft [256,256] float mask (0..1) of the face region.
    Returns None if the parser is unavailable (falls back to square paste)."""
    if bisenet is None:
        return None
    inp = cv2.resize(bgr256, (512, 512))
    x = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - BISENET_MEAN) / BISENET_STD).transpose(2, 0, 1)[None]
    seg = bisenet.run(None, {"input": x})[0][0]        # [19,512,512]
    cls = seg.argmax(0)                                # [512,512]
    m = np.isin(cls, PARSE_KEEP).astype(np.uint8) * 255
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_LINEAR)
    # shrink 1px off the seam, feather generously so the blend is invisible
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=6).astype(np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)


def color_transfer(src_bgr, ref_bgr, mask):
    """Reinhard LAB color transfer: match swapped-face color stats to the target face
    region under `mask`. Kills the tone/lighting mismatch that reads as a stuck-on mask."""
    m = mask > 0.5
    if int(m.sum()) < 50:
        return src_bgr
    s = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for c in range(3):
        sm, ss = s[..., c][m].mean(), s[..., c][m].std() + 1e-6
        rm, rs = r[..., c][m].mean(), r[..., c][m].std() + 1e-6
        out[..., c] = (s[..., c] - sm) / ss * rs + rm
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def gpen_enhance(bgr256):
    """GPEN-BFR-512 face restore -> realistic skin texture/detail, much less 'plastic' than
    GFPGAN. RGB [-1,1], 512. Falls back to GFPGAN, then to the input, if unavailable."""
    if gpen is None:
        return gfpgan_enhance(bgr256)
    x = cv2.cvtColor(cv2.resize(bgr256, (512, 512)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    o = gpen.run(None, {"input": x})[0][0].transpose(1, 2, 0)
    o = ((np.clip(o, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(o, cv2.COLOR_RGB2BGR), (256, 256))


def gfpgan_enhance(bgr256):
    """GFPGAN 1.4 face restore -> adds realistic skin texture/detail (anti-plastic).
    Runs at 512 then back to 256. Returns input unchanged if the model is unavailable."""
    if gfpgan is None:
        return bgr256
    inp = cv2.resize(bgr256, (512, 512))
    g = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    g = ((g - 0.5) / 0.5).transpose(2, 0, 1)[None]
    o = gfpgan.run(None, {"input": g})[0][0].transpose(1, 2, 0)
    o = (np.clip(o, -1, 1) + 1) / 2 * 255
    o = cv2.cvtColor(o.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(o, (256, 256))


# --- alignment stabilisation + fallback mask (quality: anti-jitter, no border-black) ---
_track = {"kps": None}          # last (smoothed) 5-point landmarks, for temporal EMA
_ELLIPSE = None


def _ellipse_mask():
    """Soft face-shaped oval in the 256 aligned space (fallback when bisenet is off/None).
    Centred slightly high and kept clear of the bottom edge so it never reaches the neck."""
    global _ELLIPSE
    if _ELLIPSE is None:
        m = np.zeros((256, 256), np.uint8)
        cv2.ellipse(m, (128, 118), (92, 108), 0, 0, 360, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=10)
        _ELLIPSE = np.clip(m.astype(np.float32) / 255.0, 0.0, 1.0)
    return _ELLIPSE.copy()


def _smooth_kps(kps, alpha=0.5):
    """Temporal EMA on the 5 landmarks to kill per-frame alignment jitter ('crooked'
    wobble). Resets when the face jumps far (new/other face)."""
    kps = kps.astype(np.float32)
    prev = _track["kps"]
    if prev is not None and np.linalg.norm(kps.mean(0) - prev.mean(0)) < 60:
        kps = alpha * kps + (1.0 - alpha) * prev
    _track["kps"] = kps
    return kps


def _paste_back(img, face256_bgr, M, mask=None):
    h, w = img.shape[:2]
    IM = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(face256_bgr, IM, (w, h), borderValue=0.0)
    if mask is None:
        # legacy square mask: erode hard + blur to hide the crop boundary
        m = np.full((256, 256), 255, dtype=np.uint8)
        warped_mask = cv2.warpAffine(m, IM, (w, h), borderValue=0.0)
        warped_mask = cv2.erode(warped_mask, np.ones((11, 11), np.uint8), iterations=1)
        warped_mask = cv2.GaussianBlur(warped_mask, (15, 15), 0)
    else:
        # mask is already face-shaped, feathered and zero at the crop border; warp + light soften
        m = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        warped_mask = cv2.warpAffine(m, IM, (w, h), borderValue=0.0)
        warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)
    warped_mask = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
    out = warped.astype(np.float32) * warped_mask + img.astype(np.float32) * (1.0 - warped_mask)
    return out.astype(np.uint8)


@torch.no_grad()
def process(frame):
    sid = STATE["source_id"]
    if sid is None or not STATE["enabled"]:
        return frame
    faces = det_app.get(frame)
    if not faces:
        _track["kps"] = None                       # forget track so next face starts clean
        return frame
    # largest face only: swapping every detection (reflections/bg faces) reads as "crooked"
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    kps = _smooth_kps(faces[0].kps)

    M = face_align.estimate_norm(kps, image_size=256)
    aimg = cv2.warpAffine(frame, M, (256, 256), borderValue=0.0)       # target aligned crop
    # valid-content mask: 0 where the aligned crop fell OUTSIDE the frame (black border).
    # Multiplying it into the paste mask is what removes the black patch below the face.
    h, w = frame.shape[:2]
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (256, 256), borderValue=0)
    valid = cv2.erode((valid > 127).astype(np.uint8), np.ones((5, 5), np.uint8), 1).astype(np.float32)

    t = _to_tensor_rgb01(aimg)                     # target: [0,1], no [-1,1] normalize
    sw = model.Swapper(t, sid)                     # [1,3,256,256]
    sw = sw[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()  # RGB256
    sw_bgr = cv2.cvtColor(sw, cv2.COLOR_RGB2BGR)

    mask = build_face_mask(aimg) if STATE["parse"] else _ellipse_mask()  # XSeg HQ mask
    mask = mask * valid                            # never paste border-black

    if STATE["color"]:
        sw_bgr = color_transfer(sw_bgr, aimg, mask)   # match target tone/lighting
    if STATE["enhance"]:
        sw_bgr = gpen_enhance(sw_bgr)                 # GPEN: realistic skin, not plastic

    return _paste_back(frame, sw_bgr, M, mask)


async def h_index(request):
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def h_set_source(request):
    data = await request.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return web.json_response({"ok": False, "msg": "decode failed"})
    ok, msg = await asyncio.get_event_loop().run_in_executor(None, compute_source_id, img)
    return web.json_response({"ok": ok, "msg": msg})


async def h_set_source_video(request):
    data = await request.read()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    try:
        loop = asyncio.get_event_loop()
        ok, msg, n = await loop.run_in_executor(None, compute_source_id_video, path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return web.json_response({"ok": ok, "msg": msg, "frames": n})


async def h_toggle(request):
    STATE["enabled"] = not STATE["enabled"]
    return web.json_response({"enabled": STATE["enabled"]})


async def h_opt(request):
    """Toggle a quality option: parse | color | enhance."""
    name = request.match_info.get("name")
    if name not in ("parse", "color", "enhance"):
        return web.json_response({"ok": False})
    STATE[name] = not STATE[name]
    avail = {"parse": bisenet is not None, "color": True, "enhance": gfpgan is not None}
    return web.json_response({"ok": True, "name": name, "on": STATE[name], "avail": avail[name]})


async def h_ws(request):
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)
    loop = asyncio.get_event_loop()
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            out = await loop.run_in_executor(None, process, frame)
            ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                await ws.send_bytes(enc.tobytes())
            _last["n"] += 1
            now = time.time()
            if now - _last["t"] >= 2.0:
                _last["fps"] = _last["n"] / (now - _last["t"])
                _last["t"], _last["n"] = now, 0
                print("[fps] %.1f  source=%s enabled=%s parse=%s color=%s enh=%s" % (
                    _last["fps"], STATE["source_id"] is not None, STATE["enabled"],
                    STATE["parse"], STATE["color"], STATE["enhance"]), flush=True)
        elif msg.type == WSMsgType.TEXT and msg.data == "ping":
            await ws.send_str("pong")
    return ws


INDEX_HTML = r"""<!doctype html><html><head><meta charset=utf-8>
<title>Realtime AlphaFace (test)</title>
<style>
body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:16px;font-weight:600;margin:0 0 12px}
#wrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
canvas{background:#000;border-radius:8px;max-width:100%}
.col{display:flex;flex-direction:column;gap:6px}
.lbl{font-size:12px;color:#9aa}
.bar{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
button,label.btn{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:13px}
button:hover,label.btn:hover{background:#333}
button.off{opacity:.5}
#stat{font-size:13px;color:#7fd}
#src{width:128px;height:128px;object-fit:cover;border-radius:6px;background:#222;border:1px solid #333}
</style></head><body>
<h1>Realtime AlphaFace — test preview (A100)</h1>
<div class=bar>
  <label class=lbl>Камера: <select id=cam></select></label>
  <label class=btn>Загрузить source (фото/видео)<input id=file type=file accept="image/*,video/*" hidden></label>
  <img id=src alt="source">
  <button id=toggle>Swap: ON</button>
  <button id=mirror>Зеркало: ON</button>
</div>
<div class=bar>
  <button id=parse data-opt=parse>Маска: parse</button>
  <button id=color data-opt=color>Цвет: ON</button>
  <button id=enhance data-opt=enhance class=off>Enhance (GFPGAN): OFF</button>
  <span class=lbl>parse+color убирают «налепленность», GFPGAN добавляет текстуру кожи (−fps)</span>
</div>
<div class=bar>
  <span id=stat>подключение…</span>
  <span id=diag class=lbl></span>
</div>
<div id=wrap>
  <div class=col><span class=lbl>Вебкамера (ноут)</span><canvas id=src_c width=640 height=480></canvas></div>
  <div class=col><span class=lbl>Результат (AlphaFace)</span><canvas id=out_c width=640 height=480></canvas></div>
</div>
<video id=v autoplay playsinline muted style=display:none></video>
<script>
const v=document.getElementById('v'), srcC=document.getElementById('src_c'), outC=document.getElementById('out_c');
const sx=srcC.getContext('2d'), ox=outC.getContext('2d'), stat=document.getElementById('stat'), diag=document.getElementById('diag');
const camSel=document.getElementById('cam');
let mirror=true, enabled=true, inflight=0, frames=0, t0=performance.now(), stream=null, dcount=0;
const MAX_INFLIGHT=2;
const cap=document.createElement('canvas'); cap.width=640; cap.height=480; const cc=cap.getContext('2d',{willReadFrequently:true});
const tiny=document.createElement('canvas'); tiny.width=32; tiny.height=24; const tc=tiny.getContext('2d',{willReadFrequently:true});
let ws;
function connect(){
  ws=new WebSocket('ws://'+location.host+'/ws'); ws.binaryType='arraybuffer';
  ws.onopen=()=>{stat.textContent='подключено к серверу';};
  ws.onclose=()=>{stat.textContent='WS отключён, переподключение…'; inflight=0; setTimeout(connect,1000);};
  ws.onmessage=(e)=>{
    if(inflight>0) inflight--;
    createImageBitmap(new Blob([e.data],{type:'image/jpeg'})).then(bmp=>{
      ox.drawImage(bmp,0,0,outC.width,outC.height); bmp.close();
      frames++; const dt=performance.now()-t0; if(dt>1000){stat.textContent=(frames*1000/dt).toFixed(1)+' fps'; frames=0; t0=performance.now();}
    }).catch(()=>{});
  };
}
function pump(){
  if(v.readyState>=2 && v.videoWidth>0){
    cc.save(); if(mirror){cc.translate(cap.width,0);cc.scale(-1,1);} cc.drawImage(v,0,0,cap.width,cap.height); cc.restore();
    sx.drawImage(cap,0,0,srcC.width,srcC.height);
    if((dcount++ % 20)===0){
      try{ tc.drawImage(cap,0,0,32,24); const d=tc.getImageData(0,0,32,24).data; let s=0; for(let i=0;i<d.length;i+=4) s+=d[i];
        const b=(s/(d.length/4)).toFixed(0); diag.textContent='камера '+v.videoWidth+'x'+v.videoHeight+' · яркость '+b+(b<8?' ⚠ ЧЁРНЫЙ КАДР':''); }catch(e){}
    }
    while(ws&&ws.readyState===1&&inflight<MAX_INFLIGHT){
      inflight++;
      cap.toBlob(b=>{ if(b) b.arrayBuffer().then(a=>ws.send(a)); else inflight--; },'image/jpeg',0.8);
      break;
    }
  } else { diag.textContent='видео не готово (readyState '+v.readyState+', '+v.videoWidth+'x'+v.videoHeight+')'; }
  requestAnimationFrame(pump);
}
async function start(deviceId){
  if(stream){ stream.getTracks().forEach(t=>t.stop()); }
  const c = deviceId ? {deviceId:{exact:deviceId}} : {width:640,height:480,facingMode:'user'};
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:c, audio:false});
    v.srcObject = stream;
    await v.play().catch(()=>{});
    const tr = stream.getVideoTracks()[0];
    stat.textContent = 'камера: '+(tr?tr.label:'?');
    await listCams(deviceId||(tr&&tr.getSettings().deviceId));
  }catch(err){ stat.textContent='ошибка камеры: '+err.name+' — '+err.message; }
}
async function listCams(selId){
  const devs = await navigator.mediaDevices.enumerateDevices();
  const cams = devs.filter(d=>d.kind==='videoinput');
  camSel.innerHTML='';
  cams.forEach((d,i)=>{ const o=document.createElement('option'); o.value=d.deviceId; o.textContent=d.label||('Камера '+(i+1)); if(d.deviceId===selId)o.selected=true; camSel.appendChild(o); });
  diag.textContent='найдено камер: '+cams.length;
}
camSel.onchange=()=>start(camSel.value);
document.getElementById('file').onchange=async(e)=>{
  const f=e.target.files[0]; if(!f)return;
  const isVid=f.type.startsWith('video');
  if(!isVid){ document.getElementById('src').src=URL.createObjectURL(f); }
  stat.textContent = isVid ? 'обрабатываю видео-source…' : 'считаю source id…';
  const url = isVid ? '/set_source_video' : '/set_source';
  const r=await fetch(url,{method:'POST',body:f}); const j=await r.json();
  stat.textContent = j.ok ? ('source установлен ✓'+(isVid?(' ('+j.frames+' кадров)'):'')) : ('ошибка source: '+j.msg);
};
document.getElementById('toggle').onclick=async()=>{ const r=await fetch('/toggle',{method:'POST'}); const j=await r.json(); enabled=j.enabled; document.getElementById('toggle').textContent='Swap: '+(enabled?'ON':'OFF'); };
document.getElementById('mirror').onclick=()=>{ mirror=!mirror; document.getElementById('mirror').textContent='Зеркало: '+(mirror?'ON':'OFF'); };
const LBL={parse:'Маска: ',color:'Цвет: ',enhance:'Enhance (GFPGAN): '};
document.querySelectorAll('button[data-opt]').forEach(btn=>{
  btn.onclick=async()=>{
    const name=btn.dataset.opt;
    const r=await fetch('/opt/'+name,{method:'POST'}); const j=await r.json();
    if(!j.avail){ btn.textContent=LBL[name]+'нет модели'; btn.classList.add('off'); return; }
    const on=j.on; btn.classList.toggle('off',!on);
    btn.textContent=LBL[name]+(name==='parse'?(on?'parse':'квадрат'):(on?'ON':'OFF'));
  };
});
connect(); pump(); start(null);
</script></body></html>"""

app = web.Application(client_max_size=32 * 1024 * 1024)
app.router.add_get("/", h_index)
app.router.add_post("/set_source", h_set_source)
app.router.add_post("/set_source_video", h_set_source_video)
app.router.add_post("/toggle", h_toggle)
app.router.add_post("/opt/{name}", h_opt)
app.router.add_get("/ws", h_ws)

if __name__ == "__main__":
    # Inside a container the bind must be 0.0.0.0, otherwise the published port
    # reaches the container's own loopback and nothing answers. Defaults stay
    # 127.0.0.1:8001 so a bare-metal run behaves exactly as before.
    web.run_app(app,
                host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8001")))

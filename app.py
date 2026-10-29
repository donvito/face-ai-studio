"""InsightFace 2.1 Studio: a local web demo of the InsightFace Python library."""
from __future__ import annotations

import base64
import glob
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore", category=FutureWarning)

import insightface  # noqa: E402
from insightface.app import FaceAnalysis, PersonAnalysis  # noqa: E402
from insightface.utils import face_align  # noqa: E402

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "photos"
VIDEOS = ROOT / "video"
WORK = ROOT / "work"
THUMBS = WORK / "thumbs"
UPLOADS = WORK / "uploads"
PEOPLE = PHOTOS / "people"
DELETED = WORK / "deleted"
NAME_FILE = "name.txt"
OUTPUTS = WORK / "outputs"
for d in (THUMBS, UPLOADS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 1600
VERIFY_THRESHOLD = 0.30
IDENTIFY_THRESHOLD = 0.30
CLUSTER_THRESHOLD = 0.30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Face AI Studio")
infer_lock = threading.Lock()
jobs: dict[str, dict] = {}
job_pool = ThreadPoolExecutor(max_workers=1)

# Models -----------------------------------------------------------------

_models: dict[str, FaceAnalysis] = {}


def face_model(name: str = "buffalo_l") -> FaceAnalysis:
    if name not in _models:
        if name == "buffalo_l":
            fa = FaceAnalysis(name=name, addons=["liveness"], liveness_mode="observe")
        else:
            fa = FaceAnalysis(name=name)
        fa.prepare(ctx_id=0)
        _models[name] = fa
    return _models[name]


def provider_of(fa: FaceAnalysis) -> str:
    det = fa.models.get("detection") or next(iter(fa.models.values()))
    return det.session.get_providers()[0]


# Image helpers ------------------------------------------------------------


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, f"Cannot read image {path.name}")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def safe_media_path(rel: str, base: Path) -> Path:
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(base.resolve())) or not p.is_file():
        raise HTTPException(404, f"Unknown media {rel}")
    return p


async def resolve_image(file: UploadFile | None, sample: str | None) -> tuple[np.ndarray, str]:
    if file is not None and file.filename:
        dest = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower() or '.jpg'}"
        dest.write_bytes(await file.read())
        return read_image(dest), str(dest.relative_to(ROOT))
    if sample:
        p = safe_media_path(sample, ROOT)
        return read_image(p), sample
    raise HTTPException(400, "Provide an image file or a sample path")


def encode_jpg(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def aligned_crop(img: np.ndarray, face) -> str:
    return encode_jpg(face_align.norm_crop(img, landmark=face.kps, image_size=112), 90)


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None


def simulate_screen_replay(img: np.ndarray) -> np.ndarray:
    """Approximate a photo of a screen: resolution loss, RGB subpixel stripes, moire, colour cast, glare, recompression."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.35), int(h * 0.35)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    subpixel_period, stripe_strength = 3.0, 0.35
    for c in range(3):
        out[..., c] *= 1 - stripe_strength + stripe_strength * (0.5 + 0.5 * np.cos(2 * np.pi * (xx / subpixel_period - c / 3)))
    moire = 0.5 + 0.5 * np.sin((xx * 0.93 + yy * 0.31) * 0.35) * np.sin((yy * 0.97 - xx * 0.27) * 0.33)
    out *= (0.8 + 0.2 * moire)[..., None]
    out = out * np.array([1.12, 1.0, 0.82], np.float32) + 18
    glare = np.exp(-(((xx - w * 0.7) / (w * 0.35)) ** 2 + ((yy - h * 0.25) / (h * 0.3)) ** 2))
    out = np.clip(out + (glare * 80)[..., None], 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def tolist(v):
    return None if v is None else np.asarray(v).round(2).tolist()


# Gallery (1:N identification) ------------------------------------------

gallery: dict[str, dict] = {}


def person_label(slug: str) -> str:
    return slug.replace("_", " ").title()


def build_gallery():
    fa = face_model()
    for d in sorted(PEOPLE.iterdir()):
        if d.is_dir():
            try:
                enroll(d.name, person_photos(d), fa)
            except HTTPException as e:
                print(f"Skipping {d.name}: {e.detail}")


def person_photos(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def display_name(d: Path) -> str:
    name_file = d / NAME_FILE
    return name_file.read_text().strip() if name_file.exists() else person_label(d.name)


def enroll(slug: str, paths: list[Path], fa: FaceAnalysis | None = None) -> dict:
    fa = fa or face_model()
    embs, crops, used = [], [], []
    for p in paths:
        try:
            img = read_image(p)
        except HTTPException:
            continue
        with infer_lock:
            f = largest(fa.get(img))
        if f is None or f.embedding is None:
            continue
        embs.append(f.normed_embedding)
        crops.append(aligned_crop(img, f))
        used.append(str(p.relative_to(ROOT)))
    if not embs:
        raise HTTPException(400, "No usable face found in the enrollment photos")
    centroid = np.mean(embs, axis=0)
    gallery[slug] = {
        "name": display_name(PEOPLE / slug),
        "embeddings": np.array(embs),
        "centroid": centroid / np.linalg.norm(centroid),
        "crops": crops,
        "photos": used,
    }
    return gallery[slug]


def identify(emb: np.ndarray) -> list[dict]:
    scores = []
    for slug, g in gallery.items():
        scores.append({"id": slug, "name": g["name"], "score": float(np.max(g["embeddings"] @ emb))})
    return sorted(scores, key=lambda s: -s["score"])


# Routes: catalogue ------------------------------------------------------


def thumb_for(p: Path) -> str:
    t = THUMBS / (str(p.relative_to(ROOT)).replace("/", "__") + ".jpg")
    if not t.exists():
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = 360 / max(h, w)
        cv2.imwrite(str(t), cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
    return "/" + str(t.relative_to(ROOT))


def video_poster(p: Path) -> str:
    t = THUMBS / (p.stem + "_poster.jpg")
    if not t.exists():
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.3))
        ok, fr = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(t), cv2.resize(fr, (480, int(fr.shape[0] * 480 / fr.shape[1]))))
    return "/" + str(t.relative_to(ROOT))


@app.get("/api/info")
def info():
    fa = face_model()
    return {
        "insightface": insightface.__version__,
        "provider": provider_of(fa),
        "models": {k: Path(m.model_file).name for k, m in fa.models.items()},
        "thresholds": {"verify": VERIFY_THRESHOLD, "identify": IDENTIFY_THRESHOLD},
    }


@app.get("/api/samples")
def samples():
    credits = {c["file"]: c for c in json.loads((PHOTOS / "CREDITS.json").read_text())}
    items = []
    for p in sorted(PHOTOS.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = str(p.relative_to(ROOT))
        group = "people" if "/people/" in rel else "groups"
        c = credits.get(rel, {})
        items.append({
            "path": rel,
            "url": "/" + rel,
            "thumb": thumb_for(p),
            "group": group,
            "person": p.parent.name if group == "people" else None,
            "title": c.get("title", p.name).removeprefix("File:"),
            "source": c.get("source", "https://github.com/deepinsight/insightface"),
            "license": c.get("license", "InsightFace sample image"),
        })
    vids = [{"path": str(p.relative_to(ROOT)), "url": "/" + str(p.relative_to(ROOT)), "poster": video_poster(p), "name": p.stem}
            for p in sorted(VIDEOS.glob("*.mp4"))]
    return {"photos": items, "videos": vids}


@app.get("/api/gallery")
def get_gallery():
    return [{"id": k, "name": g["name"], "count": len(g["photos"]), "crops": g["crops"][:4], "photos": g["photos"]}
            for k, g in gallery.items()]


@app.post("/api/gallery/enroll")
async def gallery_enroll(name: str = Form(...), files: list[UploadFile] = File(...)):
    """Save the photos to photos/people/<slug>/ (appending to an existing person) so the gallery survives restarts."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")
    if not slug:
        raise HTTPException(400, "Name is required")
    d = PEOPLE / slug
    is_new = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    added = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        p = d / f"upload_{uuid.uuid4().hex[:12]}{suffix if suffix in IMAGE_EXTS else '.jpg'}"
        p.write_bytes(await f.read())
        added.append(p)
    try:
        g = enroll(slug, person_photos(d))
    except HTTPException:
        g = None
    used = set(g["photos"]) if g else set()
    unusable = [p for p in added if str(p.relative_to(ROOT)) not in used]
    for p in unusable:
        p.unlink()
    if g is None or len(unusable) == len(added):
        if is_new:
            shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "No usable face found in the enrollment photos")
    if is_new or name.strip() != g["name"]:
        (d / NAME_FILE).write_text(name.strip())
        g["name"] = name.strip()
    return {"id": slug, "name": g["name"], "count": len(g["photos"]), "added": len(added) - len(unusable),
            "skipped": len(unusable)}


@app.delete("/api/gallery/{slug}")
def gallery_delete(slug: str):
    """Remove a person from the gallery; their folder is moved to work/deleted/ so it can be restored."""
    if slug not in gallery:
        raise HTTPException(404, "Unknown person")
    gallery.pop(slug)
    d = PEOPLE / slug
    if d.is_dir():
        DELETED.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(DELETED / f"{slug}_{int(time.time())}"))
    return {"ok": True}


# Routes: image analysis -------------------------------------------------


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), sample: str | None = Form(None), model: str = Form("buffalo_l"),
                  spoof: bool = Form(False)):
    if model not in ("buffalo_l", "raccoon_s"):
        raise HTTPException(400, "Unsupported model")
    img, src = await resolve_image(file, sample)
    if spoof:
        img = simulate_screen_replay(img)
    fa = face_model(model)
    t0 = time.perf_counter()
    with infer_lock:
        faces = fa.get(img)
    elapsed = time.perf_counter() - t0
    out = []
    for i, f in enumerate(sorted(faces, key=lambda f: f.bbox[0])):
        live = f.get("liveness")
        matches = identify(f.normed_embedding) if (model == "buffalo_l" and f.get("embedding") is not None) else []
        out.append({
            "index": i,
            "bbox": tolist(f.bbox),
            "det_score": float(f.det_score),
            "kps": tolist(f.kps),
            "landmark_2d_106": tolist(f.get("landmark_2d_106")),
            "landmark_3d_68": tolist(f.get("landmark_3d_68")[:, :2]) if f.get("landmark_3d_68") is not None else None,
            "pose": tolist(f.get("pose")),
            "age": None if f.get("age") is None else int(f.age),
            "sex": f.sex if f.get("gender") is not None else None,
            "liveness": None if live is None else {k: live.get(k) for k in ("status", "is_live", "live_score", "reason")},
            "embedding_dim": None if f.get("embedding") is None else int(f.embedding.shape[0]),
            "embedding_norm": None if f.get("embedding") is None else round(float(f.embedding_norm), 2),
            "embedding_preview": None if f.get("embedding") is None else f.normed_embedding[:48].round(3).tolist(),
            "crop": aligned_crop(img, f),
            "matches": matches[:3],
        })
    h, w = img.shape[:2]
    return {"source": src, "image": encode_jpg(img), "width": w, "height": h, "faces": out,
            "elapsed_ms": round(elapsed * 1000, 1), "model": model, "provider": provider_of(fa)}


@app.post("/api/compare")
async def compare(file_a: UploadFile | None = File(None), sample_a: str | None = Form(None),
                  file_b: UploadFile | None = File(None), sample_b: str | None = Form(None)):
    img_a, src_a = await resolve_image(file_a, sample_a)
    img_b, src_b = await resolve_image(file_b, sample_b)
    fa = face_model()
    t0 = time.perf_counter()
    with infer_lock:
        fa_a, fa_b = largest(fa.get(img_a)), largest(fa.get(img_b))
    if fa_a is None or fa_b is None:
        raise HTTPException(422, "No face found in " + ("image A" if fa_a is None else "image B"))
    sim = float(np.dot(fa_a.normed_embedding, fa_b.normed_embedding))
    return {
        "similarity": sim,
        "threshold": VERIFY_THRESHOLD,
        "same_person": sim >= VERIFY_THRESHOLD,
        "a": {"source": src_a, "crop": aligned_crop(img_a, fa_a), "image": encode_jpg(img_a, 80), "bbox": tolist(fa_a.bbox), "width": img_a.shape[1], "height": img_a.shape[0]},
        "b": {"source": src_b, "crop": aligned_crop(img_b, fa_b), "image": encode_jpg(img_b, 80), "bbox": tolist(fa_b.bbox), "width": img_b.shape[1], "height": img_b.shape[0]},
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


_cluster_cache: dict | None = None


@app.post("/api/cluster")
def cluster(threshold: float = Form(CLUSTER_THRESHOLD)):
    global _cluster_cache
    from sklearn.cluster import AgglomerativeClustering

    fa = face_model()
    t0 = time.perf_counter()
    if _cluster_cache is None:
        rows = []
        for p in sorted(PHOTOS.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            img = read_image(p)
            with infer_lock:
                faces = fa.get(img)
            for f in faces:
                if f.get("embedding") is None or f.det_score < 0.6:
                    continue
                rows.append({"photo": str(p.relative_to(ROOT)), "crop": aligned_crop(img, f), "emb": f.normed_embedding})
        _cluster_cache = {"rows": rows, "embed_ms": round((time.perf_counter() - t0) * 1000)}
    rows = _cluster_cache["rows"]
    X = np.array([r["emb"] for r in rows])
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - threshold).fit_predict(X)
    clusters: dict[int, list] = {}
    for r, lab in zip(rows, labels):
        clusters.setdefault(int(lab), []).append(r)
    result = []
    for lab, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        c = np.mean([m["emb"] for m in members], axis=0)
        guess = identify(c / np.linalg.norm(c))
        name = guess[0]["name"] if guess and guess[0]["score"] >= 0.4 and len(members) > 1 else None
        result.append({"size": len(members), "photos": len({m["photo"] for m in members}), "guess": name,
                       "faces": [{"crop": m["crop"], "photo": m["photo"]} for m in members]})
    return {"faces": len(rows), "clusters": result, "embed_ms": _cluster_cache["embed_ms"],
            "cluster_ms": round((time.perf_counter() - t0) * 1000)}


# Routes: video jobs -----------------------------------------------------


def new_job(kind: str) -> dict:
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"id": jid, "kind": kind, "status": "queued", "progress": 0.0, "stage": "queued", "result": None, "error": None, "started": time.time()}
    return jobs[jid]


def run_job(job: dict, fn):
    def wrapper():
        job["status"] = "running"
        try:
            job["result"] = fn(job)
            job["status"] = "done"
            job["progress"] = 1.0
        except Exception as e:  # surfaced to the UI
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
        job["elapsed_s"] = round(time.time() - job["started"], 1)
    job_pool.submit(wrapper)


async def resolve_video(file: UploadFile | None, sample: str | None) -> Path:
    if file is not None and file.filename:
        dest = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower() or '.mp4'}"
        dest.write_bytes(await file.read())
        return dest
    if sample:
        return safe_media_path(sample, ROOT)
    raise HTTPException(400, "Provide a video file or sample path")


def reference_photos(people: list[str]) -> list[Path]:
    paths = []
    for slug in people:
        g = gallery.get(slug)
        if g:
            paths += [ROOT / p for p in g["photos"]]
    return paths


def to_browser_mp4(src: Path, dest: Path, audio_from: Path | None = None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22", "-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, check=True)


@app.post("/api/video/privateframe")
async def privateframe(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       mode: str = Form("all"), method: str = Form("gaussian"), people: str = Form("")):
    if mode not in ("all", "blur_only", "exempt"):
        raise HTTPException(400, "mode must be all, blur_only or exempt")
    if method not in ("gaussian", "mosaic"):
        raise HTTPException(400, "method must be gaussian or mosaic")
    src = await resolve_video(file, sample)
    selected = [p for p in people.split(",") if p]
    if mode != "all" and not selected:
        raise HTTPException(400, "Pick at least one reference person for this mode")
    job = new_job("privateframe")

    def work(job):
        from insightface.app.privateframe import default_output_paths, run_streaming_pipeline
        from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH

        out_dir = OUTPUTS / job["id"]
        paths = default_output_paths(str(src), str(out_dir))
        overrides = {"render.redaction.method": method, "recognition.mode": mode}
        if mode != "all":
            ref_dir = out_dir / "references"
            ref_dir.mkdir(parents=True, exist_ok=True)
            for p in reference_photos(selected):
                shutil.copy(p, ref_dir / f"{p.parent.name}__{p.name}")
            overrides["recognition.reference_dir"] = str(ref_dir)

        def progress(current, total, stage):
            job["stage"] = stage
            job["progress"] = (current / total) if total else 0.0

        run_streaming_pipeline(config_path=DEFAULT_CONFIG_PATH, input_path=paths.source, workdir=paths.workdir,
                                        result_path=paths.result_json, redacted_path=paths.result_video,
                                        debug_path=None, config_overrides=overrides, progress=progress)
        data = json.loads(Path(paths.result_json).read_text())
        return {"video": "/" + str(Path(paths.result_video).relative_to(ROOT)),
                "json": "/" + str(Path(paths.result_json).relative_to(ROOT)),
                "source": "/" + str(src.relative_to(ROOT)), "summary": summarize_pf(data, mode),
                "mode": mode, "method": method, "people": [person_label(p) for p in selected]}

    run_job(job, work)
    return job


def summarize_pf(data: dict, mode: str) -> dict:
    tracks = (data.get("recognition") or {}).get("tracks") or {}
    obs = data.get("observations") or []
    confirmed = {t for t, r in tracks.items() if r.get("status") == "CONFIRMED"}

    def redacted(track_id):
        if mode == "all":
            return True
        return (track_id in confirmed) == (mode == "blur_only")

    n_redacted = sum(redacted(o.get("track_id")) for o in obs)
    meta = (data.get("source_video") or {}).get("metadata") or {}
    return {"face_tracks": len({o.get("track_id") for o in obs}), "face_regions": len(obs),
            "redacted_face_regions": n_redacted, "kept_face_regions": len(obs) - n_redacted,
            "matched_tracks": len(confirmed), "frame_count": meta.get("frame_count")}


@app.post("/api/video/person")
async def person_video(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       people: str = Form(""), analysis_fps: float = Form(10.0)):
    src = await resolve_video(file, sample)
    selected = [p for p in people.split(",") if p] or list(gallery)
    job = new_job("person")

    def work(job):
        out_dir = OUTPUTS / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(fps / analysis_fps))
        raw = out_dir / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        registrations = []
        timeline: dict[str, list[float]] = {}
        stats = {"face": 0, "body": 0, "unmatched": 0}
        job["stage"] = "loading cheetah_s"
        with PersonAnalysis(name="cheetah_s") as pa:
            for slug in selected:
                r = pa.register(slug, [str(p) for p in reference_photos([slug])])
                registrations.append({"id": slug, "name": person_label(slug), "accepted": r.accepted,
                                      "rejected": [x["reason"] for x in r.rejected]})
            job["stage"] = "analyzing"
            idx, last, infer_s = -1, [], 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % step == 0:
                    t0 = time.perf_counter()
                    last = pa.match(pa.get(frame))
                    pa.update(last)
                    infer_s += time.perf_counter() - t0
                    for m in last:
                        stats[m.matched_by or "unmatched"] += 1
                        if m.person_id:
                            timeline.setdefault(m.person_id, []).append(round(idx / fps, 2))
                writer.write(draw_person(frame, last, idx / fps))
                job["progress"] = idx / total * 0.95
        cap.release()
        writer.release()
        job["stage"] = "encoding"
        out = out_dir / "person_analysis.mp4"
        to_browser_mp4(raw, out, audio_from=src)
        raw.unlink(missing_ok=True)
        appearances = []
        for pid, ts in timeline.items():
            appearances.append({"id": pid, "name": person_label(pid), "first": ts[0], "last": ts[-1],
                                "seconds": round(len(set(ts)) * step / fps, 1), "times": sorted(set(ts))})
        return {"video": "/" + str(out.relative_to(ROOT)), "source": "/" + str(src.relative_to(ROOT)),
                "duration": round(total / fps, 2), "registrations": registrations,
                "appearances": sorted(appearances, key=lambda a: a["first"]), "stats": stats,
                "analyzed_frames": total // step, "avg_ms_per_frame": round(infer_s / max(1, total // step) * 1000, 1)}

    run_job(job, work)
    return job


PALETTE = [(255, 140, 66), (90, 200, 120), (230, 90, 200), (80, 190, 250), (60, 220, 230), (200, 120, 255)]


def color_for(pid: str | None):
    if pid is None:
        return (0, 165, 255)
    return PALETTE[sum(map(ord, str(pid))) % len(PALETTE)]


def draw_person(frame, matches, t):
    out = frame.copy()
    for m in matches:
        p = m.observation
        color = color_for(m.person_id)
        label = person_label(m.person_id) if m.person_id else "unknown"
        if m.person_id:
            label += f" {m.similarity:.2f} ({m.matched_by})"
        if p.body_bbox is not None:
            x1, y1, x2, y2 = map(int, p.body_bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if p.face is not None:
            fx1, fy1, fx2, fy2 = map(int, p.face.bbox)
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), color, 1)
        box = p.body_bbox if p.body_bbox is not None else (p.face.bbox if p.face is not None else None)
        if box is None:
            continue
        x1, y1 = int(box[0]), int(box[1])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 6, y1)
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 8, y), color, -1)
        cv2.putText(out, label, (x1 + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(out, f"{t:5.1f}s  PersonAnalysis cheetah_s", (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[jid]


# Static -----------------------------------------------------------------


@app.on_event("startup")
def startup():
    t0 = time.perf_counter()
    build_gallery()
    face_model("raccoon_s")
    print(f"Models ready and gallery enrolled in {time.perf_counter() - t0:.1f}s ({len(gallery)} people)")


app.mount("/photos", StaticFiles(directory=PHOTOS), name="photos")
app.mount("/video", StaticFiles(directory=VIDEOS), name="video")
app.mount("/work", StaticFiles(directory=WORK), name="work")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")

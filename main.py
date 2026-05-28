"""
Garbage Detection Backend — FastAPI
Serves YOLO detections over REST API.
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import cv2
import math
import numpy as np
import base64
import os
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ultralytics import YOLO

# ─── Config ────────────────────────────────────────────────
MODEL_PATH   = "Weights/best.pt"
CLASS_NAMES  = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']
CONF_THRESH  = 0.30
SAVE_DIR     = Path("Detected_Images")
SAVE_DIR.mkdir(exist_ok=True)

# ─── App Setup ─────────────────────────────────────────────
app = FastAPI(title="GarbageDetector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static frontend files (index.html lives next to main.py)
frontend_dir = Path(__file__).parent / "static"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Load model once at startup
print("Loading YOLO model…")
model = YOLO(MODEL_PATH)
print("Model loaded ✓")


# ─── Helpers ───────────────────────────────────────────────
def run_detection(img_bgr: np.ndarray, conf_thresh: float = CONF_THRESH):
    """Run YOLO on a BGR image, return annotated image + detection list."""
    results   = model(img_bgr)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls       = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            label     = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

            # Draw bounding box
            color = (0, 200, 80)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # Pill-shaped label background
            text      = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            pad       = 5
            cv2.rectangle(img_bgr,
                          (x1, y1 - th - 2*pad),
                          (x1 + tw + 2*pad, y1),
                          color, -1)
            cv2.putText(img_bgr, text,
                        (x1 + pad, y1 - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
            })

    return img_bgr, detections


def encode_image(img_bgr: np.ndarray) -> str:
    """Encode BGR image → base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


# ─── Models ────────────────────────────────────────────────
class FramePayload(BaseModel):
    """Base64-encoded JPEG frame from the browser webcam."""
    image: str          # data:image/jpeg;base64,<data>  OR plain base64
    conf:  Optional[float] = CONF_THRESH


# ─── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "GarbageDetector API running — open index.html in browser"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = CONF_THRESH):
    """Accept an image file, return annotated image + detections as JSON."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    raw   = await file.read()
    arr   = np.frombuffer(raw, np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    annotated, detections = run_detection(img.copy(), conf)
    return JSONResponse({
        "detections":       detections,
        "count":            len(detections),
        "annotated_image":  encode_image(annotated),
    })


@app.post("/detect/frame")
async def detect_frame(payload: FramePayload):
    """Accept a base64 JPEG frame (from browser webcam), return detections."""
    raw_b64 = payload.image
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode frame")

    annotated, detections = run_detection(img.copy(), payload.conf)
    return JSONResponse({
        "detections":       detections,
        "count":            len(detections),
        "annotated_image":  encode_image(annotated),
    })


@app.post("/save")
async def save_image(payload: FramePayload):
    """Save an annotated frame to disk."""
    raw_b64 = payload.image
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    raw = base64.b64decode(raw_b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    fname = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    fpath = SAVE_DIR / fname
    cv2.imwrite(str(fpath), img)
    return {"saved": fname, "path": str(fpath)}


@app.get("/saved")
def list_saved():
    files = sorted(SAVE_DIR.glob("*.jpg"), reverse=True)
    return {"files": [f.name for f in files[:50]]}
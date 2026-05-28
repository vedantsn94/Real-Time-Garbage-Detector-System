# WasteEye — Garbage Detection Web App

A real-time garbage detection website powered by YOLOv8 + FastAPI.

## Project Structure

```
GarbageDetector/
├── main.py              ← FastAPI backend (replaces app.py)
├── requirements.txt
├── static/
│   └── index.html       ← Frontend website
├── Weights/
│   └── best.pt          ← Your trained model (already here)
├── Media/               ← Test images/videos (already here)
└── Detected_Images/     ← Auto-created; saved frames go here
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the website
Go to **http://localhost:8000** in your browser.

---

## Features

| Mode | How it works |
|------|-------------|
| **Upload Image** | Drag & drop or browse a file → YOLO detects → view annotated result |
| **Live Camera** | Uses your browser webcam → sends frames to backend every 300ms |
| **Mobile Camera** | Enter IP Webcam URL (Android) → polls `/shot.jpg` snapshot endpoint |
| **Saved** | View all previously saved detection screenshots |

---

## Deploying to the web (optional)

To make this accessible from anywhere:

```bash
# Using ngrok (quick public URL)
ngrok http 8000
```

Or deploy to **Railway**, **Render**, or **Fly.io** — all support FastAPI apps with GPU/CPU inference. Upload your `Weights/best.pt` as part of the repo.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Check if server is running |
| `POST` | `/detect/image` | Upload image file → returns annotated image + detections |
| `POST` | `/detect/frame` | Send base64 frame → returns annotated frame + detections |
| `POST` | `/save` | Save an annotated frame to `Detected_Images/` |
| `GET`  | `/saved` | List all saved images |
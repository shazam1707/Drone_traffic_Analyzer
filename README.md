# Smart Drone Traffic Analyzer

A desktop application for automated vehicle detection, tracking, and reporting from drone footage. Built with PyQt6, YOLOv8, and ByteTrack — processes video locally, requires no internet connection, and exports a structured Excel report.

---

## Table of Contents

1. [Demo](#demo)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Local Setup](#local-setup)
5. [Usage](#usage)
6. [Architecture](#architecture)
7. [Tracking Methodology & Edge Cases](#tracking-methodology--edge-cases)
8. [Engineering Assumptions](#engineering-assumptions)
9. [Project Structure](#project-structure)
10. [Known Limitations](#known-limitations)

---

## Features

- Upload any drone video (`.mp4`, `.avi`, `.mov`) and get real-time annotated playback
- Detects and tracks cars, buses, and trucks using YOLOv8 + ByteTrack
- Counts each unique vehicle exactly once across the entire video
- Exports a two-sheet Excel report: headline summary + full per-vehicle detection log
- Cancel processing at any time without crashing
- Runs entirely offline — no API calls, no cloud dependency

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 |  |
| PyQt6 | ≥ 6.5 | Desktop GUI and threading |
| OpenCV (`opencv-python`) | ≥ 4.8 | Video I/O, frame annotation |
| Ultralytics | ≥ 8.0 | YOLOv8 inference |
| Supervision | ≥ 0.18 | ByteTrack wrapper, `sv.Detections` |
| openpyxl | ≥ 3.1 | Excel report generation |
| NumPy | ≥ 1.24 | Frame array manipulation |



---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/shazam1707/drone-traffic-analyzer.git
cd smart-drone-traffic-analyzer
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install PyQt6 opencv-python ultralytics supervision openpyxl numpy
```

### 4. Download the YOLO model weights

On first run, Ultralytics will automatically download `yolov8n.pt` (~6 MB) into the working directory. If you are in a restricted network environment, download it manually:

```bash
# From the project root
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

To use a more accurate (but slower) model, replace `YOLO_MODEL = "yolov8n.pt"` with `"yolov8s.pt"` or `"yolov8m.pt"` in `core/pipeline.py`.

### 5. Run the application

```bash
python main.py
```

---

## Usage

1. Click **Upload video** and select a drone video file.
2. Processing starts immediately — annotated frames appear in the left panel in real time.
3. The right panel shows a live detection log with each new unique vehicle as it is first spotted.
4. The progress bar tracks completion. The status bar shows frame rate and running vehicle count.
5. Click **Cancel** at any time to stop processing gracefully.
6. When complete, click **Download report** to save the Excel file to a location of your choice.

**Output files** are written alongside the source video automatically:
- `<video_name>_annotated.mp4` — original video with bounding boxes and track IDs overlaid
- `<video_name>_report.xlsx` — two-sheet Excel report

---

## Architecture

### Overview

The application is split into four modules with strict separation of concerns:

```
main.py               Entry point — instantiates MainWindow
ui/main_window.py     PyQt6 GUI — layout, buttons, table, video display
core/worker.py        QObject worker — bridges GUI thread and pipeline
core/pipeline.py      Pure CV pipeline — YOLO + ByteTrack, zero Qt imports
core/reporter.py      Excel report writer — openpyxl, zero Qt imports
```

### Thread model

PyQt6 requires that all UI updates happen on the **main thread**. Running video inference on the main thread would freeze the GUI entirely. The solution is a standard Qt worker-thread pattern:

```
Main thread (UI)                  Background thread
────────────────                  ─────────────────
MainWindow                        QThread
  │                                 │
  └─ creates VideoWorker            └─ VideoWorker.run()
       moveToThread(thread)               │
                                          └─ process_video()
                                               │
                                    callbacks fire signals
                                    ↓ (thread-safe Qt signal queue)
                                MainWindow slots update UI
```

`VideoWorker` is a `QObject` moved onto a `QThread`. When the thread starts, it calls `worker.run()`. All communication back to the UI happens exclusively through **PyQt signals**, which Qt automatically delivers on the correct thread via its internal event queue. This means `_update_frame`, `_add_table_row`, and `_on_finished` always execute safely on the main thread — no manual locking required.

The `stop_flag` is a plain Python `lambda: self._stop` checked at the top of every frame loop iteration. This is intentional: it avoids `QMutex` complexity for a single boolean, and the worst-case latency before cancellation is one frame's processing time — imperceptible to the user.

### Signal contract

| Signal | Payload | Fires when |
|---|---|---|
| `frame_ready` | `np.ndarray` (BGR) | Every processed frame |
| `vehicle_detected` | `dict` | First time a tracker ID is seen |
| `progress_updated` | `int` (0–100) | Every frame (including skipped) |
| `status_message` | `str` | Every 30 processed frames + key events |
| `finished` | `dict` | Pipeline completes successfully |
| `error_occurred` | `str` | Any unhandled exception |

### Pipeline has zero Qt imports

`core/pipeline.py` and `core/reporter.py` contain no PyQt6 code whatsoever. The pipeline communicates entirely through plain Python callables passed in from `VideoWorker`. This keeps the CV logic independently testable — you can call `process_video()` from a script, a test suite, or a CLI without importing Qt at all.

---

## Tracking Methodology & Edge Cases

### Detection

YOLOv8 small (`yolov8s.pt`) runs on every Nth frame (default `FRAME_SKIP = 2`). Only COCO class IDs 2 (car), 3 (mtorcycle), 5 (bus), 6 (train) and 7 (truck) are requested — passing `classes=VEHICLE_CLASSES` to the model means non-vehicle detections are filtered before they ever reach the tracker, reducing noise and compute.

### Tracking with ByteTrack

Raw YOLO detections are fed into ByteTrack via the Supervision wrapper (`sv.ByteTrack`). ByteTrack assigns a stable `tracker_id` to each vehicle and maintains that ID across frames even through brief occlusions or missed detections.

Key parameters and their effect:

| Parameter | Value | Rationale |
|---|---|---|
| `track_activation_threshold` | 0.35 | Minimum confidence for a detection to activate a new track. Lower values catch more vehicles but increase false positives. |
| `lost_track_buffer` | 45 frames | How long ByteTrack keeps a track alive after the vehicle disappears (e.g. behind a tree). At 25 fps this is ~1.8 seconds. |
| `minimum_matching_threshold` | 0.8 | IoU threshold for matching a detection to an existing track. Higher values reduce ID switches on dense traffic. |

### Preventing double-counting

The core anti-duplication mechanism is the `unique_vehicles` dictionary in `pipeline.py`, keyed by `tracker_id`:

```python
if tid not in unique_vehicles:
    record = { ... }
    unique_vehicles[tid] = record
    vehicle_callback(record)
```

A vehicle is logged exactly once — the first frame it receives a tracker ID. All subsequent detections of the same vehicle in later frames update ByteTrack's internal state but do not produce a new record. This means the reported count reflects unique physical vehicles, not detection events.

### Edge cases handled

**Vehicles leaving and re-entering the frame:** ByteTrack's `lost_track_buffer` (45 frames) keeps a track alive during brief exits. If the vehicle returns within that window it receives the same ID and is not double-counted. If it returns after the buffer expires, ByteTrack assigns a new ID and it is counted again — this is intentional and correct behaviour for a traffic counter (the vehicle made a second pass).

**Overlapping vehicles / occlusion:** ByteTrack uses both IoU overlap and a Kalman filter to predict where a vehicle should be on the next frame. Partial occlusion — one vehicle briefly hidden behind another — is handled by the Kalman prediction maintaining the track through the hidden frames.

**Frame skipping continuity:** On skipped frames the raw (unannotated) frame is still written to the output video to preserve correct frame count and audio sync. Detection simply does not run on those frames. ByteTrack is not updated on skipped frames, which is a known trade-off — very fast-moving vehicles could theoretically be missed if they traverse the entire field of view within a single skipped frame.

**Bounding box coordinate scaling:** When a frame is resized for inference (width > 1280px), YOLO returns bounding box coordinates in the smaller frame's coordinate space. These are scaled back to original resolution before being passed to ByteTrack and the annotator:

```python
if scale < 1.0 and len(detections) > 0:
    detections.xyxy = detections.xyxy / scale
```

Without this correction, boxes would be drawn in the wrong position on the annotated output video.

---

## Engineering Assumptions

### Camera motion and angle

The system assumes **nadir or near-nadir drone footage** — the camera is pointing straight down or at a shallow downward angle, with the drone hovering or moving slowly. This is the typical configuration for traffic surveys.

It does **not** account for:
- **Severe camera motion:** if the drone is panning rapidly, ByteTrack's Kalman filter predictions will diverge from actual vehicle positions, causing ID switches and potentially double-counting. A production system would incorporate camera motion compensation (homography estimation between consecutive frames) before running the tracker.
- **Oblique angles:** steep side-on angles cause vehicles to appear significantly elongated and change shape as they move relative to the camera. The COCO-trained model handles moderate oblique angles reasonably well but may degrade on footage taken from below 30° elevation.
- **Camera shake / vibration:** minor vibration is tolerated by ByteTrack's IoU matching. Severe shake may require stabilisation pre-processing.

### Model utilisation

YOLOv8 small was chosen as the default for three reasons: it ships as a single ~6 MB file with no separate download step, it runs at real-time speeds on CPU, and its COCO training includes sufficient drone-perspective vehicle examples for reasonable accuracy. It is not fine-tuned on aerial imagery.

### Accuracy vs inference speed trade-off

The two primary levers for this trade-off are `FRAME_SKIP` and `YOLO_MODEL`:

| Configuration | Processing speed | Accuracy risk |
|---|---|---|
| `yolov8n.pt`, `FRAME_SKIP=2` | ~15–20 fps (CPU) | May miss fast vehicles; nano model has lower mAP |
| `yolov8s.pt`, `FRAME_SKIP=2` | ~8–12 fps (CPU) | Better detection; same skip risk |
| `yolov8n.pt`, `FRAME_SKIP=1` | ~8–10 fps (CPU) | Highest recall; slowest on CPU |
| `yolov8n.pt`, `FRAME_SKIP=2` | ~60+ fps (GPU) | Recommended for production |



The `MAX_DIMENSION = 1280` resize cap provides an additional speed gain on 4K footage with minimal accuracy loss, since YOLOv8's default inference resolution is 640px — frames are already being downsampled internally. Capping at 1280px before inference avoids unnecessary memory bandwidth with no meaningful detection quality trade-off.


### Output video quality

The annotated output video is written at the **original input resolution** using the `mp4v` codec. This prioritises visual quality for review purposes. File sizes can be large for long 4K recordings. A production pipeline might add a configurable output resolution or use `h264` encoding (requires OpenCV built with FFmpeg).

---

## Project Structure

```
smart-drone-traffic-analyzer/
│
├── main.py                  # Entry point
├── requirements.txt
├── README.md
│
├── ui/
│   └── main_window.py       # PyQt6 MainWindow — all GUI code
│
└── core/
    ├── worker.py            # VideoWorker QObject — thread bridge
    ├── pipeline.py          # CV pipeline — YOLO + ByteTrack
    └── reporter.py          # Excel report writer
```

---


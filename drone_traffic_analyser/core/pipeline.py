# core/pipeline.py

import cv2
import numpy as np
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import supervision as sv


# ── Constants ─────────────────────────────────────────────────────────────────

VEHICLE_CLASSES = [2, 3, 5, 6, 7]   # car=2, motorcycle=3, bus=5, train=6, truck=7

CLASS_COLORS = {
    "car":        (86, 180, 233),    # blue
    "motorcycle": (240, 228, 66),    # yellow
    "bus":        (230, 159, 0),     # orange
    "train":      (204, 121, 167),   # pink/purple
    "truck":      (0, 158, 115),     # teal/green
}

YOLO_MODEL      = "yolov8s.pt"   # nano — fast; swap to yolov8s.pt for accuracy
CONFIDENCE      = 0.35            # detection confidence threshold
IOU_THRESHOLD   = 0.45            # NMS IoU threshold
FRAME_SKIP      = 2               # process every Nth frame (1 = every frame)
TRACK_BUFFER    = 45              # frames ByteTrack holds a lost track alive
MAX_DIMENSION   = 1280            # resize large frames to this width for speed


# ── Helper: frame resize ───────────────────────────────────────────────────────

def _resize_if_needed(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Downscale frame so its width doesn't exceed MAX_DIMENSION.
    Returns (resized_frame, scale_factor).
    scale_factor < 1.0 means the frame was shrunk.
    """
    h, w = frame.shape[:2]
    if w <= MAX_DIMENSION:
        return frame, 1.0
    scale = MAX_DIMENSION / w
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


# ── Helper: BGR frame → annotated frame ───────────────────────────────────────

def _annotate(
    frame: np.ndarray,
    detections: sv.Detections,
    class_names: dict,
) -> np.ndarray:
    """
    Draw bounding boxes and track-ID labels onto a copy of frame.
    Uses per-class colors defined in CLASS_COLORS.
    """
    annotated = frame.copy()

    for xyxy, tracker_id, class_id in zip(
        detections.xyxy,
        detections.tracker_id if detections.tracker_id is not None else [],
        detections.class_id   if detections.class_id   is not None else [],
    ):
        x1, y1, x2, y2 = map(int, xyxy)
        class_name = class_names.get(int(class_id), "vehicle")
        color = CLASS_COLORS.get(class_name, (200, 200, 200))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Label background + text
        label = f"#{tracker_id} {class_name}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - lh - baseline - 4),
            (x1 + lw + 4, y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


# ── Main pipeline function ─────────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_video_path: str,
    progress_callback,       # callable(int)        — emits 0–100
    frame_callback,          # callable(np.ndarray) — emits annotated BGR frame
    vehicle_callback,        # callable(dict)       — emits new vehicle record
    status_callback,         # callable(str)        — emits status text
    stop_flag,               # callable() → bool    — returns True if cancelled
) -> dict:
    """
    Full CV pipeline: load video → detect → track → annotate → write output.

    Returns a summary dict:
        {
            "total":       int,
            "duration_s":  float,
            "vehicles":    list[dict],
            "report_path": str,         (filled by caller/worker)
        }

    All UI updates go through the callbacks — this function has zero Qt imports.
    """

    # ── 1. Open video ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    status_callback(f"Opened video: {orig_w}×{orig_h} @ {fps:.1f} fps, {total_frames} frames")

    # ── 2. Video writer (output annotated video) ───────────────
    # We write at original resolution for quality
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))

    # ── 3. Load model + tracker ────────────────────────────────
    status_callback("Loading YOLO model…")
    model   = YOLO(YOLO_MODEL)
    tracker = sv.ByteTrack(track_activation_threshold=CONFIDENCE,
                           lost_track_buffer=TRACK_BUFFER,
                           minimum_matching_threshold=0.8,
                           frame_rate=int(fps))

    class_names = model.names   # {0: 'person', 2: 'car', ...}

    # ── 4. State ───────────────────────────────────────────────
    unique_vehicles: dict[int, dict] = {}   # tracker_id → vehicle record
    start_time = time.time()
    frame_idx  = 0

    # ── 5. Main loop ───────────────────────────────────────────
    while True:
        if stop_flag():
            status_callback("Processing cancelled by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Progress (always update, even on skipped frames)
        progress = min(int((frame_idx / total_frames) * 100), 99)
        progress_callback(progress)

        # Frame skip — write raw frame so output video has correct length
        if frame_idx % FRAME_SKIP != 0:
            writer.write(frame)
            continue

        # ── 5a. Resize for inference (speed optimisation) ──────
        small_frame, scale = _resize_if_needed(frame)

        # ── 5b. YOLO detection ─────────────────────────────────
        results = model(
            small_frame,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        # Scale bboxes back up to original resolution if we resized
        if scale < 1.0 and len(detections) > 0:
            detections.xyxy = detections.xyxy / scale

        # ── 5c. ByteTrack update ───────────────────────────────
        detections = tracker.update_with_detections(detections)

        # ── 5d. Unique vehicle logic ───────────────────────────
        if detections.tracker_id is not None:
            for tracker_id, class_id in zip(
                detections.tracker_id, detections.class_id
            ):
                tid = int(tracker_id)
                if tid not in unique_vehicles:
                    record = {
                        "tracker_id":  tid,
                        "class":       class_names.get(int(class_id), "vehicle"),
                        "first_frame": frame_idx,
                        "timestamp_s": round(frame_idx / fps, 2),
                    }
                    unique_vehicles[tid] = record
                    vehicle_callback(record)   # live table update in UI

        # ── 5e. Annotate frame + emit to UI ────────────────────
        annotated = _annotate(frame, detections, class_names)
        writer.write(annotated)
        frame_callback(annotated)   # shown in QLabel

        # ── 5f. Status update every 30 processed frames ────────
        if frame_idx % (FRAME_SKIP * 30) == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            status_callback(
                f"Frame {frame_idx}/{total_frames} — "
                f"{len(unique_vehicles)} vehicles — "
                f"{fps_actual:.1f} fps processing"
            )

    # ── 6. Cleanup ─────────────────────────────────────────────
    cap.release()
    writer.release()

    duration = round(time.time() - start_time, 2)
    progress_callback(100)

    # ── 7. Summary ─────────────────────────────────────────────
    type_counts = Counter(v["class"] for v in unique_vehicles.values())
    status_callback(
        f"Done — {len(unique_vehicles)} unique vehicles in {duration}s"
    )

    return {
        "total":       len(unique_vehicles),
        "duration_s":  duration,
        "type_counts": dict(type_counts),
        "vehicles":    list(unique_vehicles.values()),
    }
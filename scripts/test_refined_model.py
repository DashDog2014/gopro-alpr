from __future__ import annotations

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100000"
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import json
import math
import subprocess
from typing import Optional

import cv2
from ultralytics import YOLO


# ----------------------------
# Config
# ----------------------------
VIDEO_PATH = Path("data/remuxed/GH340092_clean.MP4")   # change as needed
MODEL_PATH = Path("runs/detect/runs/model_compare/finetuned/weights/best.pt")  # change if needed
OUTPUT_CSV = Path("data/output/tracked_vehicle_events.csv")

SAVE_BEST_CROPS = True
CROPS_DIR = Path("data/output/best_crops")

DEVICE = 0                  # set to "cpu" if needed
CONF_THRESH = 0.4
IMG_SIZE = 960
TRACKER_CFG = "bytetrack.yaml"   # built into ultralytics
SAMPLE_EVERY_N_FRAMES = 1        # set >1 if you want to skip frames
MAX_MISSED_FRAMES = 15           # finalize a track if unseen this many processed frames

UTC = timezone.utc


# ----------------------------
# Metadata helpers
# ----------------------------
def get_video_creation_time_utc(video_path: Path) -> Optional[datetime]:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_entries", "format_tags=creation_time",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        creation = (data.get("format", {}).get("tags", {}) or {}).get("creation_time")
        if creation:
            return datetime.fromisoformat(creation.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        pass
    return None


# ----------------------------
# Color estimation
# ----------------------------
def estimate_vehicle_color(crop_bgr) -> tuple[str, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return "", 0.0

    # Center crop to reduce road/background
    y1 = int(h * 0.2)
    y2 = int(h * 0.8)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    roi = crop_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        roi = crop_bgr

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return "", 0.0

    masks = {
        "black": (V < 50),
        "white": (S < 30) & (V > 190),
        "gray": (S < 40) & (V >= 50) & (V <= 190),
        "red": (((H < 10) | (H > 170)) & (S > 60) & (V > 50)),
        "orange": ((H >= 10) & (H < 20) & (S > 60) & (V > 60)),
        "yellow": ((H >= 20) & (H < 35) & (S > 50) & (V > 70)),
        "green": ((H >= 35) & (H < 85) & (S > 50) & (V > 50)),
        "blue": ((H >= 85) & (H < 135) & (S > 50) & (V > 50)),
        "brown": ((H >= 5) & (H < 25) & (S > 50) & (V >= 40) & (V < 140)),
    }

    scores = {k: float(mask.sum()) / total for k, mask in masks.items()}
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < 0.08:
        return "unknown", best_score
    return best_label, best_score


# ----------------------------
# Tracking data model
# ----------------------------
@dataclass
class BestObservation:
    frame_idx: int
    timestamp_utc: str
    video_time_sec: float
    class_name: str
    class_conf: float
    color_name: str
    color_conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    crop_path: str
    score: float


@dataclass
class TrackState:
    track_id: int
    video_source: str
    first_frame_idx: int
    last_frame_idx: int
    first_seen_time_utc: str
    last_seen_time_utc: str
    first_seen_video_sec: float
    last_seen_video_sec: float
    frames_seen: int = 0
    best_obs: Optional[BestObservation] = None
    class_votes: dict[str, float] = field(default_factory=dict)
    color_votes: dict[str, float] = field(default_factory=dict)


# ----------------------------
# Scoring helpers
# ----------------------------
def compute_best_frame_score(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    conf: float,
    frame_w: int,
    frame_h: int,
) -> float:
    area = max(0, x2 - x1) * max(0, y2 - y1)

    # Penalize boxes near the edges
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(cx - frame_w / 2.0) / (frame_w / 2.0) if frame_w else 1.0
    dy = abs(cy - frame_h / 2.0) / (frame_h / 2.0) if frame_h else 1.0
    center_penalty = 1.0 - 0.35 * min(1.0, math.sqrt(dx * dx + dy * dy))

    return area * max(conf, 1e-6) * center_penalty


def choose_vote_label(votes: dict[str, float]) -> str:
    if not votes:
        return ""
    return max(votes, key=votes.get)


# ----------------------------
# Finalization
# ----------------------------
def finalize_track(writer, track: TrackState):
    if track.best_obs is None:
        return

    best = track.best_obs
    final_class = choose_vote_label(track.class_votes) or best.class_name
    final_color = choose_vote_label(track.color_votes) or best.color_name

    writer.writerow([
        track.track_id,
        track.video_source,
        track.first_frame_idx,
        track.last_frame_idx,
        track.frames_seen,
        track.first_seen_time_utc,
        track.last_seen_time_utc,
        track.first_seen_video_sec,
        track.last_seen_video_sec,
        best.frame_idx,
        best.timestamp_utc,
        best.video_time_sec,
        final_class,
        best.class_conf,
        final_color,
        best.color_conf,
        "",   # brand
        "",   # brand_conf
        "",   # model
        "",   # model_conf
        best.x1,
        best.y1,
        best.x2,
        best.y2,
        best.crop_path,
    ])


# ----------------------------
# Main
# ----------------------------
def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_BEST_CROPS:
        CROPS_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    video_source = VIDEO_PATH.name
    video_start_utc = get_video_creation_time_utc(VIDEO_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not read FPS")

    print(f"Video: {video_source}")
    print(f"Model: {MODEL_PATH}")
    print(f"FPS: {fps:.3f}")
    print(f"Using tracker: {TRACKER_CFG}")
    if video_start_utc:
        print(f"creation_time UTC: {video_start_utc.isoformat()}")

    active_tracks: dict[int, TrackState] = {}

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id",
            "video_source",
            "first_frame_idx",
            "last_frame_idx",
            "frames_seen",
            "first_seen_time_utc",
            "last_seen_time_utc",
            "first_seen_video_sec",
            "last_seen_video_sec",
            "best_frame_idx",
            "best_time_utc",
            "best_video_time_sec",
            "vehicle_class",
            "class_conf",
            "vehicle_color",
            "color_conf",
            "brand",
            "brand_conf",
            "model",
            "model_conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "crop_path",
        ])

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
                frame_idx += 1
                continue

            video_time_sec = frame_idx / fps
            timestamp_utc = ""
            if video_start_utc is not None:
                timestamp_utc = (video_start_utc + timedelta(seconds=video_time_sec)).isoformat()

            results = model.track(
                source=frame,
                persist=True,
                conf=CONF_THRESH,
                imgsz=IMG_SIZE,
                tracker=TRACKER_CFG,
                device=DEVICE,
                verbose=False,
            )

            r = results[0]
            frame_h, frame_w = frame.shape[:2]

            seen_track_ids_this_frame = set()

            if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                track_ids = r.boxes.id.int().cpu().numpy()
                names = r.names

                for box, conf, cls_id, track_id in zip(xyxy, confs, class_ids, track_ids):
                    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_w, x2)
                    y2 = min(frame_h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    seen_track_ids_this_frame.add(int(track_id))
                    class_name = names[int(cls_id)]

                    crop = frame[y1:y2, x1:x2]
                    color_name, color_conf = estimate_vehicle_color(crop)

                    # create or update track state
                    if int(track_id) not in active_tracks:
                        active_tracks[int(track_id)] = TrackState(
                            track_id=int(track_id),
                            video_source=video_source,
                            first_frame_idx=frame_idx,
                            last_frame_idx=frame_idx,
                            first_seen_time_utc=timestamp_utc,
                            last_seen_time_utc=timestamp_utc,
                            first_seen_video_sec=video_time_sec,
                            last_seen_video_sec=video_time_sec,
                        )

                    state = active_tracks[int(track_id)]
                    state.last_frame_idx = frame_idx
                    state.last_seen_time_utc = timestamp_utc
                    state.last_seen_video_sec = video_time_sec
                    state.frames_seen += 1
                    state.class_votes[class_name] = state.class_votes.get(class_name, 0.0) + float(conf)
                    if color_name:
                        state.color_votes[color_name] = state.color_votes.get(color_name, 0.0) + float(color_conf)

                    best_score = compute_best_frame_score(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        conf=float(conf),
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                    should_replace = (
                        state.best_obs is None or
                        best_score > state.best_obs.score
                    )

                    if should_replace:
                        crop_path = ""
                        if SAVE_BEST_CROPS:
                            crop_name = f"{VIDEO_PATH.stem}_track{int(track_id):05d}_best.jpg"
                            crop_path_obj = CROPS_DIR / crop_name
                            cv2.imwrite(str(crop_path_obj), crop)
                            crop_path = str(crop_path_obj)

                        state.best_obs = BestObservation(
                            frame_idx=frame_idx,
                            timestamp_utc=timestamp_utc,
                            video_time_sec=video_time_sec,
                            class_name=class_name,
                            class_conf=float(conf),
                            color_name=color_name,
                            color_conf=float(color_conf),
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            crop_path=crop_path,
                            score=float(best_score),
                        )

            # finalize tracks not seen recently
            to_finalize = []
            for tid, state in active_tracks.items():
                if tid not in seen_track_ids_this_frame:
                    if frame_idx - state.last_frame_idx >= MAX_MISSED_FRAMES:
                        to_finalize.append(tid)

            for tid in to_finalize:
                finalize_track(writer, active_tracks[tid])
                del active_tracks[tid]

            frame_idx += 1

        # finalize remaining tracks at EOF
        for tid in sorted(active_tracks):
            finalize_track(writer, active_tracks[tid])

    cap.release()
    print(f"Done. Wrote tracked vehicle events to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
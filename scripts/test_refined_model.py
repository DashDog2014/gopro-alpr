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
VIDEO_OG_PATH = Path("data/raw/test-run/GH340092.MP4")
VIDEO_PATH = Path("data/remuxed/GH340092_clean.MP4")
MODEL_PATH = Path("runs/detect/runs/model_compare/finetuned/weights/best.pt")
OUTPUT_CSV = Path("data/output/tracked_vehicle_events.csv")

SAVE_BEST_CROPS = True
CROPS_DIR = Path("data/output/best_crops")

SAVE_DEBUG_VIDEO = True
DEBUG_VIDEO_PATH = Path("data/output/tracked_debug.mp4")
DEBUG_VIDEO_FPS = None   # use source FPS if None

DEVICE = 0
CONF_THRESH = 0.4
IMG_SIZE = 960
TRACKER_CFG = "bytetrack.yaml"
SAMPLE_EVERY_N_FRAMES = 1
MAX_MISSED_FRAMES = 15

BOX_THICKNESS = 2
FONT_SCALE = 0.7
TEXT_THICKNESS = 2

UTC = timezone.utc
#tracking things
MIN_FRAMES_SEEN_TO_KEEP = 2 #@2 it saw 2563 vehicles
DRAW_ONLY_CONFIRMED_TRACKS = True
MERGE_FRAGMENTED_TRACKS = True
MAX_REID_GAP_FRAMES = 45
MAX_REID_CENTER_X_DIST = 180
MAX_REID_CENTER_Y_DIST = 120
REQUIRE_SAME_CLASS_FOR_REID = True

# ----------------------------
# Metadata helpers
# ----------------------------
def get_video_creation_time_utc(video_og_path: Path) -> Optional[datetime]:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_entries", "format_tags=creation_time",
            str(video_og_path),
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
    first_center_x: float = 0.0
    last_center_x: float = 0.0
    frames_seen: int = 0
    best_obs: Optional[BestObservation] = None
    class_votes: dict[str, float] = field(default_factory=dict)
    color_votes: dict[str, float] = field(default_factory=dict)


# ----------------------------
# Scoring / labels / drawing
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


def infer_travel_direction(track: TrackState, min_dx: float = 30.0) -> str:
    dx = track.last_center_x - track.first_center_x
    if dx >= min_dx:
        return "north"
    if dx <= -min_dx:
        return "south"
    return "unknown"


def get_track_color(track_id: int) -> tuple[int, int, int]:
    r = (37 * track_id) % 255
    g = (97 * track_id) % 255
    b = (157 * track_id) % 255
    return int(b), int(g), int(r)  # BGR


def draw_box_label(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
    box_thickness: int = 2,
    font_scale: float = 0.7,
    text_thickness: int = 2,
):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )

    text_y1 = max(0, y1 - th - baseline - 6)
    text_y2 = text_y1 + th + baseline + 6
    text_x2 = min(frame.shape[1], x1 + tw + 8)

    cv2.rectangle(frame, (x1, text_y1), (text_x2, text_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, text_y2 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

# ---------------------------------------------------
# more helper functions for merging fragmented ID's
# ---------------------------------------------------
def best_center(track: TrackState) -> tuple[float, float]:
    best = track.best_obs
    if best is None:
        return 0.0, 0.0
    return (best.x1 + best.x2) / 2.0, (best.y1 + best.y2) / 2.0


def best_box_area(track: TrackState) -> float:
    best = track.best_obs
    if best is None:
        return 0.0
    return max(0, best.x2 - best.x1) * max(0, best.y2 - best.y1)


def track_main_class(track: TrackState) -> str:
    if track.class_votes:
        return choose_vote_label(track.class_votes)
    if track.best_obs is not None:
        return track.best_obs.class_name
    return ""


def track_main_color(track: TrackState) -> str:
    if track.color_votes:
        return choose_vote_label(track.color_votes)
    if track.best_obs is not None:
        return track.best_obs.color_name
    return ""


def should_merge_tracks(a: TrackState, b: TrackState) -> bool:
    """
    Returns True when b looks like a continuation/reappearance of a.

    This is designed for cases where a car is briefly obstructed and ByteTrack
    gives it a new ID when it reappears.
    """
    if a.best_obs is None or b.best_obs is None:
        return False

    # b must occur after a
    frame_gap = b.first_frame_idx - a.last_frame_idx
    if frame_gap < 0:
        return False

    if frame_gap > MAX_REID_GAP_FRAMES:
        return False

    class_a = track_main_class(a)
    class_b = track_main_class(b)
    if REQUIRE_SAME_CLASS_FOR_REID and class_a and class_b and class_a != class_b:
        return False

    ax, ay = best_center(a)
    bx, by = best_center(b)

    if abs(ax - bx) > MAX_REID_CENTER_X_DIST:
        return False

    if abs(ay - by) > MAX_REID_CENTER_Y_DIST:
        return False

    # Avoid merging vehicles moving in opposite directions when direction is known.
    dir_a = infer_travel_direction(a)
    dir_b = infer_travel_direction(b)
    if dir_a != "unknown" and dir_b != "unknown" and dir_a != dir_b:
        return False

    return True


def merge_two_tracks(a: TrackState, b: TrackState) -> TrackState:
    """
    Merge b into a. Keep a.track_id as the canonical ID.
    """
    a.first_frame_idx = min(a.first_frame_idx, b.first_frame_idx)
    a.last_frame_idx = max(a.last_frame_idx, b.last_frame_idx)

    if b.first_seen_video_sec < a.first_seen_video_sec:
        a.first_seen_video_sec = b.first_seen_video_sec
        a.first_seen_time_utc = b.first_seen_time_utc
        a.first_center_x = b.first_center_x

    if b.last_seen_video_sec > a.last_seen_video_sec:
        a.last_seen_video_sec = b.last_seen_video_sec
        a.last_seen_time_utc = b.last_seen_time_utc
        a.last_center_x = b.last_center_x

    a.frames_seen += b.frames_seen

    for label, score in b.class_votes.items():
        a.class_votes[label] = a.class_votes.get(label, 0.0) + score

    for label, score in b.color_votes.items():
        a.color_votes[label] = a.color_votes.get(label, 0.0) + score

    if a.best_obs is None or (
        b.best_obs is not None and b.best_obs.score > a.best_obs.score
    ):
        a.best_obs = b.best_obs

    return a


def merge_fragmented_tracks(tracks: list[TrackState]) -> list[TrackState]:
    if not MERGE_FRAGMENTED_TRACKS:
        return tracks

    tracks = [
        t for t in tracks
        if t.best_obs is not None and t.frames_seen >= MIN_FRAMES_SEEN_TO_KEEP
    ]

    tracks.sort(key=lambda t: t.first_frame_idx)

    merged: list[TrackState] = []

    for track in tracks:
        merged_into_existing = False

        # Search recent merged tracks backwards first.
        for existing in reversed(merged):
            if should_merge_tracks(existing, track):
                merge_two_tracks(existing, track)
                merged_into_existing = True
                break

        if not merged_into_existing:
            merged.append(track)

    return merged

def keep_track(track: TrackState) -> bool:
    if track.best_obs is None:
        return False

    if track.frames_seen < MIN_FRAMES_SEEN_TO_KEEP:
        return False

    return True

# ----------------------------
# Finalization
# ----------------------------
def write_track_row(writer, track: TrackState):
    best = track.best_obs
    if best is None:
        return

    final_class = choose_vote_label(track.class_votes) or best.class_name
    final_color = choose_vote_label(track.color_votes) or best.color_name
    travel_direction = infer_travel_direction(track)

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
        travel_direction,
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
    if SAVE_DEBUG_VIDEO:
        DEBUG_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_source}")
    print(f"Model: {MODEL_PATH}")
    print(f"FPS: {fps:.3f}")
    print(f"Using tracker: {TRACKER_CFG}")
    if video_start_utc:
        print(f"creation_time UTC: {video_start_utc.isoformat()}")

    debug_writer = None
    if SAVE_DEBUG_VIDEO:
        out_fps = float(fps if DEBUG_VIDEO_FPS is None else DEBUG_VIDEO_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(
            str(DEBUG_VIDEO_PATH),
            fourcc,
            out_fps,
            (frame_w, frame_h),
        )
        if not debug_writer.isOpened():
            raise RuntimeError(f"Could not open debug video writer: {DEBUG_VIDEO_PATH}")

    active_tracks: dict[int, TrackState] = {}
    finished_tracks: list[TrackState] = []

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
            "travel_direction",
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

            debug_frame = frame.copy()

            if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
                if debug_writer is not None:
                    debug_writer.write(debug_frame)
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

                    track_id_int = int(track_id)
                    seen_track_ids_this_frame.add(track_id_int)
                    class_name = names[int(cls_id)]
                    crop = frame[y1:y2, x1:x2]
                    color_name, color_conf = estimate_vehicle_color(crop)
                    cx = (x1 + x2) / 2.0

                    if track_id_int not in active_tracks:
                        active_tracks[track_id_int] = TrackState(
                            track_id=track_id_int,
                            video_source=video_source,
                            first_frame_idx=frame_idx,
                            last_frame_idx=frame_idx,
                            first_seen_time_utc=timestamp_utc,
                            last_seen_time_utc=timestamp_utc,
                            first_seen_video_sec=video_time_sec,
                            last_seen_video_sec=video_time_sec,
                            first_center_x=cx,
                            last_center_x=cx,
                        )

                    state = active_tracks[track_id_int]
                    state.last_frame_idx = frame_idx
                    state.last_seen_time_utc = timestamp_utc
                    state.last_seen_video_sec = video_time_sec
                    state.last_center_x = cx
                    state.frames_seen += 1

                    state.class_votes[class_name] = state.class_votes.get(class_name, 0.0) + float(conf)
                    if color_name:
                        state.color_votes[color_name] = state.color_votes.get(color_name, 0.0) + float(color_conf)

                    best_score = compute_best_frame_score(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
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
                            crop_name = f"{VIDEO_PATH.stem}_track{track_id_int:05d}_best.jpg"
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

                    # Optionally hide tracks from the debug video until they survive long enough.
                    # This prevents one-frame obstruction glitches from showing up as extra IDs.
                    if not DRAW_ONLY_CONFIRMED_TRACKS or state.frames_seen >= MIN_FRAMES_SEEN_TO_KEEP:
                        direction = infer_travel_direction(state) if state.frames_seen > 1 else "unknown"
                        draw_color = get_track_color(track_id_int)
                        label = f"ID {track_id_int} | {class_name} | {float(conf):.2f} | {direction}"
                        draw_box_label(
                            debug_frame,
                            x1, y1, x2, y2,
                            label=label,
                            color=draw_color,
                            box_thickness=BOX_THICKNESS,
                            font_scale=FONT_SCALE,
                            text_thickness=TEXT_THICKNESS,
                        )

            to_finalize = []
            for tid, state in active_tracks.items():
                if tid not in seen_track_ids_this_frame:
                    if frame_idx - state.last_frame_idx >= MAX_MISSED_FRAMES:
                        to_finalize.append(tid)

            for tid in to_finalize:
                track = active_tracks[tid]
                if keep_track(track):
                    finished_tracks.append(track)
                del active_tracks[tid]

            if debug_writer is not None:
                debug_writer.write(debug_frame)

            frame_idx += 1

        for tid in sorted(active_tracks):
            track = active_tracks[tid]
            if keep_track(track):
                finished_tracks.append(track)

        final_tracks = merge_fragmented_tracks(finished_tracks)

        for track in final_tracks:
            write_track_row(writer, track)

    cap.release()
    if debug_writer is not None:
        debug_writer.release()
        print(f"Saved debug video to {DEBUG_VIDEO_PATH}")

    print(f"Done. Wrote tracked vehicle events to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
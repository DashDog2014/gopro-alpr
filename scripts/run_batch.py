import os
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "100000") #fix max read bug

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from zoneinfo import ZoneInfo

import _bootstrap  # noqa: F401
from src.db.events import VehicleEvent, insert_events_batch
from src.db.detections import Detection, insert_detections_batch
from src.vision.yolo_vehicle import VehicleDetector

import json
import subprocess

from datetime import timezone
UTC = timezone.utc

import uuid

@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    start_ts_utc: datetime


def guess_video_start_time_utc(video_path: Path) -> datetime:
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
            # e.g. 2026-02-17T14:09:17.000000Z
            return datetime.fromisoformat(creation.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        pass

    # fallback: filesystem time -> UTC (this can reflect copy time, so only fallback)
    ts = video_path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=UTC)


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        # fallback if container doesn't report fps reliably
        fps = 30.0

    start_ts_utc = guess_video_start_time_utc(video_path)
    print(f"Using start_ts_utc={start_ts_utc.isoformat()} for {video_path.name}")
    print("Start time (mtn):", start_ts_utc.isoformat())

    return VideoInfo(path=video_path, fps=fps, frame_count=frame_count, start_ts_utc=start_ts_utc)


def run_on_video(video_path: Path, sample_fps: float, batch_size: int, dry_run: bool, max_events: int | None, rotate: str) -> int:
    run_id = uuid.uuid4()
    detector = VehicleDetector(model_name="yolov8n.pt", conf=0.40)
    print("run_id:", run_id)
    info = get_video_info(video_path)
    print(f"\n=== {info.path.name} ===")
    print(f"fps={info.fps:.3f}, frames={info.frame_count}, start_ts_utc={info.start_ts_utc.isoformat()}")

    step_frames = max(1, int(round(info.fps / sample_fps)))
    print(f"Sampling ~{sample_fps} fps => every {step_frames} frames")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    events: list[VehicleEvent] = []
    dets: list[Detection] = []
    
    inserted_total = 0
    produced_total = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Rotate every frame (or at least every frame you might process)
        if rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == "180":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if frame_idx == 0:
            Path("data/frames").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"data/frames/debug_first_{info.path.stem}.jpg", frame)

        if frame_idx % step_frames == 0:
            offset_sec = frame_idx / info.fps
            ts_utc = info.start_ts_utc + timedelta(seconds=offset_sec)

            events.append(
                VehicleEvent(
                    run_id=run_id,
                    video_file=info.path.name,
                    frame_idx=frame_idx,
                    timestamp_mtn=ts_utc,
                    # CV fields left empty for now
                    vehicle_type=None,
                    vehicle_color=None,
                    color_conf=None,
                    plate_text=None,
                    plate_conf=None,
                )
            )
            boxes = detector.detect(frame)
            """print(f"frame {frame_idx}: boxes={len(boxes)}")
            if boxes:
                b0 = boxes[0]
                print(" first box:", b0.cls_name, b0.conf, b0.x1, b0.y1, b0.x2, b0.y2)"""
            if produced_total < 5:  # only first few sampled frames
                dbg = frame.copy()
                for b in boxes:
                    cv2.rectangle(dbg, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                    cv2.putText(
                        dbg,
                        f"{b.cls_name} {b.conf:.2f}",
                        (b.x1, max(20, b.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                Path("data/frames/boxed").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"data/frames/boxed/{info.path.stem}_f{frame_idx}.jpg", dbg)
            for b in boxes:
                Path("data/crops/vehicles").mkdir(parents=True, exist_ok=True)
                crop = frame[b.y1:b.y2, b.x1:b.x2]
                crop_path = f"data/crops/vehicles/{info.path.stem}_f{frame_idx}_{b.cls_name}_{int(b.conf*100)}.jpg"
                cv2.imwrite(crop_path, crop)
                dets.append(
                    Detection(
                        run_id=run_id,
                        video_file=info.path.name,
                        frame_idx=frame_idx,
                        ts_utc=ts_utc,
                        det_type="vehicle",
                        x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
                        vehicle_class=b.cls_name,
                        det_conf=b.conf,
                        model=detector.model_name,
                        crop_path=crop_path
                    )
                )
            produced_total += 1

            # Insert in chunks
            if len(events) >= batch_size:
                if dry_run:
                    print(f"[dry-run] would insert {len(events)} rows")
                    inserted_total += len(events)
                else:
                    n = insert_events_batch(events)
                    inserted_total += n
                    print(f"Inserted {n} rows (total {inserted_total})")
                events.clear()
            if len(dets) >= batch_size:
                if dry_run:
                    print(f"[dry-run] would insert {len(dets)} detections")
                else:
                    m = insert_detections_batch(dets)
                    print(f"Inserted {m} detections")
                dets.clear()

            if max_events is not None and produced_total >= max_events:
                print(f"Reached max_events={max_events}; stopping early.")
                break

        frame_idx += 1

    cap.release()

    # Flush remainder
    if events:
        if dry_run:
            print(f"[dry-run] would insert {len(events)} rows")
            inserted_total += len(events)
        else:
            n = insert_events_batch(events)
            inserted_total += n
            print(f"Inserted {n} rows (total {inserted_total})")
    if dets:
        if dry_run:
            print(f"[dry-run] would insert {len(dets)} detections")
        else:
            insert_detections_batch(dets)
            print(f"Inserted {len(dets)} detections (attempted)")

    print(f"Done. Produced ~{produced_total} events. Inserted {inserted_total}.")
    return inserted_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="data/raw", help="Folder containing MP4 files")
    ap.add_argument("--sample_fps", type=float, default=5.0, help="How many frames/sec to sample")
    ap.add_argument("--batch_size", type=int, default=200, help="Rows per DB insert batch")
    ap.add_argument("--dry_run", action="store_true", help="Don’t write to DB; just print what would happen")
    ap.add_argument("--max_events", type=int, default=None, help="Stop after producing this many events per video")
    ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none",
                help="Rotate frames after decode (for portrait GoPro)")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise SystemExit(f"Video dir not found: {video_dir.resolve()}")

    videos = sorted({*video_dir.glob("*.MP4"), *video_dir.glob("*.mp4")})
    if not videos:
        raise SystemExit(f"No MP4 files found in {video_dir.resolve()}")

    total = 0
    for vp in videos:
        total += run_on_video(vp, args.sample_fps, args.batch_size, args.dry_run, args.max_events, args.rotate)

    print(f"\nALL DONE. Inserted total rows: {total}")


if __name__ == "__main__":
    main()
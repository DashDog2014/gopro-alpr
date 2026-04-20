from pathlib import Path
import cv2

INPUT_DIR = Path("data/raw/test_run")
OUTPUT_DIR = Path("data/vehicle_dataset/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4"}

for video_path in sorted(INPUT_DIR.iterdir()):
    if video_path.suffix.lower() not in VIDEO_EXTS:
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Could not read FPS for {video_path}")
        cap.release()
        continue

    step = max(1, int(round(fps / 2)))   # 2 frames per second
    frame_idx = 0
    saved = 0
    stem = video_path.stem

    print(f"Processing {video_path.name} at {fps:.2f} fps, saving every {step} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            out_path = OUTPUT_DIR / f"{stem}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames from {video_path.name}")

print("Done.")
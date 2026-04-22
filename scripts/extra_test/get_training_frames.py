from pathlib import Path
import subprocess

INPUT_DIR = Path("data/raw/test-run")
OUTPUT_DIR = Path("data/vehicle_dataset/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for video_path in sorted(INPUT_DIR.glob("*.MP4")) + sorted(INPUT_DIR.glob("*.mp4")):
    stem = video_path.stem
    out_pattern = OUTPUT_DIR / f"{stem}_frame_%06d.jpg"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", "fps=0.2",
        str(out_pattern)
    ]

    print(f"Processing {video_path.name}")
    subprocess.run(cmd, check=True)
    print(f"Done {video_path.name}")

print("Done.")
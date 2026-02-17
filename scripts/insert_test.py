import sys
from pathlib import Path

# Add repo root to import path so "import src..." works
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.db.events import VehicleEvent, insert_event, latest_events

if __name__ == "__main__":
    new_id = insert_event(
        VehicleEvent(
            video_file="test.mp4",
            frame_idx=123,
            vehicle_type="car",
            vehicle_color="red",
            color_conf=0.9,
            plate_text="ABC123",
            plate_conf=0.8,
        )
    )
    print("Inserted:", new_id)
    print("Latest:", latest_events(5))
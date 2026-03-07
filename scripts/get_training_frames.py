import cv2
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

VIDEO_DIR = Path("data/remuxed")
OUTPUT_DIR = Path("data/vehicle_dataset")

images_dir = OUTPUT_DIR / "images"
labels_dir = OUTPUT_DIR / "labels"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# load model
model = YOLO("yolov8s.pt")

# vehicle classes in COCO
VEHICLE_CLASSES = [2, 3, 5, 7]

# tracker
tracker = sv.ByteTrack()

vehicle_memory = {}

for video in VIDEO_DIR.glob("*.mp4"):

    print(f"Processing {video}")

    cap = cv2.VideoCapture(str(video))
    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # sample every 15 frames (~2 fps for 30fps video)
        if frame_id % 15 != 0:
            continue

        results = model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)

        # keep only vehicles
        mask = [c in VEHICLE_CLASSES for c in detections.class_id]
        detections = detections[mask]

        detections = tracker.update_with_detections(detections)

        for box, track_id in zip(detections.xyxy, detections.tracker_id):

            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]

            area = (x2-x1) * (y2-y1)

            # keep the largest frame for each vehicle
            if track_id not in vehicle_memory or area > vehicle_memory[track_id]["area"]:

                vehicle_memory[track_id] = {
                    "image": crop,
                    "area": area
                }

    cap.release()

# save results
for track_id, data in vehicle_memory.items():

    img_path = images_dir / f"vehicle_{track_id}.jpg"
    cv2.imwrite(str(img_path), data["image"])

    # simple YOLO label (whole image)
    label_path = labels_dir / f"vehicle_{track_id}.txt"

    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 1.0 1.0\n")

print("Vehicle dataset created.")
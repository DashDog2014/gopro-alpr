from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ultralytics import YOLO


# COCO class ids for vehicles
COCO_VEHICLE_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_name: str
    conf: float


class VehicleDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25):
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[BBox]:
        """
        frame_bgr: OpenCV frame (BGR uint8).
        Returns pixel bboxes.
        """
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        r = results[0]

        out: list[BBox] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out

        # xyxy in pixels
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        h, w = frame_bgr.shape[:2]

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            if k not in COCO_VEHICLE_IDS:
                continue
            # clamp + int
            x1i = int(max(0, min(w - 1, round(x1))))
            y1i = int(max(0, min(h - 1, round(y1))))
            x2i = int(max(0, min(w - 1, round(x2))))
            y2i = int(max(0, min(h - 1, round(y2))))
            if x2i <= x1i or y2i <= y1i:
                continue

            out.append(
                BBox(
                    x1=x1i, y1=y1i, x2=x2i, y2=y2i,
                    cls_name=COCO_VEHICLE_IDS[k],
                    conf=float(c),
                )
            )

        return out
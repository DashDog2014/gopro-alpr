from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from psycopg2.extras import execute_values

from src.db.db import get_connect  # your existing helper


@dataclass
class Detection:
    run_id: UUID
    video_file: str
    frame_idx: int
    ts_utc: datetime

    det_type: str  # 'vehicle' or 'plate'
    x1: int
    y1: int
    x2: int
    y2: int

    vehicle_class: Optional[str] = None
    det_conf: Optional[float] = None

    plate_text: Optional[str] = None
    plate_conf: Optional[float] = None

    color_label: Optional[str] = None
    color_conf: Optional[float] = None

    model: Optional[str] = None
    crop_path: Optional[str] = None


def insert_detections_batch(dets: list[Detection]) -> int:
    if not dets:
        return 0

    rows = [
        (
            str(d.run_id),          # UUID -> str to avoid psycopg2 adaptation issues in execute_values
            d.video_file,
            d.frame_idx,
            d.ts_utc,
            d.det_type,
            d.x1, d.y1, d.x2, d.y2,
            d.vehicle_class,
            d.det_conf,
            d.plate_text,
            d.plate_conf,
            d.color_label,
            d.color_conf,
            d.model,
            d.crop_path,
        )
        for d in dets
    ]

    sql = """
    INSERT INTO detections
      (run_id, video_file, frame_idx, ts_utc, det_type, x1, y1, x2, y2,
       vehicle_class, det_conf, plate_text, plate_conf, color_label, color_conf, model, crop_path)
    VALUES %s
    ON CONFLICT DO NOTHING
    """

    conn = get_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, page_size=500)
        return len(dets)
    finally:
        conn.close()
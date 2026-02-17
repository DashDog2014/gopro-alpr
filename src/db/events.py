from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Any
from datetime import datetime

from psycopg2.extras import execute_values

from .db import get_connect


@dataclass
class VehicleEvent:
    video_file: str
    frame_idx: int
    timestamp_mtn: Optional[datetime] = None  # if None, DB default now()
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    color_conf: Optional[float] = None
    plate_text: Optional[str] = None
    plate_conf: Optional[float] = None
INSERT_SQL = """
INSERT INTO vehicle_events
  (video_file, frame_idx, timestamp_mtn, vehicle_type, vehicle_color, color_conf, plate_text, plate_conf)
VALUES
  (%s, %s, COALESCE(%s, DEFAULT), %s, %s, %s, %s, %s)
RETURNING id;
"""


def insert_event(e: VehicleEvent) -> int:
    """Insert one event and return the new row id."""
    conn = get_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO vehicle_events
                      (video_file, frame_idx, timestamp_mtn, vehicle_type, vehicle_color, color_conf, plate_text, plate_conf)
                    VALUES
                      (%s, %s, COALESCE(%s, now()), %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        e.video_file,
                        e.frame_idx,
                        e.timestamp_mtn,
                        e.vehicle_type,
                        e.vehicle_color,
                        e.color_conf,
                        e.plate_text,
                        e.plate_conf,
                    ),
                )
                return int(cur.fetchone()[0])
    finally:
        conn.close()


def insert_events_batch(events: Sequence[VehicleEvent]) -> int:
    """Insert many events efficiently. Returns number of rows inserted."""
    if not events:
        return 0

    rows: list[tuple[Any, ...]] = [
        (
            e.video_file,
            e.frame_idx,
            e.timestamp_mtn,
            e.vehicle_type,
            e.vehicle_color,
            e.color_conf,
            e.plate_text,
            e.plate_conf,
        )
        for e in events
    ]

    conn = get_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO vehicle_events
                      (video_file, frame_idx, timestamp_mtn, vehicle_type, vehicle_color, color_conf, plate_text, plate_conf)
                    VALUES %s
                    """,
                    rows,
                )
        return len(rows)
    finally:
        conn.close()


def latest_events(limit: int = 20) -> list[tuple]:
    """Fetch latest events (for quick debugging)."""
    conn = get_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, video_file, frame_idx, timestamp_mtn, vehicle_type, vehicle_color, plate_text
                FROM vehicle_events
                ORDER BY id DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()
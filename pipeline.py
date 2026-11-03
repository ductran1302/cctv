import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WRIST_IDS = (9, 10)
ACTION_LABELS = {
    "VAO KHU VUC": "Vào khu vực",
    "RA KHAY": "Ra khay",
    "BAM MAY TINH BANG": "Bấm máy tính bảng",
    "KHONG THAO TAC": "Không thao tác",
}


def open_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_zones(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def zones_to_pixels(zones, width, height):
    out = {}
    for name, pts in zones.items():
        if len(pts) >= 3:
            out[name] = np.array(
                [[float(x) * width, float(y) * height] for x, y in pts],
                dtype=np.float32,
            )
    return out


def inside(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def extract_tracks(result):
    tracks = []
    if result.keypoints is None or result.boxes is None:
        return tracks
    xy = result.keypoints.xy.cpu().numpy()
    if xy.size == 0:
        return tracks
    conf = result.keypoints.conf
    conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id
    ids = ids.cpu().numpy() if ids is not None else [None] * len(boxes)
    for i in range(min(len(boxes), len(xy))):
        kpts = [(xy[i][j][0], xy[i][j][1], conf[i][j]) for j in range(len(xy[i]))]
        track_id = int(ids[i]) if ids[i] is not None else None
        tracks.append({"id": track_id, "box": boxes[i], "kpts": kpts})
    return tracks


class StationLogic:
    def __init__(self, tap_debounce=1.5, idle_seconds=120):
        self.tap_debounce = tap_debounce
        self.idle_seconds = idle_seconds
        self.last_tap = 0.0
        self.tap_count = 0
        self.present = False
        self.idle_warned = False
        self.seq = 0

    def next_id(self):
        self.seq += 1
        return self.seq

    def make(self, action, cycle, track_id=None):
        return {
            "id": self.next_id(),
            "ts": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "label": ACTION_LABELS.get(action, action),
            "track_id": track_id,
            "cycle_sec": cycle,
            "level": "Tự động ghi nhận",
        }

    def update(self, now, tracks, zones):
        events = []
        station = zones.get("station")
        tablet = zones.get("tablet")

        if station is not None:
            present = any(
                inside(((t["box"][0] + t["box"][2]) / 2, t["box"][3]), station)
                for t in tracks
            )
        else:
            present = bool(tracks)

        if present and not self.present:
            events.append(self.make("VAO KHU VUC", None))
            self.idle_warned = False
        if not present and self.present:
            events.append(self.make("RA KHAY", None))
            self.tap_count = 0
            self.idle_warned = False
        self.present = present

        if tablet is not None and present:
            hit = False
            hit_id = None
            for t in tracks:
                for idx in WRIST_IDS:
                    if idx < len(t["kpts"]):
                        x, y, c = t["kpts"][idx]
                        if c >= 0.3 and inside((x, y), tablet):
                            hit = True
                            hit_id = t["id"]
                            break
                if hit:
                    break
            if hit and now - self.last_tap > self.tap_debounce:
                cycle = None if self.last_tap == 0 else round(now - self.last_tap, 2)
                self.last_tap = now
                self.tap_count += 1
                events.append(self.make("BAM MAY TINH BANG", cycle, hit_id))
                self.idle_warned = False

        if (
            present
            and self.last_tap
            and now - self.last_tap > self.idle_seconds
            and not self.idle_warned
        ):
            self.idle_warned = True
            events.append(
                self.make("KHONG THAO TAC", round(now - self.last_tap, 1))
            )

        return events


def draw_zones(frame, zones, tap_count):
    colors = {
        "station": (255, 255, 0),
        "tablet": (255, 200, 0),
        "tray": (0, 165, 255),
    }
    for name, poly in zones.items():
        color = colors.get(name, (200, 200, 200))
        cv2.polylines(frame, [poly.astype(np.int32)], True, color, 2)
        label = name.upper()
        if name == "tray":
            label = f"KHAY THANH PHAM: {tap_count} SP"
        x = int(poly[0][0])
        y = max(18, int(poly[0][1]) - 8)
        cv2.putText(
            frame,
            label,
            (max(0, x), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


class Pipeline:
    def __init__(self, source, model_path, zones_path, conf=0.35, device=None):
        self.source = source
        self.model_path = model_path
        self.zones = load_zones(zones_path)
        self.conf = conf
        self.device = device
        self.logic = StationLogic()
        self.on_event = None
        self.history = []
        self._history_lock = threading.Lock()
        self._lock = threading.Lock()
        self._jpeg = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def history_events(self):
        with self._history_lock:
            return list(self.history)

    def _emit(self, event):
        with self._history_lock:
            self.history.append(event)
            if len(self.history) > 500:
                del self.history[:-500]
        if self.on_event:
            self.on_event(event)

    def _run(self):
        model = YOLO(self.model_path)
        cap = open_capture(self.source)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.release()
                time.sleep(1.0)
                cap = open_capture(self.source)
                continue

            results = model.track(
                frame,
                persist=True,
                conf=self.conf,
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml",
            )
            result = results[0]
            plotted = result.plot()
            height, width = frame.shape[:2]
            zones_px = zones_to_pixels(self.zones, width, height)

            for event in self.logic.update(
                time.time(), extract_tracks(result), zones_px
            ):
                self._emit(event)

            draw_zones(plotted, zones_px, self.logic.tap_count)

            ok2, buf = cv2.imencode(
                ".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ok2:
                with self._lock:
                    self._jpeg = buf.tobytes()

        cap.release()

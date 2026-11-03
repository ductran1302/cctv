import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def camera_source():
    value = os.getenv("CAMERA_SOURCE", "0")
    return int(value) if value.isdigit() else value


MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "yolov8n-pose.pt"))
ZONES_PATH = os.getenv("ZONES_PATH", str(BASE_DIR / "zones.json"))
CONF = float(os.getenv("CONF", "0.35"))
DEVICE = os.getenv("DEVICE") or None
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

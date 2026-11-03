import json
from pathlib import Path

import cv2
import numpy as np

import config

ZONE_NAMES = ["station", "tablet", "tray"]
COLORS = [(255, 255, 0), (255, 200, 0), (0, 165, 255)]
points = {name: [] for name in ZONE_NAMES}
state = {"index": 0, "frame": None, "scale": 1.0}


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    fx = int(x / state["scale"])
    fy = int(y / state["scale"])
    points[ZONE_NAMES[state["index"]]].append([fx, fy])


def render():
    frame = state["frame"].copy()
    scale = state["scale"]
    disp = cv2.resize(frame, None, fx=scale, fy=scale)
    for i, name in enumerate(ZONE_NAMES):
        pts = points[name]
        for p in pts:
            cv2.circle(
                disp, (int(p[0] * scale), int(p[1] * scale)), 4, COLORS[i], -1
            )
        if len(pts) >= 2:
            arr = (np.array(pts, dtype=np.float32) * scale).astype(np.int32)
            cv2.polylines(disp, [arr], len(pts) >= 3, COLORS[i], 2)
        cv2.putText(
            disp,
            f"[{i + 1}] {name}: {len(pts)} diem",
            (10, 25 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS[i],
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("calibrate", disp)


def grab_frame():
    cap = cv2.VideoCapture(config.camera_source())
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    frame = grab_frame()
    if frame is None:
        raise SystemExit("Khong doc duoc nguon camera")
    state["frame"] = frame
    height, width = frame.shape[:2]
    state["scale"] = min(1.0, 1200 / width)

    cv2.namedWindow("calibrate", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("calibrate", on_mouse)
    print(
        "1/2/3 chon vung | chuot trai them diem | r xoa vung | "
        "c chup lai | s luu | q thoat"
    )

    while True:
        render()
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("1"), ord("2"), ord("3")):
            state["index"] = key - ord("1")
        if key == ord("r"):
            points[ZONE_NAMES[state["index"]]].clear()
        if key == ord("c"):
            fresh = grab_frame()
            if fresh is not None:
                state["frame"] = fresh
        if key == ord("s"):
            out = {}
            for name in ZONE_NAMES:
                pts = points[name]
                if len(pts) >= 3:
                    out[name] = [
                        [round(x / width, 5), round(y / height, 5)] for x, y in pts
                    ]
            Path(config.ZONES_PATH).write_text(
                json.dumps(out, indent=2), encoding="utf-8"
            )
            print("Da luu", config.ZONES_PATH)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

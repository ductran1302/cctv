import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import Pipeline

BASE = Path(__file__).parent
pipeline = Pipeline(
    config.camera_source(),
    config.MODEL_PATH,
    config.ZONES_PATH,
    config.CONF,
    config.DEVICE,
)


class Hub:
    def __init__(self):
        self.loop = None
        self.queues = set()

    def set_loop(self, loop):
        self.loop = loop

    def register(self):
        q = asyncio.Queue()
        self.queues.add(q)
        return q

    def unregister(self, q):
        self.queues.discard(q)

    def publish(self, event):
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._put, event)

    def _put(self, event):
        for q in list(self.queues):
            q.put_nowait(event)


hub = Hub()
pipeline.on_event = hub.publish


@asynccontextmanager
async def lifespan(app):
    hub.set_loop(asyncio.get_running_loop())
    pipeline.start()
    yield
    pipeline.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/events")
async def events():
    return pipeline.history_events()


@app.get("/video")
async def video():
    async def stream():
        while True:
            frame = pipeline.latest_jpeg()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(
        stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    q = hub.register()
    for event in pipeline.history_events()[-50:]:
        await ws.send_json(event)
    try:
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(q)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)

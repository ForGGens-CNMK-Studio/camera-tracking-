import asyncio
from contextlib import asynccontextmanager
import json
import logging
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from human_demo_py.alerting import AlertingGate, AlertingGateDebug
from human_demo_py.config import MODEL_NAME, DEBUG
from human_demo_py.depthai_utils import DepthAI, DepthAIDebug
from human_demo_py.height import HeightGuardian, HeightGuardianDebug
from human_demo_py.utils.global_settings import settings
from threading import Lock

log = logging.getLogger(__name__)
streaming = True
streaming_lock = Lock()
heights_lock = Lock()
frame = None
heights = []
heights_json: bytes = b"[]"


class Main:
    depthai_class = DepthAI
    height_guardian_class = HeightGuardian  # Use the new height detection class
    alerting_gate_class = AlertingGate

    def __init__(self):
        self.depthai = self.depthai_class(MODEL_NAME)
        self.height_guardian = self.height_guardian_class()
        self.alerting_gate = self.alerting_gate_class()

    def parse_frame(self, frame, bboxes, pose_detections):
        # global heights
        height_results = self.height_guardian.parse_frame(frame, bboxes)
        # You could add additional alerting logic based on height if needed
        for result in height_results:
            height = result["height"]
            log.info(f"Detected person's height: {height:.2f} meters")

        # with heights_lock:  # Ensure thread-safe access
        #     heights = height_results
        #     log.debug("Heights: %s", heights)

        return (
            height_results,
            False,
        )  # No alerting for height, unless you want to add height-based alerts

    def run(self):
        global heights, frame
        try:
            log.info("Setup complete, parsing frames...")
            for framed, bboxes, pose_detections in self.depthai.capture():
                heights = self.parse_frame(frame, bboxes, pose_detections)[0]
                frame = framed
                # if framed is not None:
                #     log.debug("Frame shape: %s", framed.shape)
        finally:
            del self.depthai


class MainDebug(Main):
    depthai_class = DepthAIDebug
    height_guardian_class = HeightGuardianDebug
    alerting_gate_class = AlertingGateDebug

    def __init__(self):
        super().__init__()

    def parse_frame(self, frame, bboxes, pose_detections):
        global heights
        height_results, should_alert = super().parse_frame(
            frame, bboxes, pose_detections
        )

        log.debug("Height results: %s", height_results)
        log.debug("Global Height results: %s", heights)

        if frame is not None:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)

            if key == ord("q"):
                raise StopIteration()

        return height_results, should_alert


if __name__ == "__main__":
    if DEBUG:
        log.info("Setting up debug run...")
        MainDebug().run()
    else:
        log.info("Setting up non-debug run...")
        Main().run()


def generate_frames():
    global frame, streaming
    while True:
        with streaming_lock:
            if not streaming:
                continue
        # Encode frame as JPEG
        if frame is None:
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        frameb = buffer.tobytes()
        # log.debug("Frame size: %d", len(frameb))

        # Yield frame as a byte stream
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frameb + b"\r\n")

        # sleep to send at most 24 frames per second
        time.sleep(1 / 24)


async def generate_heights():
    global heights_json
    while True:
        # Use asyncio sleep for non-blocking delay
        await asyncio.sleep(1 / 24)

        # Lock can be moved inside where it's strictly necessary
        with streaming_lock:
            if not streaming:
                continue

        log.debug("Heights to send: %s", heights)
        # with heights_lock:
        #     # JSON conversion and yield (convert to bytes here)
        #     heights_json = json.dumps(heights).encode("utf-8")
        #     log.debug("Heights: %s", heights_json)
        yield (
            b"--height\r\n"
            b"Content-Type: application/json\r\n\r\n" + heights_json + b"\r\n"
        )


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    camera_thread = threading.Thread(target=MainDebug().run)
    try:
        # Start the app in debug mode
        camera_thread.start()
        yield
    finally:
        # Stop the app
        camera_thread.join(5)


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
    root_path=settings.ROOT_PATH,
)

if settings.BACKEND_CORS_ORIGINS:
    log.info("CORS origins: %s", settings.BACKEND_CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"Hello": "World"}


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/height_feed")
async def height_feed():
    return StreamingResponse(
        generate_heights(), media_type="multipart/x-mixed-replace; boundary=height"
    )


@app.post("/control")
async def control_stream(action: str):
    global streaming
    with streaming_lock:
        if action == "pause":
            streaming = False
        elif action == "play":
            streaming = True
    return JSONResponse(content={"status": "success", "action": action})

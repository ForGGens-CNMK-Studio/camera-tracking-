import logging
import uuid
from pathlib import Path

from depthai_sdk.managers import (
    PipelineManager,
    NNetManager,
    BlobManager,
    PreviewManager,
)
from depthai_sdk import Previews, getDeviceInfo
import blobconverter
import cv2
import depthai as dai
from imutils.video import FPS
import numpy as np

log = logging.getLogger(__name__)


class DepthAI:
    def create_pipeline(self, model_name):
        log.info("Creating DepthAI pipeline...")

        pipeline = dai.Pipeline()
        # pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(
            dai.node.MobileNetSpatialDetectionNetwork
        )
        spatialDetectionNetwork.input.setBlocking(False)

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        camRgb.preview.link(xoutRgb.input)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutPose = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutPose.setStreamName("pose")

        # Set up pose estimation model
        poseEstimation = pipeline.create(dai.node.NeuralNetwork)
        poseEstimation.setBlobPath(
            blobconverter.from_zoo(name="human-pose-estimation-0001", shaves=6)
        )

        # Properties
        camRgb.setPreviewSize(544, 320)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs
        stereo.initialConfig.setConfidenceThreshold(255)
        stereo.initialConfig.setMedianFilter(
            dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
        )
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        spatialDetectionNetwork.setBlobPath(
            blobconverter.from_zoo(name=model_name, shaves=6)
        )
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Linking pose estimation
        camRgb.preview.link(poseEstimation.input)
        poseEstimation.out.link(xoutPose.input)

        camRgb.preview.link(spatialDetectionNetwork.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        log.info("Pipeline created.")
        return pipeline

    def __init__(self, model_name):
        self.pipeline = self.create_pipeline(model_name)
        self.detections = []
        self.poses = []

    def capture(self):
        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = (
                dai.CameraBoardSocket.LEFT in cams
                and dai.CameraBoardSocket.RIGHT in cams
            )
            if not depth_enabled:
                raise RuntimeError(
                    "Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(
                        cams
                    )
                )
            device.startPipeline(self.pipeline)
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(
                name="detections", maxSize=4, blocking=False
            )
            poseNNQueue = device.getOutputQueue(name="pose", maxSize=4, blocking=False)

            while True:
                frame = previewQueue.get().getCvFrame()
                inDet = detectionNNQueue.tryGet()
                inPose = poseNNQueue.tryGet()

                if inDet is not None:
                    self.detections = inDet.detections

                # Parse pose data
                if inPose is not None:
                    # log.debug(f"Pose data attributes: {dir(inPose)}")
                    # self.poses = inPose.getFirstLayerFp16()
                    # log.debug(f"Got pose data {self.poses[0]}")
                    # # Process pose keypoints here
                    heatmaps = np.array(
                        inPose.getLayerFp16("Mconv7_stage2_L2")
                    ).reshape((1, 19, 32, 57))
                    pafs = np.array(inPose.getLayerFp16("Mconv7_stage2_L1")).reshape(
                        (1, 38, 32, 57)
                    )
                    heatmaps = heatmaps.astype("float32")
                    pafs = pafs.astype("float32")
                    # log.debug(f"Got pose data 1: {heatmaps=}, {pafs=}")
                    self.poses = np.concatenate((heatmaps, pafs), axis=1)
                    # log.debug(f"Got pose data 2: {outputs=}")

                bboxes = []
                height = frame.shape[0]
                width = frame.shape[1]
                for detection in self.detections:
                    bboxes.append(
                        {
                            "id": uuid.uuid4(),
                            "label": detection.label,
                            "confidence": detection.confidence,
                            "x_min": int(detection.xmin * width),
                            "x_max": int(detection.xmax * width),
                            "y_min": int(detection.ymin * height),
                            "y_max": int(detection.ymax * height),
                            "depth_x": detection.spatialCoordinates.x / 1000,
                            "depth_y": detection.spatialCoordinates.y / 1000,
                            "depth_z": detection.spatialCoordinates.z / 1000,
                        }
                    )

                yield frame, bboxes, self.poses

    def __del__(self):
        del self.pipeline


class DepthAIDebug(DepthAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = FPS()
        self.fps.start()

    def extract_keypoints(self, pose_detections):
        """
        This function processes the output of the pose detection network
        to extract human keypoints (like head, shoulders, etc.)

        :param pose_detections: Array of pose detection output from the neural network.
                                Format will vary depending on the model.
        :return: List of (x, y) coordinates for each keypoint.
        """
        # Assuming pose_detections is a flat array of keypoints, reshape to 2D
        # For example, if there are 18 keypoints and each keypoint has an (x, y) coordinate
        keypoints = np.array(pose_detections).reshape(-1, 2)  # Reshape as needed

        # Optionally normalize keypoints to fit the image dimensions
        # You may need to multiply keypoints by image width/height if they are normalized

        return keypoints

    def draw_skeleton(self, frame, keypoints):
        for i, (x, y) in enumerate(keypoints):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoints
        # Optionally connect the keypoints with lines to form the skeleton

    def capture(self):
        for frame, detections, pose_detections in super().capture():
            self.fps.update()
            # Example of keypoints being used to draw skeleton
            if len(pose_detections):
                keypoints = self.extract_keypoints(pose_detections)
                self.draw_skeleton(frame, keypoints)

            for detection in detections:
                cv2.rectangle(
                    frame,
                    (detection["x_min"], detection["y_min"]),
                    (detection["x_max"], detection["y_max"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "x: {}".format(round(detection["depth_x"], 1)),
                    (detection["x_min"], detection["y_min"] + 30),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    "y: {}".format(round(detection["depth_y"], 1)),
                    (detection["x_min"], detection["y_min"] + 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    "z: {}".format(round(detection["depth_z"], 1)),
                    (detection["x_min"], detection["y_min"] + 70),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    "conf: {}".format(round(detection["confidence"], 1)),
                    (detection["x_min"], detection["y_min"] + 90),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    "label: {}".format(detection["label"], 1),
                    (detection["x_min"], detection["y_min"] + 110),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
            yield frame, detections, pose_detections

    def __del__(self):
        super().__del__()
        self.fps.stop()
        log.info("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        log.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

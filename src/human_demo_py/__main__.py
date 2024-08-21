import depthai as dai
import numpy as np
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutDepth = pipeline.createXLinkOut()
xoutRgb = pipeline.createXLinkOut()

xoutDepth.setStreamName("depth")
xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Stereo depth
stereo.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inDepth = depthQueue.get()
        inRgb = rgbQueue.get()

        depthFrame = inDepth.getFrame()
        colorFrame = inRgb.getCvFrame()

        # Calculate height
        # Assuming the person is in the middle of the frame
        height_pixel = depthFrame.shape[0]

        # Convert pixel to real-world distance
        # You might need to calibrate this conversion factor based on the camera's position
        # and the specific setup where the camera is being used.
        # Generally, this would involve some calibration against known height.
        height_meters = np.mean(depthFrame[:, depthFrame.shape[1] // 2]) / 1000

        print(f"Estimated height: {height_meters:.2f} meters")

        # Display the color frame for debugging
        cv2.imshow("Color", colorFrame)

        if cv2.waitKey(1) == ord("q"):
            break

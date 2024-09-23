import itertools
import logging
import math
import cv2

log = logging.getLogger(__name__)


class HeightGuardian:
    def parse_frame(self, frame, detections):
        results = []
        for detection in detections:
            # Calculate height using the y_min and y_max of the bounding box
            height = (
                detection["depth_z"]
                * (detection["y_max"] - detection["y_min"])
                / frame.shape[0]
            )  # Normalize to the frame height
            log.info(f"Person's height: {height} meters")

            results.append(
                {
                    "height": height,
                    "detection": detection,
                }
            )

        return results


class HeightGuardianDebug(HeightGuardian):
    def parse_frame(self, frame, boxes):
        results = super().parse_frame(frame, boxes)
        if frame is None:
            return results
        overlay = frame.copy()
        for result in results:
            x1 = result["detection"]["x_min"]
            y1 = result["detection"]["y_min"]
            x2 = result["detection"]["x_max"]
            y2 = result["detection"]["y_max"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"{result['height']:.2f}m",
                (x1, y1),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        return results

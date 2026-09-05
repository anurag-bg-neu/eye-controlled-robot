"""Isolates the iris inside an eye frame and estimates the pupil centre.

Derived from the GazeTracking library (MIT).
"""

import cv2
import numpy as np


class PupilDetection:
    """Binarises an eye frame and takes the iris centroid as the pupil."""

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detectiris_location(eye_frame)

    @staticmethod
    def imageprocessing_operations(frame_eye, threshold):
        """Blur, erode and binarise the eye frame so only the iris remains."""
        kernelss = np.ones((3, 3), np.uint8)
        newframe = cv2.bilateralFilter(frame_eye, 10, 15, 15)
        newframe = cv2.erode(newframe, kernelss, iterations=3)
        newframe = cv2.threshold(newframe, threshold, 255, cv2.THRESH_BINARY)[1]
        return newframe

    def detectiris_location(self, eye_frame):
        """Set self.x and self.y to the iris centroid, or leave them None."""
        self.iris_frame = self.imageprocessing_operations(eye_frame, self.threshold)

        # Slicing the last two return values keeps this working on OpenCV 3 and 4.
        contours, _ = cv2.findContours(
            self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            centre_moments = cv2.moments(contours[-2])
            self.x = int(centre_moments["m10"] / centre_moments["m00"])
            self.y = int(centre_moments["m01"] / centre_moments["m00"])
        except (IndexError, ZeroDivisionError):
            pass

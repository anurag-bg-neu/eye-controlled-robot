"""Isolates a single eye from a frame and measures its blink ratio.

Derived from the GazeTracking library (MIT).
"""

import math

import cv2
import numpy as np

from .pupil_location_detection import PupilDetection


class Isolateeye:
    """Crops one eye out of a frame and locates the pupil inside it."""

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, originalsframe, landmark, sidess, thresholds):
        self.framess = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.blinking = None

        self._analyzing(originalsframe, landmark, sidess, thresholds)

    @staticmethod
    def _findmiddlepoint(point1, point2):
        """Midpoint between two landmark points."""
        x_coord = int((point1.x + point2.x) / 2)
        y_coord = int((point1.y + point2.y) / 2)
        return (x_coord, y_coord)

    def _toisolateframe(self, framess, landmark, pointsss):
        """Mask out everything but the eye region and crop to its bounding box."""
        regioneye = np.array(
            [(landmark.part(point).x, landmark.part(point).y) for point in pointsss]
        )
        regioneye = regioneye.astype(np.int32)

        height, width = framess.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [regioneye], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, framess.copy(), mask=mask)

        margin = 5
        min_x = np.min(regioneye[:, 0]) - margin
        max_x = np.max(regioneye[:, 0]) + margin
        min_y = np.min(regioneye[:, 1]) - margin
        max_y = np.max(regioneye[:, 1]) + margin

        self.framess = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)
        height, width = self.framess.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinkingratio(self, landmark, pointsss):
        """Eye width divided by eye height. Larger means more closed."""
        left_landmark = (landmark.part(pointsss[0]).x, landmark.part(pointsss[0]).y)
        right_landmark = (landmark.part(pointsss[3]).x, landmark.part(pointsss[3]).y)
        top_landmark = self._findmiddlepoint(
            landmark.part(pointsss[1]), landmark.part(pointsss[2])
        )
        bottom_landmark = self._findmiddlepoint(
            landmark.part(pointsss[5]), landmark.part(pointsss[4])
        )

        eye_width = math.hypot(
            left_landmark[0] - right_landmark[0], left_landmark[1] - right_landmark[1]
        )
        eye_height = math.hypot(
            top_landmark[0] - bottom_landmark[0], top_landmark[1] - bottom_landmark[1]
        )

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyzing(self, originalsframe, landmark, sidess, thresholdcheck):
        """Isolate the requested eye and run pupil detection on it."""
        if sidess == 0:
            pointsss = self.LEFT_EYE_POINTS
        elif sidess == 1:
            pointsss = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinkingratio(landmark, pointsss)
        self._toisolateframe(originalsframe, landmark, pointsss)

        if not thresholdcheck.iscomplete():
            thresholdcheck.evaluate(self.framess, sidess)

        threshold = thresholdcheck.threshold(sidess)
        self.pupil = PupilDetection(self.framess, threshold)

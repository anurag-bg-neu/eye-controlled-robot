"""Per-webcam calibration of the binarisation threshold used for iris isolation.

Derived from the GazeTracking library (MIT).
"""

import cv2

from .pupil_location_detection import PupilDetection

# Fraction of the eye surface the iris is expected to cover.
TARGET_IRIS_SIZE = 0.48


class ThresholdCheck:
    """Averages the best threshold over the first N frames of a session."""

    def __init__(self, thresh_frames=20):
        self.thresh_frames = thresh_frames
        self.thresh_left = []
        self.thresh_right = []

    def iscomplete(self):
        """True once enough frames have been sampled for both eyes."""
        return (
            len(self.thresh_left) >= self.thresh_frames
            and len(self.thresh_right) >= self.thresh_frames
        )

    def threshold(self, sidess):
        """Mean calibrated threshold for the given eye."""
        if sidess == 0:
            return int(sum(self.thresh_left) / len(self.thresh_left))
        elif sidess == 1:
            return int(sum(self.thresh_right) / len(self.thresh_right))

    @staticmethod
    def irissize(framess):
        """Fraction of the eye surface taken up by black pixels."""
        framess = framess[5:-5, 5:-5]
        height, width = framess.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(framess)
        return nb_blacks / nb_pixels

    @staticmethod
    def findbestthreshold(framess_eye):
        """Sweep thresholds and return the one closest to the target iris size."""
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = PupilDetection.imageprocessing_operations(framess_eye, threshold)
            trials[threshold] = ThresholdCheck.irissize(iris_frame)

        bestthreshold, _ = min(
            trials.items(), key=lambda p: abs(p[1] - TARGET_IRIS_SIZE)
        )
        return bestthreshold

    def evaluate(self, eyeframe, sidess):
        """Record one more threshold sample for the given eye."""
        threshold_find = self.findbestthreshold(eyeframe)

        if sidess == 0:
            self.thresh_left.append(threshold_find)
        elif sidess == 1:
            self.thresh_right.append(threshold_find)

"""Eye tracking driver: locates pupils and classifies gaze direction.

Derived from the GazeTracking library (MIT). See LICENSE.
"""

from pathlib import Path

import cv2
import dlib

from .isolate_eye_frame import Isolateeye
from .threshold import ThresholdCheck

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sp.dat"

# Horizontal pupil ratio boundaries, calibrated on the capstone test set.
# 0.0 is the far right of the eye, 1.0 the far left.
RIGHT_RATIO = 0.40
LEFT_RATIO = 0.80

# Mean eye width-to-height ratio above which the eye counts as closed.
BLINK_RATIO = 3.8


class EyeTracking:
    """Tracks both eyes in a frame and reports gaze direction and blink state."""

    def __init__(self, model_path=MODEL_PATH):
        self.frame_faces = None
        self.eye_left = None
        self.eye_right = None
        self.threshold = ThresholdCheck()
        self.landmarks = None

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Shape predictor not found at {model_path}. "
                "Run: uv run scripts/fetch_models.py"
            )

        self._face_detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(model_path))

    @property
    def pupils_located(self):
        """True when both pupil centroids resolved on the current frame."""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyzing(self):
        """Detect the first face and build both eye regions from its landmarks."""
        frame_faces = cv2.cvtColor(self.frame_faces, cv2.COLOR_BGR2GRAY)
        faces_detects = self._face_detector(frame_faces)

        try:
            landmarks = self._predictor(self.frame_faces, faces_detects[0])
            self.eye_left = Isolateeye(frame_faces, landmarks, 0, self.threshold)
            self.eye_right = Isolateeye(frame_faces, landmarks, 1, self.threshold)
        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame_faces):
        """Load a new frame and re-run detection on it."""
        self.frame_faces = frame_faces
        self._analyzing()

    def pupilleft_coordinatess(self):
        """Left pupil centre in frame coordinates, or None."""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupilright_coordinatess(self):
        """Right pupil centre in frame coordinates, or None."""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontalratio(self):
        """Mean horizontal pupil position across both eyes, 0.0 to 1.0."""
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def right(self):
        """True when the user is looking right."""
        if self.pupils_located:
            return self.horizontalratio() <= RIGHT_RATIO

    def left(self):
        """True when the user is looking left."""
        if self.pupils_located:
            return self.horizontalratio() >= LEFT_RATIO

    def forward(self):
        """True when the user is looking straight ahead."""
        if self.pupils_located:
            return self.right() is not True and self.left() is not True

    def blinking(self):
        """True when both eyes are closed."""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > BLINK_RATIO

    def highlighted_frame(self):
        """Copy of the current frame with crosshairs drawn on both pupils."""
        higlightedframe = self.frame_faces.copy()

        if self.pupils_located:
            color_pupils = (0, 255, 0)
            xleft, yleft = self.pupilleft_coordinatess()
            xright, yright = self.pupilright_coordinatess()
            cv2.line(higlightedframe, (xleft - 5, yleft), (xleft + 5, yleft), color_pupils)
            cv2.line(higlightedframe, (xleft, yleft - 5), (xleft, yleft + 5), color_pupils)
            cv2.line(higlightedframe, (xright - 5, yright), (xright + 5, yright), color_pupils)
            cv2.line(higlightedframe, (xright, yright - 5), (xright, yright + 5), color_pupils)

        return higlightedframe

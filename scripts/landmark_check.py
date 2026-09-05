"""Sanity check the shape predictor by drawing the left eye landmarks live.

Useful for confirming models/sp.dat loads and tracks before running main.py.
"""

import argparse
from pathlib import Path

import cv2
import dlib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sp.dat"


def midpoint(p1, p2):
    """Midpoint between two landmark points."""
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model missing at {MODEL_PATH}. Run scripts/fetch_models.py")

    capture = cv2.VideoCapture(args.camera)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_PATH))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for face in detector(gray):
                landmarks = predictor(gray, face)
                left = (landmarks.part(36).x, landmarks.part(36).y)
                right = (landmarks.part(39).x, landmarks.part(39).y)
                top = midpoint(landmarks.part(37), landmarks.part(38))
                bottom = midpoint(landmarks.part(41), landmarks.part(40))

                cv2.line(frame, left, right, (0, 255, 0), 2)
                cv2.line(frame, top, bottom, (0, 255, 0), 2)

            cv2.imshow("Landmark check", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

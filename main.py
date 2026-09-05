"""Drive a robot from eye direction captured on a webcam.

Classifies each frame as left, right, forward or blink, and optionally
forwards a single-character command over serial to the Arduino sketch
in arduino/eye_controlled_robot.
"""

import argparse
import csv
from pathlib import Path

import cv2

from eye_tracking import EyeTracking

# Command byte sent to the Arduino for each direction.
COMMANDS = {"forward": b"f", "right": b"r", "left": b"l", "stop": b"s"}

# Consecutive blink frames that toggle the lock.
LOCK_FRAMES = 20

CSV_HEADER = ["x_coordinate", "y_coordinate", "direction"]


def open_serial(port, baud):
    """Open the Arduino serial port, or return None when no port was given."""
    if not port:
        return None
    import serial

    return serial.Serial(port, baud, timeout=1)


def open_log(path):
    """Open the session CSV for appending, writing the header if the file is new."""
    if not path:
        return None, None
    path = Path(path)
    is_new = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", newline="")
    writer = csv.writer(handle)
    if is_new:
        writer.writerow(CSV_HEADER)
    return handle, writer


def classify(eye):
    """Return the direction label for the current frame, or None."""
    if eye.blinking():
        return "stop"
    if eye.right():
        return "right"
    if eye.left():
        return "left"
    if eye.forward():
        return "forward"
    return None


def run(camera, port, baud, log_path, model):
    """Main capture loop. Press Esc to quit."""
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}")

    eye = EyeTracking(model) if model else EyeTracking()
    link = open_serial(port, baud)
    handle, writer = open_log(log_path)

    blink_streak = 0
    locked = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            eye.refresh(frame)
            frame = eye.highlighted_frame()
            direction = classify(eye)

            if direction == "stop":
                blink_streak += 1
            else:
                blink_streak = 0

            if blink_streak == LOCK_FRAMES:
                locked = not locked
                blink_streak = 0

            if locked:
                label = "Locked"
            elif direction:
                label = direction.capitalize()
                if link:
                    link.write(COMMANDS[direction])
                pupil = eye.pupilleft_coordinatess()
                # Skip the row when the pupil was not resolved, otherwise the
                # CSV picks up short rows that break downstream plotting.
                if writer and pupil and direction != "stop":
                    writer.writerow([pupil[0], pupil[1], direction.capitalize()])
            else:
                label = ""

            cv2.putText(
                frame, label, (80, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.8, (61, 165, 235), 2,
            )
            cv2.imshow("OUTPUT VIDEO", frame)

            if cv2.waitKey(30) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if link:
            link.close()
        if handle:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--port", help="Arduino serial port, e.g. COM4 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=9600, help="serial baud rate")
    parser.add_argument("--log", help="append the session to this CSV file")
    parser.add_argument("--model", help="shape predictor path, defaults to models/sp.dat")
    args = parser.parse_args()

    run(args.camera, args.port, args.baud, args.log, args.model)


if __name__ == "__main__":
    main()

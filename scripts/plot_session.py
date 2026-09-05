"""Plot a logged session CSV: direction counts and pupil scatter."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_session.csv"


def load(path):
    """Read a session CSV and fail loudly if the expected columns are missing."""
    frame = pd.read_csv(path)
    missing = {"x_coordinate", "y_coordinate", "direction"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV, help="session CSV")
    parser.add_argument("--out", help="save the figure here instead of showing it")
    args = parser.parse_args()

    frame = load(args.csv)
    plt.style.use("ggplot")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(x="direction", data=frame, ax=axes[0])
    axes[0].set_title("Frames per direction")

    sns.scatterplot(
        x="x_coordinate", y="y_coordinate", hue="direction", data=frame, ax=axes[1]
    )
    axes[1].set_title("Left pupil position by direction")

    figure.tight_layout()
    if args.out:
        figure.savefig(args.out, dpi=150)
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

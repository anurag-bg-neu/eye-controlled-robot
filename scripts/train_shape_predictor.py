"""Train a custom 68-point shape predictor.

Preserved from Appendix A of the capstone report, which is not distributed
with this repo, where it appears as tsp.py. The script itself was not in the
project archive; it is transcribed from the report with two errors corrected
(a missing dlib import, and an options/choices variable name mismatch in the
printed listing).

Note that the model the tracker actually ships with is dlib's stock
shape_predictor_68_face_landmarks.dat, fetched by scripts/fetch_models.py.
This script is kept because the capstone did run training experiments, but
its output is not what main.py loads. It writes to models/custom_sp.dat so
it cannot clobber the stock model.

Requires an iBUG 300-W style annotation XML listing each training image and
its 68 landmark points. That file is not in this repo; generate one with
dlib's imglab tool.
"""

import argparse
import multiprocessing
from pathlib import Path

import dlib


def build_options():
    """Training hyperparameters as used for the capstone model."""
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 4
    options.nu = 0.1
    options.cascade_depth = 15
    options.feature_pool_size = 400
    options.num_test_splits = 50
    options.oversampling_amount = 5
    options.oversampling_translation_jitter = 0.1
    options.be_verbose = True
    options.num_threads = multiprocessing.cpu_count()
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="annotation XML, e.g. trained_data.xml")
    parser.add_argument("--out", default="models/custom_sp.dat", help="output model path")
    args = parser.parse_args()

    if not Path(args.xml).is_file():
        raise SystemExit(f"Annotation file not found: {args.xml}")

    options = build_options()
    print(options)

    print("training")
    dlib.train_shape_predictor(args.xml, args.out, options)

    print("evaluating")
    error = dlib.test_shape_predictor(args.xml, args.out)
    print(f"Training error: {error}")


if __name__ == "__main__":
    main()

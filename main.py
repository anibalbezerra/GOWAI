#!/usr/bin/env python3
"""
main.py — CLI for GOWAI.

Thin wrapper around HeavyIonRegressor, which handles both AuAu and
PbPb families transparently.  Training is still delegated to the
legacy HeavyIonRegressionModel path, kept for backward compatibility;
production training is done with the standalone scripts.
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))

from HeavyIonRegressor import HeavyIonRegressor   # the new unified loader
from data_processor import DataProcessor          # unchanged


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

SYSTEM_DIR = {"PbPb": "2.76_PbPb", "AuAu": "0.2_AuAu"}


def resolve_model_dir(base_dir, system, energy_or_entropy, all_or_pt):
    """Return the directory that holds the model files for a given config."""
    tag = f"{energy_or_entropy}_{all_or_pt}"
    return Path(base_dir) / SYSTEM_DIR[system] / tag


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description="Heavy Ion Collision Regression Model — GOWAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Predict on raw TRENTO event files using the shipped PbPb entropy+pt model
  python main.py --system PbPb --energy entropy --all-or-pt pt \\
                 --x-data events.txt --predict --save-predictions

  # Predict on a TFRecord (which already contains normalized inputs + labels)
  python main.py --system AuAu --energy entropy --all-or-pt pt \\
                 --tfrecord-file val.tfrecord --predict --plot-results

  # Train a new model with the legacy pipeline (not the production path)
  python main.py --system AuAu --energy entropy --all-or-pt pt \\
                 --x-data x.txt --y-data y.txt --train
        """,
    )

    # Configuration
    parser.add_argument("--system", choices=["PbPb", "AuAu"], required=True)
    parser.add_argument("--energy", "--energy-or-entropy",
                        dest="energy_or_entropy",
                        choices=["energy", "entropy"], required=True)
    parser.add_argument("--all-or-pt", dest="all_or_pt",
                        choices=["all", "pt"], required=True)

    # Model
    parser.add_argument("--trained-models-dir", type=str,
                        default=str(Path(__file__).parent / "src" / "trained_models"),
                        help="Root directory of the shipped trained models")
    parser.add_argument("--image-size", type=int, default=280)

    # Data
    parser.add_argument("--x-data", type=str,
                        help="Raw event file: N rows of H*W pixel intensities")
    parser.add_argument("--y-data", type=str,
                        help="Labels file (training only)")
    parser.add_argument("--tfrecord-file", type=str,
                        help="TFRecord with (image, sum, label) entries")
    parser.add_argument("--tfrecord-dir", type=str,
                        help="Directory containing train/val TFRecords (training only)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--plot-results", action="store_true")

    # Modes
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    # Training-only (legacy)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")

    return parser


# ---------------------------------------------------------------------
# Prediction — uses HeavyIonRegressor, handles raw and TFRecord inputs
# ---------------------------------------------------------------------

def run_prediction(args, output_dir):
    model_dir = resolve_model_dir(
        args.trained_models_dir, args.system,
        args.energy_or_entropy, args.all_or_pt,
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"\n{'='*60}\nPREDICTION MODE\n{'='*60}")
    print(f"  System    : {args.system}")
    print(f"  Observable: {args.energy_or_entropy}")
    print(f"  Selection : {args.all_or_pt}")
    print(f"  Model dir : {model_dir}")

    regressor = HeavyIonRegressor(model_dir, system=(
        "0.2_AuAu" if args.system == "AuAu" else "2.76_PbPb"
    )).load()

    # ---- Load input --------------------------------------------------
    if args.tfrecord_file:
        # TFRecord path: the file already contains the preprocessed dual-
        # channel image and normalized sum, so we bypass the regressor's
        # raw-input preprocessor and call the model directly.
        images_norm, sums_norm, labels = _read_tfrecord(args.tfrecord_file)
        print(f"  Loaded {len(images_norm)} events from TFRecord")

        preds = regressor.model(
            {"image_input": images_norm,
             "sum_input": sums_norm,
             **({"pt_true_input": np.zeros_like(sums_norm)}
                if regressor.pre.needs_pt_true_input else {})},
            training=False,
        ).numpy()
        nch_pred, pt_pred = regressor.pre.to_physical(preds)

        if labels is not None:
            nch_true, pt_true = regressor.pre.to_physical(labels)
        else:
            nch_true = pt_true = None

    elif args.x_data:
        # Raw input path: (N, H, W) integer pixel intensities.  This is
        # the case the distributed pipeline is meant to serve.
        raw = np.loadtxt(args.x_data)
        raw = raw.reshape(-1, args.image_size, args.image_size)
        print(f"  Loaded {len(raw)} raw events from {args.x_data}")

        nch_pred, pt_pred = regressor.predict(raw)
        nch_true = pt_true = None

    else:
        raise ValueError("Prediction requires --x-data or --tfrecord-file")

    # ---- Save -------------------------------------------------------
    if args.save_predictions:
        out = output_dir / "predictions.csv"
        if nch_true is not None:
            data = np.column_stack([nch_true, pt_true, nch_pred, pt_pred])
            header = "Nch_true,Pt_true,Nch_pred,Pt_pred"
        else:
            data = np.column_stack([nch_pred, pt_pred])
            header = "Nch,Pt"
        np.savetxt(out, data, delimiter=",", header=header, comments="")
        print(f"  Predictions saved to {out}")

    # ---- Plots ------------------------------------------------------
    if args.plot_results and nch_true is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, true, pred, name in [(ax1, nch_true, nch_pred, "Nch"),
                                     (ax2, pt_true, pt_pred, "p_T")]:
            ax.scatter(true, pred, alpha=0.5, s=20)
            lo = min(true.min(), pred.min())
            hi = max(true.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=2)
            ax.set_xlabel(f"True {name}")
            ax.set_ylabel(f"Predicted {name}")
            ax.set_title(f"{name}: true vs predicted")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_out = output_dir / "predictions_vs_truth.png"
        plt.savefig(plot_out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved to {plot_out}")

    print(f"\n  Sample predictions (Nch, pT):")
    for i in range(min(5, len(nch_pred))):
        print(f"    {i+1}: Nch={nch_pred[i]:.4f}  pT={pt_pred[i]:.6f}")

    return nch_pred, pt_pred


def _read_tfrecord(tfrecord_file):
    """Read a single TFRecord written by preprocessing_v3/v4.

    Returns (images_norm, sums_norm, labels_norm).  All arrays are
    already in the normalized space used during training.
    """
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([2], tf.float32),
        "sum":   tf.io.FixedLenFeature([1], tf.float32),
    }

    images, sums, labels = [], [], []
    for raw in tf.data.TFRecordDataset(tfrecord_file):
        p = tf.io.parse_single_example(raw, feature)
        img = tf.reshape(tf.io.decode_raw(p["image"], tf.float32), (280, 280, 2))
        images.append(img.numpy())
        sums.append(p["sum"].numpy())
        labels.append(p["label"].numpy())

    if not images:
        raise ValueError(f"Empty TFRecord: {tfrecord_file}")
    return (np.stack(images).astype(np.float32),
            np.stack(sums).astype(np.float32),
            np.stack(labels).astype(np.float32))


# ---------------------------------------------------------------------
# Training — legacy, kept for backward compatibility
# ---------------------------------------------------------------------

def run_training(args, output_dir):
    """Legacy training path.

    Production training is done with the standalone regressor_v5 and
    regressor_v7 scripts, which implement the evidential head, the
    polynomial baseline, the pinball loss, and the gap-aware checkpoint.
    This entry point is kept only so that the CLI is not broken; it does
    NOT reproduce the shipped models.
    """
    print(f"\n{'='*60}\nTRAINING MODE (legacy)\n{'='*60}")
    print("WARNING: This path builds a plain ResNet-50 regressor and does")
    print("         not reproduce the shipped AuAu / PbPb models.")
    print("         Use regressor_v5.py / regressor_v7.py for production.")

    from HeavyIonRegressionModel import HeavyIonRegressionModel

    model_handler = HeavyIonRegressionModel(image_size=args.image_size)
    model_handler.build_model()
    data_processor = DataProcessor(image_size=args.image_size)

    if not (args.tfrecord_dir or args.x_data):
        raise ValueError("Training requires --tfrecord-dir or --x-data")

    if args.tfrecord_dir:
        train_ds = data_processor.load_tfrecord_dataset(
            str(Path(args.tfrecord_dir) / "train_dataset.tfrecord"),
            args.batch_size,
        )
        val_ds = data_processor.load_tfrecord_dataset(
            str(Path(args.tfrecord_dir) / "val_dataset.tfrecord"),
            args.batch_size, shuffle=False,
        )
    else:
        x_train, x_val, y_train, y_val = data_processor.load_text_data(
            args.x_data, args.y_data
        )
        model_handler.scaling_factors = {
            "max_image": float(np.max(x_train)),
            "max_labels": [float(np.max(y_train, axis=0)[0]),
                           float(np.max(y_train, axis=0)[1])],
            "max_sum": float(np.max(np.sum(x_train, axis=(1, 2)))),
        }
        tf_dir = output_dir / "tfrecords"
        tf_dir.mkdir(parents=True, exist_ok=True)
        data_processor.create_tfrecord(
            x_train, y_train, str(tf_dir / "train_dataset.tfrecord"),
            model_handler.scaling_factors,
        )
        data_processor.create_tfrecord(
            x_val, y_val, str(tf_dir / "val_dataset.tfrecord"),
            model_handler.scaling_factors,
        )
        train_ds = data_processor.load_tfrecord_dataset(
            str(tf_dir / "train_dataset.tfrecord"), args.batch_size
        )
        val_ds = data_processor.load_tfrecord_dataset(
            str(tf_dir / "val_dataset.tfrecord"), args.batch_size, shuffle=False
        )

    history = model_handler.model.fit(
        train_ds, epochs=args.epochs, validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(patience=10,
                                             restore_best_weights=True),
        ],
    )

    model_handler.save_model(str(output_dir), "legacy_model")
    print(f"  Training finished.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("CNN4QGP-Hydro: Heavy Ion Collision Regression Model")
    print("="*60)

    args = setup_argument_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train and args.predict:
        raise ValueError("Cannot specify both --train and --predict")

    if args.predict:
        run_prediction(args, output_dir)
    elif args.train:
        run_training(args, output_dir)
    else:
        print("\nNo mode requested. Use --predict or --train.")


if __name__ == "__main__":
    main()
"""
TensorFlow FC classifier for MeV energy-bin identification.
"""

import csv
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_utils import (
    bin_drm,
    generate_synthetic_data,
    load_drm,
    normalize_apply,
    normalize_fit,
)

#Config
XLSX_PATH       = "res/200x200.xlsx"
N_VALUES        = [10, 20, 50, 100, 200]
SAMPLES_PER_BIN = 100
MAX_EPOCHS      = 300
BATCH_SIZE      = 32
PATIENCE        = 40          # early-stop patience on val_accuracy
SEED            = 42

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warnings


def build_model(n_classes: int) -> tf.keras.Model:
    """200 → 512 → 256 → 128 → n_classes FC network with BatchNorm + ReLU."""
    inp = tf.keras.Input(shape=(200,), name="detector_response")
    x = tf.keras.layers.Dense(512)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="energy_bin")(x)
    return tf.keras.Model(inp, out, name=f"mev_classifier_n{n_classes}")


class EpochMetricsCallback(tf.keras.callbacks.Callback):
    """Compute sklearn precision / recall / F1 (macro) on validation set each epoch."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.history: dict[str, list] = {
            "precision": [],
            "recall": [],
            "f1": [],
            "efficiency": [],   # physics term for recall
        }

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        y_pred = np.argmax(
            self.model.predict(self.X_val, verbose=0, batch_size=256), axis=1
        )
        prec = precision_score(self.y_val, y_pred, average="macro", zero_division=0)
        rec  = recall_score(   self.y_val, y_pred, average="macro", zero_division=0)
        f1   = f1_score(       self.y_val, y_pred, average="macro", zero_division=0)

        self.history["precision"].append(float(prec))
        self.history["recall"].append(   float(rec))
        self.history["f1"].append(        float(f1))
        self.history["efficiency"].append(float(rec))

        # Print a summary line every 10 epochs
        if (epoch + 1) % 10 == 0 and logs is not None:
            print(
                f"  ep {epoch+1:3d} | "
                f"loss {logs.get('loss', 0):.4f} | "
                f"val_loss {logs.get('val_loss', 0):.4f} | "
                f"val_acc {logs.get('val_accuracy', 0):.4f} | "
                f"F1 {f1:.4f} | eff {rec:.4f}"
            )


def train_for_n(
    drm: np.ndarray,
    n: int,
    samples_per_bin: int,
    rng: np.random.Generator,
) -> tuple[dict, tf.keras.Model]:
    print(f"\n{'='*65}")
    print(f"  n = {n}  |  {50/n:.2f} MeV/bin  |  {n * samples_per_bin} total samples")
    print(f"{'='*65}")

    #Data
    drm_binned = bin_drm(drm, n)                              # (200, n)
    X, y = generate_synthetic_data(drm_binned, samples_per_bin, rng)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    mean, std   = normalize_fit(X_train)
    X_train_n   = normalize_apply(X_train, mean, std)
    X_val_n     = normalize_apply(X_val,   mean, std)

    #Model
    tf.random.set_seed(SEED)
    model = build_model(n)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    metrics_cb  = EpochMetricsCallback(X_val_n, y_val)
    early_stop  = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1
    )
    reduce_lr   = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
    )

    # Training
    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[metrics_cb, early_stop, reduce_lr],
        verbose=0,
    )

    epochs_run = len(history.history["loss"])

    # Final evaluation on validation set
    y_pred_final = np.argmax(
        model.predict(X_val_n, verbose=0, batch_size=256), axis=1
    )
    cm = confusion_matrix(y_val, y_pred_final, labels=np.arange(n))
    np.save(f"results_n{n}_confusion.npy", cm)

    final_acc  = float(history.history["val_accuracy"][-1])
    final_f1   = metrics_cb.history["f1"][-1]
    final_eff  = metrics_cb.history["efficiency"][-1]

    print(f"\n  Stopped at epoch {epochs_run}")
    print(f"  Val accuracy : {final_acc:.4f}")
    print(f"  Macro F1     : {final_f1:.4f}")
    print(f"  Efficiency   : {final_eff:.4f}  (macro recall)")

    results = {
        "n":              n,
        "epochs_trained": epochs_run,
        "mev_per_bin":    round(50.0 / n, 4),
        "train_loss":     [float(v) for v in history.history["loss"]],
        "val_loss":       [float(v) for v in history.history["val_loss"]],
        "train_accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy":   [float(v) for v in history.history["val_accuracy"]],
        "precision":      metrics_cb.history["precision"],
        "recall":         metrics_cb.history["recall"],
        "f1":             metrics_cb.history["f1"],
        "efficiency":     metrics_cb.history["efficiency"],
        "norm_mean":      mean.tolist(),
        "norm_std":       std.tolist(),
    }
    return results, model


def save_csv_results(all_results: dict) -> None:
    """Write two CSVs: a summary (one row per n) and per-epoch detail rows."""
    # Summary: one row per n with final-epoch metrics
    summary_path = "results_summary.csv"
    summary_fields = [
        "n", "mev_per_bin", "epochs_trained",
        "val_accuracy", "macro_f1", "macro_precision", "macro_recall", "efficiency",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for res in all_results.values():
            w.writerow({
                "n":               res["n"],
                "mev_per_bin":     res["mev_per_bin"],
                "epochs_trained":  res["epochs_trained"],
                "val_accuracy":    round(res["val_accuracy"][-1], 6),
                "macro_f1":        round(res["f1"][-1], 6),
                "macro_precision": round(res["precision"][-1], 6),
                "macro_recall":    round(res["recall"][-1], 6),
                "efficiency":      round(res["efficiency"][-1], 6),
            })
    print(f"  Saved {summary_path}")

    # Per-epoch: one file per n
    epoch_fields = [
        "epoch", "train_loss", "val_loss",
        "train_accuracy", "val_accuracy",
        "precision", "recall", "f1", "efficiency",
    ]
    for res in all_results.values():
        n = res["n"]
        epoch_path = f"results_epochs_n{n}.csv"
        with open(epoch_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=epoch_fields)
            w.writeheader()
            for ep in range(res["epochs_trained"]):
                w.writerow({
                    "epoch":          ep + 1,
                    "train_loss":     round(res["train_loss"][ep], 6),
                    "val_loss":       round(res["val_loss"][ep], 6),
                    "train_accuracy": round(res["train_accuracy"][ep], 6),
                    "val_accuracy":   round(res["val_accuracy"][ep], 6),
                    "precision":      round(res["precision"][ep], 6),
                    "recall":         round(res["recall"][ep], 6),
                    "f1":             round(res["f1"][ep], 6),
                    "efficiency":     round(res["efficiency"][ep], 6),
                })
        print(f"  Saved {epoch_path}")


def print_summary_table(all_results: dict) -> None:
    print("\n" + "=" * 65)
    print(f"  {'n':>4}  {'MeV/bin':>8}  {'epochs':>7}  "
          f"{'val_acc':>8}  {'F1':>7}  {'efficiency':>10}")
    print("  " + "-" * 60)
    for key, res in all_results.items():
        print(
            f"  {res['n']:>4}  {res['mev_per_bin']:>8.2f}  "
            f"{res['epochs_trained']:>7}  "
            f"{res['val_accuracy'][-1]:>8.4f}  "
            f"{res['f1'][-1]:>7.4f}  "
            f"{res['efficiency'][-1]:>10.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM loaded: shape={drm.shape}  min={drm.min():.3f}  max={drm.max():.3f}")

    # Load any existing results so already-trained n values can be skipped
    json_path = "training_results.json"
    all_results: dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)

    for n in N_VALUES:
        model_path = f"model_mev_n{n}.keras"
        if os.path.exists(model_path) and str(n) in all_results:
            print(f"\n  n={n}: {model_path} exists — skipping training.")
            continue
        results, model = train_for_n(drm, n, SAMPLES_PER_BIN, rng)
        all_results[str(n)] = results
        model.save(model_path)
        # Persist after each n so a crash mid-run doesn't lose earlier work
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {model_path}")

    print_summary_table(all_results)

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    print("\nSaving CSVs...")
    save_csv_results(all_results)
    print("Run visualize_mev.py to generate all figures.")

"""
train_pff_bounded_gated.py — PFF parameter regressor, v2: two fixes over
train_pff.py's ReLU-mean + heteroscedastic-NLL head (model_pff_relu_uncertainty.keras),
tried side by side rather than replacing it.

Problem this addresses: on real (out-of-distribution) shots, that model's
per-parameter sigma routinely saturated at its clip ceiling (logvar=+-10,
i.e. sigma_norm=exp(5)=148.4x a parameter's own physical range) -- e.g.
a2=+-146.93, a3=+-14841.32 on real shots. Both numbers are literally the same
saturation event, just scaled by each parameter's own (very different) unit
range -- a3's range (0-100) is ~100x a2's (0.01-1), so the identical
saturated head produces a far more dramatic-looking absolute sigma. Fixing
the saturation mechanism therefore fixes a2 and a3 (and a4, which saturated
identically on some shots) together, not one at a time. Two independent
changes, both tried here:

1. Bounded log-variance activation (tanh-scaled) instead of a raw linear
   layer + post-hoc clip, for every continuous head. A hard clip has zero
   gradient past the boundary -- once training pushes a unit to the ceiling,
   nothing pulls it back. tanh keeps a gradient everywhere and caps sigma at
   LOGVAR_MAX regardless of how far an input extrapolates, instead of the
   previous +-10 clip (max sigma_norm = exp(5) ~= 148.4). LOGVAR_MAX=4 caps
   it at exp(2) ~= 7.4 -- a ~20x tighter ceiling.

2. Gated bump structure for a3/a4/a5/a6. a3 (bump amplitude) is bimodal by
   construction (data_utils.sample_pff_spectra: exactly 0 for half the
   training data, ~[5,100] for the rest) -- a single Gaussian NLL head has
   no way to represent "either exactly 0, or somewhere in a wide range" and
   is forced to inflate sigma to hedge between the two regimes even
   in-distribution. This splits it into an explicit binary classifier (bump
   present or not, trained with BCE) plus a magnitude regression for
   a3/a4/a5/a6 conditioned on -- and only supervised on -- bump-present
   samples, combined at decode time via the standard Bernoulli-Gaussian
   mixture mean/variance:
     E[a3]   = p_bump * mu_given_bump
     Var[a3] = p_bump*sigma_given_bump^2 + p_bump*(1-p_bump)*mu_given_bump^2
   a4/a5/a6 (bump centre/energy-dependent-width coefficients) are undefined
   when there's no bump, so they're reported as their given-bump values
   directly rather than mixed with 0.

   [Session note: originally a4/a5 (centre/fixed width); a5 was later
   repurposed and a6 added when data_utils.pff_func's bump denominator
   changed from a constant a5 to an energy-dependent a5*x + a6/x. The gating
   logic here is unaffected by that -- it just grew from 3 to 4
   bump-conditional outputs.]

Entirely separate from train_pff.py / model_pff_relu_uncertainty.keras --
kept side by side to compare, not a replacement. Shares data_utils.py
(PFF_PARAM_BOUNDS, normalize_pff_params, generate_pff_training_data, ...)
but not train_pff.py's pff_nll_loss/build_model/PFFMetricsCallback, since
the output layout here (13-wide, gated) is structurally different from
train_pff.py's (12-wide, mu+logvar per parameter) -- see decode_v2 below
for why a shared decode helper doesn't apply either.

Usage
-----
  python train_pff_bounded_gated.py

Outputs
-------
  model_pff_v2.keras
  pff_training_results_v2.json
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_utils import (
    PFF_PARAM_BOUNDS,
    denormalize_pff_params,
    load_drm,
    mev_bin_centers,
    normalize_apply,
    normalize_fit,
    normalize_pff_params,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Config -- identical to train_pff.py's so the two are comparable apples-to-apples
XLSX_PATH     = "res/drm/200x200.xlsx"
N_SAMPLES     = 20_000
BUMP_FRACTION = 0.5
MAX_EPOCHS    = 300
BATCH_SIZE    = 64
PATIENCE      = 40
SEED          = 42
LEARNING_RATE = 2e-4

LOGVAR_MAX = 4.0   # tanh-bounded logvar in (-4, 4) -> sigma_norm in (0.135, 7.39);
                   # vs train_pff.py's hard clip at +-10 -> sigma_norm up to 148.4

MODEL_PATH   = "model_pff_v2.keras"
RESULTS_JSON = "pff_training_results_v2.json"

PARAM_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> tf.keras.Model:
    """
    200 -> 512 -> 256 -> 128 -> gated heads:
      - a1, a2      : ReLU mean + tanh-bounded logvar (plain continuous, unimodal)
      - bump_logit  : P(bump present) pre-sigmoid logit, trained with BCE
      - a3,a4,a5,a6 : ReLU mean + tanh-bounded logvar, magnitude/position/width
                      GIVEN a bump is present (only supervised on bump-present
                      samples -- see pff_loss_v2). a5,a6 are the two
                      coefficients of the energy-dependent bump width
                      a5*x + a6/x (data_utils.pff_func), both undefined
                      without a bump exactly like a4.

    Output: 13-wide concat
      [a1_mu, a2_mu, a1_logvar, a2_logvar, bump_logit,
       a3_mu, a4_mu, a5_mu, a6_mu, a3_logvar, a4_logvar, a5_logvar, a6_logvar]
    """
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

    a12_mean = tf.keras.layers.Dense(2, activation="relu", name="a12_mean")(x)
    a12_logvar_tanh = tf.keras.layers.Dense(2, activation="tanh", name="a12_logvar_tanh")(x)
    a12_logvar = tf.keras.layers.Rescaling(scale=LOGVAR_MAX, name="a12_logvar")(a12_logvar_tanh)

    bump_logit = tf.keras.layers.Dense(1, name="bump_logit")(x)

    a3456_mean = tf.keras.layers.Dense(4, activation="relu", name="a3456_mean")(x)
    a3456_logvar_tanh = tf.keras.layers.Dense(4, activation="tanh", name="a3456_logvar_tanh")(x)
    a3456_logvar = tf.keras.layers.Rescaling(scale=LOGVAR_MAX, name="a3456_logvar")(a3456_logvar_tanh)

    out = tf.keras.layers.Concatenate(name="pff_params_v2")(
        [a12_mean, a12_logvar, bump_logit, a3456_mean, a3456_logvar]
    )
    return tf.keras.Model(inp, out, name="pff_regressor_v2")


def pff_loss_v2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Three terms over the gated v2 head (see build_model):
      1. Heteroscedastic NLL for a1, a2 -- plain continuous, no gating needed.
      2. Binary cross-entropy for bump presence (label = a3_true > 0).
      3. Heteroscedastic NLL for a3, a4, a5, a6 given bump present, masked to
         bump-present samples only -- unlike train_pff.py's continuous-a3
         weighting trick, this uses the TRUE bump label as a hard 0/1 mask,
         since bump presence is now its own explicit target (term 2) rather
         than something inferred from a3's own magnitude.
    """
    a12_mu, a12_logvar     = y_pred[:, 0:2], y_pred[:, 2:4]
    bump_logit             = y_pred[:, 4:5]
    a3456_mu, a3456_logvar = y_pred[:, 5:9], y_pred[:, 9:13]

    a12_true   = y_true[:, 0:2]
    a3_true    = y_true[:, 2:3]
    a3456_true = y_true[:, 2:6]
    bump_true  = tf.cast(a3_true > 0.0, tf.float32)          # (B, 1)

    nll_12  = 0.5 * (tf.exp(-a12_logvar) * tf.square(a12_true - a12_mu) + a12_logvar)
    loss_12 = tf.reduce_mean(nll_12)

    bce       = tf.keras.losses.binary_crossentropy(bump_true, bump_logit, from_logits=True)
    loss_bump = tf.reduce_mean(bce)

    nll_3456  = 0.5 * (tf.exp(-a3456_logvar) * tf.square(a3456_true - a3456_mu) + a3456_logvar)
    mask_3456 = tf.repeat(bump_true, 4, axis=1)               # a3,a4,a5,a6 all bump-conditional
    loss_3456 = tf.reduce_sum(nll_3456 * mask_3456) / (tf.reduce_sum(mask_3456) + 1e-6)

    return loss_12 + loss_bump + loss_3456


def decode_v2(pff_out: np.ndarray, param_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the 13-wide v2 output into physical-units (mean, sigma, p_bump).

    a3's reported mean/sigma are the Bernoulli-Gaussian mixture mean/variance
    (p_bump * mu_given_bump, mixture variance) -- a3=0 should read as
    "probably no bump" when p_bump is low, not "bump of size mu_given_bump,
    reported regardless of whether one exists". a4/a5/a6 are reported as
    their given-bump values directly, since bump position/width conditioned
    on "no bump" isn't a meaningful quantity to mix toward.

    Works on a single sample, shape (13,) -> (6,),(6,),(1,), or a batch,
    shape (N, 13) -> (N,6),(N,6),(N,1).
    """
    a12_mu_n      = pff_out[..., 0:2]
    a12_logvar    = pff_out[..., 2:4]
    bump_logit    = pff_out[..., 4:5]
    a3456_mu_n    = pff_out[..., 5:9]
    a3456_logvar  = pff_out[..., 9:13]

    p_bump  = 1.0 / (1.0 + np.exp(-bump_logit))              # (..., 1)
    sigma12_n   = np.exp(0.5 * np.clip(a12_logvar, -20.0, 20.0))
    sigma3456_n = np.exp(0.5 * np.clip(a3456_logvar, -20.0, 20.0))

    lo, hi = param_bounds[:, 0], param_bounds[:, 1]
    span = hi - lo

    a12_mu_phys    = a12_mu_n * span[0:2] + lo[0:2]
    a12_sigma_phys = sigma12_n * span[0:2]

    a3456_mu_phys_given    = a3456_mu_n * span[2:6] + lo[2:6]
    a3456_sigma_phys_given = sigma3456_n * span[2:6]

    a3_mu_given    = a3456_mu_phys_given[..., 0:1]
    a3_sigma_given = a3456_sigma_phys_given[..., 0:1]
    a3_mu_mix  = p_bump * a3_mu_given
    a3_var_mix = p_bump * a3_sigma_given ** 2 + p_bump * (1.0 - p_bump) * a3_mu_given ** 2
    a3_sigma_mix = np.sqrt(np.maximum(a3_var_mix, 0.0))

    mu_phys    = np.concatenate([a12_mu_phys, a3_mu_mix, a3456_mu_phys_given[..., 1:4]], axis=-1)
    sigma_phys = np.concatenate([a12_sigma_phys, a3_sigma_mix, a3456_sigma_phys_given[..., 1:4]], axis=-1)

    mu_phys = np.clip(mu_phys, lo, hi)
    return mu_phys, sigma_phys, p_bump


# ---------------------------------------------------------------------------
# Metrics / spectrum reconstruction (mirrors train_pff.py's helpers)
# ---------------------------------------------------------------------------

def _pff_batch(energy_bins: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Vectorised PFF evaluation. params shape (N, 6), returns (N, len(energy_bins))."""
    a1, a2, a3, a4, a5, a6 = (params[:, j] for j in range(6))
    x = energy_bins[np.newaxis, :]
    brems = a1[:, None] * np.exp(-a2[:, None] * x)
    width = a5[:, None] * x + a6[:, None] / x
    bump  = a3[:, None] * np.exp(-(x - a4[:, None]) ** 2 / width)
    return brems + bump


class PFFMetricsCallbackV2(tf.keras.callbacks.Callback):
    """
    Same role as train_pff.py's PFFMetricsCallback, adapted for the gated
    v2 head: adds bump-classifier accuracy, decodes a3 via the
    Bernoulli-Gaussian mixture (decode_v2), and additionally tracks max
    sigma per parameter -- the direct check on whether tanh-bounding
    actually capped the saturation blowups train_pff.py showed.
    """

    def __init__(self, X_val: np.ndarray, y_val_norm: np.ndarray, energy_bins: np.ndarray) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val_norm = y_val_norm
        self.energy_bins = energy_bins
        self.bump_mask = y_val_norm[:, 2] > 0.0
        self.history: dict[str, list] = (
            {f"mae_{n}": [] for n in PARAM_NAMES}
            | {f"mae_{n}_bump": [] for n in PARAM_NAMES}
            | {f"max_sigma_{n}": [] for n in PARAM_NAMES}
            | {"spectrum_mse": [], "spectrum_rel_mse": [],
               "coverage_1sigma": [], "coverage_1sigma_bump": [], "bump_accuracy": []}
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        pff_out = self.model.predict(self.X_val, verbose=0, batch_size=256)
        y_pred, sigma_pred, p_bump = decode_v2(pff_out, PFF_PARAM_BOUNDS)
        y_true = denormalize_pff_params(self.y_val_norm)
        bmask = self.bump_mask

        for j, name in enumerate(PARAM_NAMES):
            abs_err = np.abs(y_pred[:, j] - y_true[:, j])
            self.history[f"mae_{name}"].append(float(abs_err.mean()))
            self.history[f"max_sigma_{name}"].append(float(sigma_pred[:, j].max()))
            if bmask.any():
                self.history[f"mae_{name}_bump"].append(float(abs_err[bmask].mean()))
            else:
                self.history[f"mae_{name}_bump"].append(float("nan"))

        z = (y_pred - y_true) / np.maximum(sigma_pred, 1e-6)
        self.history["coverage_1sigma"].append(float(np.mean(np.abs(z[:, :3]) <= 1.0)))
        self.history["coverage_1sigma_bump"].append(
            float(np.mean(np.abs(z[bmask][:, 3:]) <= 1.0)) if bmask.any() else float("nan")
        )

        bump_pred_label = p_bump[:, 0] > 0.5
        self.history["bump_accuracy"].append(float(np.mean(bump_pred_label == bmask)))

        s_true = _pff_batch(self.energy_bins, y_true)
        s_pred = _pff_batch(self.energy_bins, y_pred)
        sq_err = (s_true - s_pred) ** 2
        self.history["spectrum_mse"].append(float(sq_err.mean()))
        bump_bins = self.energy_bins > 5.0
        s_true_b = s_true[:, bump_bins]
        sq_err_b = sq_err[:, bump_bins]
        rel_sq = sq_err_b / np.maximum(s_true_b, 1.0) ** 2
        self.history["spectrum_rel_mse"].append(float(rel_sq.mean()))

        if logs is not None:
            logs["spec_mse"] = self.history["spectrum_mse"][-1]

        if (epoch + 1) % 10 == 0 and logs:
            mae_all  = " | ".join(f"{n}={self.history[f'mae_{n}'][-1]:.3f}"      for n in PARAM_NAMES)
            mae_bump = " | ".join(f"{n}={self.history[f'mae_{n}_bump'][-1]:.3f}" for n in PARAM_NAMES)
            max_sig  = " | ".join(f"{n}={self.history[f'max_sigma_{n}'][-1]:.2f}" for n in PARAM_NAMES)
            print(
                f"  ep {epoch+1:3d} | val_loss {logs.get('val_loss', 0):.5f} | "
                f"spec_mse {self.history['spectrum_mse'][-1]:.2f} | "
                f"rel_mse {self.history['spectrum_rel_mse'][-1]:.4f} | "
                f"1sig cov {self.history['coverage_1sigma'][-1]*100:.0f}% "
                f"(bump {self.history['coverage_1sigma_bump'][-1]*100:.0f}%) | "
                f"bump_acc {self.history['bump_accuracy'][-1]*100:.0f}%\n"
                f"           all : {mae_all}\n"
                f"           bump: {mae_bump}\n"
                f"           max sigma: {max_sig}"
            )


def _timed_generate(drm, n_samples, rng, bump_fraction):
    from data_utils import sample_pff_spectra

    energy_bins = mev_bin_centers(drm.shape[1])

    t0 = time.perf_counter()
    spectra, params = sample_pff_spectra(n_samples, energy_bins, rng, bump_fraction)
    sample_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    responses = (drm @ spectra.T).T
    sigma = np.sqrt(np.maximum(responses, 1e-8))
    noise = rng.standard_normal(responses.shape) * sigma
    X = np.clip(responses + noise, 0.0, None)
    row_sums = X.sum(axis=1, keepdims=True)
    X = (X / np.maximum(row_sums, 1e-12)).astype(np.float32)
    numpy_time = time.perf_counter() - t1

    return X, params.astype(np.float32), sample_time, numpy_time


if __name__ == "__main__":
    t_start = time.perf_counter()

    rng = np.random.default_rng(SEED)
    drm = load_drm(XLSX_PATH)
    print(f"DRM shape: {drm.shape}  min={drm.min():.3f}  max={drm.max():.3f}")

    n_bump = int(N_SAMPLES * BUMP_FRACTION)
    n_no_bump = N_SAMPLES - n_bump
    print(f"Generating {N_SAMPLES} samples ({n_bump} with bump, {n_no_bump} without)...")

    X, y_params, t_sample, t_np = _timed_generate(drm, N_SAMPLES, rng, BUMP_FRACTION)
    y_norm = normalize_pff_params(y_params)

    print(f"  Sampling loop    : {t_sample:.2f}s")
    print(f"  NumPy (DRM+noise): {t_np:.3f}s")
    print(f"X shape: {X.shape}  y shape: {y_params.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_norm, test_size=0.2, random_state=SEED
    )
    mean, std = normalize_fit(X_train)
    X_train_n = normalize_apply(X_train, mean, std)
    X_val_n = normalize_apply(X_val, mean, std)

    tf.random.set_seed(SEED)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=pff_loss_v2,
    )
    model.summary()

    energy_bins = mev_bin_centers(drm.shape[1])
    metrics_cb = PFFMetricsCallbackV2(X_val_n, y_val, energy_bins)
    spec_ckpt = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="spec_mse", save_best_only=True, mode="min", verbose=0,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=False, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0
    )

    t_train_start = time.perf_counter()
    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[metrics_cb, spec_ckpt, early_stop, reduce_lr],
        verbose=0,
    )
    t_train = time.perf_counter() - t_train_start

    epochs_run = len(history.history["loss"])
    best_ep = int(np.argmin(metrics_cb.history["spectrum_mse"]))

    print(f"\nStopped at epoch {epochs_run}  (best epoch: {best_ep + 1})")
    print(f"Training wall time : {t_train:.1f}s  ({t_train/epochs_run:.2f}s/epoch)")
    print(f"Best val_loss      : {history.history['val_loss'][best_ep]:.6f}")
    print(f"Spectrum MSE       : {metrics_cb.history['spectrum_mse'][best_ep]:.4f}  (at best epoch)")
    print(f"Spectrum rel-MSE   : {metrics_cb.history['spectrum_rel_mse'][best_ep]:.6f}  (bump region, E>5 MeV)")
    print(f"Bump classifier acc: {metrics_cb.history['bump_accuracy'][best_ep]*100:.1f}%")
    print(f"{'Param':>4}  {'MAE (all)':>10}  {'MAE (bump only)':>15}  {'Max sigma':>10}  (at best epoch)")
    for n in PARAM_NAMES:
        print(f"  {n:>2}  {metrics_cb.history[f'mae_{n}'][best_ep]:>10.4f}  "
              f"{metrics_cb.history[f'mae_{n}_bump'][best_ep]:>15.4f}  "
              f"{metrics_cb.history[f'max_sigma_{n}'][best_ep]:>10.2f}")
    print(f"1-sigma coverage   : a1/a2/a3={metrics_cb.history['coverage_1sigma'][best_ep]*100:.0f}% "
          f"(want ~68%)  a4/a5(bump only)={metrics_cb.history['coverage_1sigma_bump'][best_ep]*100:.0f}%")

    model.save(MODEL_PATH)

    results = {
        "epochs_trained":        epochs_run,
        "best_epoch":            best_ep + 1,
        "n_samples":             N_SAMPLES,
        "bump_fraction":         BUMP_FRACTION,
        "learning_rate":         LEARNING_RATE,
        "logvar_max":            LOGVAR_MAX,
        "param_bounds":          PFF_PARAM_BOUNDS.tolist(),
        "param_names":           PARAM_NAMES,
        "train_loss":            [float(v) for v in history.history["loss"]],
        "val_loss":              [float(v) for v in history.history["val_loss"]],
        "best_val_loss":         float(history.history["val_loss"][best_ep]),
        "spectrum_mse":          metrics_cb.history["spectrum_mse"],
        "spectrum_rel_mse":      metrics_cb.history["spectrum_rel_mse"],
        "best_spectrum_mse":     metrics_cb.history["spectrum_mse"][best_ep],
        "best_spectrum_rel_mse": metrics_cb.history["spectrum_rel_mse"][best_ep],
        "coverage_1sigma":       metrics_cb.history["coverage_1sigma"],
        "coverage_1sigma_bump":  metrics_cb.history["coverage_1sigma_bump"],
        "best_coverage_1sigma":      metrics_cb.history["coverage_1sigma"][best_ep],
        "best_coverage_1sigma_bump": metrics_cb.history["coverage_1sigma_bump"][best_ep],
        "bump_accuracy":         metrics_cb.history["bump_accuracy"],
        "best_bump_accuracy":    metrics_cb.history["bump_accuracy"][best_ep],
        "norm_mean":             mean.tolist(),
        "norm_std":              std.tolist(),
        **{f"mae_{n}":           metrics_cb.history[f"mae_{n}"]      for n in PARAM_NAMES},
        **{f"mae_{n}_bump":      metrics_cb.history[f"mae_{n}_bump"] for n in PARAM_NAMES},
        **{f"best_mae_{n}":      metrics_cb.history[f"mae_{n}"][best_ep]      for n in PARAM_NAMES},
        **{f"best_mae_{n}_bump": metrics_cb.history[f"mae_{n}_bump"][best_ep] for n in PARAM_NAMES},
        **{f"max_sigma_{n}":     metrics_cb.history[f"max_sigma_{n}"] for n in PARAM_NAMES},
        **{f"best_max_sigma_{n}": metrics_cb.history[f"max_sigma_{n}"][best_ep] for n in PARAM_NAMES},
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {MODEL_PATH} and {RESULTS_JSON}")

    t_total = time.perf_counter() - t_start
    print(f"\n--- Timing summary ---")
    print(f"  Sampling loop     : {t_sample:.2f}s")
    print(f"  NumPy (DRM+noise) : {t_np:.3f}s")
    print(f"  Training          : {t_train:.1f}s  ({epochs_run} epochs, {t_train/epochs_run:.2f}s/epoch)")
    print(f"  Total wall time   : {t_total:.1f}s")

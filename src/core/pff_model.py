"""
pff_model.py — shared PFF regressor architecture (v2: tanh-bounded logvar +
gated bump classifier), loss, decode helper, and validation-metrics
callback, extracted out of train_pff_bounded_gated.py so every PFF trainer,
plus evaluate_pff_ensemble.py and infer_cnn_ensemble.py, import the same
definitions instead of reaching into a training script.

See train_pff_bounded_gated.py's module docstring for the full rationale
(tanh-bounded logvar over a hard clip; gated bump classifier for a3's
bimodal 0-or-[5,100] distribution).
"""

import numpy as np
import tensorflow as tf

from src.core.data_utils import PFF_PARAM_BOUNDS, denormalize_pff_params

LOGVAR_MAX = 4.0   # tanh-bounded logvar in (-4, 4) -> sigma_norm in (0.135, 7.39);
                   # vs train_pff.py's hard clip at +-10 -> sigma_norm up to 148.4

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

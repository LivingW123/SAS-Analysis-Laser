"""
pff_ensemble_utils.py — combine predictions from the 5-member PFF regressor
ensemble (train_pff_ensemble_member.py) into one (mean, sigma) per parameter
that separates aleatoric from epistemic uncertainty.

Each member is architecturally identical to model_pff_v3.keras (v2's
tanh-bounded logvar + gated bump classifier, decoded via
train_pff_bounded_gated.decode_v2) and only learns aleatoric uncertainty --
"how noisy is this measurement, assuming my model is right." None of them
individually can express "I've never seen anything like this input." The
ensemble's cross-member disagreement is what stands in for that: on inputs
similar to what all 5 were trained on, their predictions should cluster
tightly; on genuinely unfamiliar (e.g. real, out-of-distribution) inputs,
each member extrapolates differently based on its own idiosyncratic learned
weights, so they spread out. That spread (variance of the 5 means) is
combined with each member's own aleatoric variance via the standard law of
total variance:

    Var[total] = E_member[Var[aleatoric]] + Var_member[E[mean]]
               = mean_over_members(sigma_i^2)  +  var_over_members(mu_i)

which is exactly the decomposition behind deep ensembles as an approximate
Bayesian posterior (Lakshminarayanan et al., 2017).
"""

import numpy as np


def decode_ensemble(
    mus: np.ndarray, sigmas: np.ndarray, p_bumps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine M ensemble members' per-parameter (mean, sigma) and bump
    probability into ensemble-level statistics.

    Parameters
    ----------
    mus, sigmas : (M, ..., 6) physical-units per-member mean/sigma (already
                  decoded via train_pff_bounded_gated.decode_v2 for each
                  member individually)
    p_bumps     : (M, ..., 1) per-member bump-presence probability

    Returns
    -------
    mean          : (..., 6) ensemble mean (mean of the M means)
    sigma_total   : (..., 6) sqrt(aleatoric_var + epistemic_var)
    sigma_epistemic : (..., 6) sqrt(epistemic_var) alone -- the part no
                      single model could have told you; this is the number
                      to watch for "is this shot unfamiliar" checks
    p_bump_mean   : (..., 1) ensemble mean bump probability
    p_bump_std    : (..., 1) cross-member std of bump probability -- its own
                    epistemic signal for the discrete bump/no-bump call
    """
    mean = mus.mean(axis=0)
    aleatoric_var = (sigmas ** 2).mean(axis=0)
    epistemic_var = mus.var(axis=0)
    sigma_total = np.sqrt(aleatoric_var + epistemic_var)
    sigma_epistemic = np.sqrt(epistemic_var)
    p_bump_mean = p_bumps.mean(axis=0)
    p_bump_std = p_bumps.std(axis=0)
    return mean, sigma_total, sigma_epistemic, p_bump_mean, p_bump_std

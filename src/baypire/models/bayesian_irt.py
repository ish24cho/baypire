from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOCAL_CACHE = Path(__file__).resolve().parents[3] / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={LOCAL_CACHE / 'pytensor'}")


@dataclass
class BayesianIRTPrediction:
    mean_probabilities: np.ndarray
    posterior_probability_draws: np.ndarray | None
    posterior_prompt_scores: np.ndarray | None
    posterior_predictive_prompt_scores: np.ndarray | None
    diagnostics: dict


def fit_bayesian_irt(
    experiment,
    mask,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 42,
    target_accept: float = 0.97,
    sigma_intercept: float = 5.0,
    tau: float = 1.0,
    max_posterior_draws: int = 2000,
) -> BayesianIRTPrediction:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    Y = experiment.Y
    rows, cols = np.where(mask.observed)
    y_obs = Y[rows, cols].astype("int8")
    if y_obs.size == 0:
        mu_hat = np.full_like(Y, 0.5, dtype=float)
        return BayesianIRTPrediction(mu_hat, None, None, None, {"converged": False, "reason": "no observed cells"})

    I, J = experiment.shape
    coords = {"prompt": np.arange(I), "item": np.arange(J), "obs_id": np.arange(len(y_obs))}
    with pm.Model(coords=coords) as model:
        prompt_idx = pm.Data("prompt_idx", rows.astype(int), dims="obs_id")
        item_idx = pm.Data("item_idx", cols.astype(int), dims="obs_id")
        alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_intercept)
        sigma_a = pm.HalfNormal("sigma_a", sigma=tau)
        sigma_b = pm.HalfNormal("sigma_b", sigma=tau)
        a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, dims="prompt")
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, dims="item")
        a = pm.Deterministic("a", sigma_a * (a_raw - pt.mean(a_raw)), dims="prompt")
        b = pm.Deterministic("b", sigma_b * (b_raw - pt.mean(b_raw)), dims="item")
        eta = alpha + a[prompt_idx] - b[item_idx]
        pm.Bernoulli("y_like", logit_p=eta, observed=y_obs, dims="obs_id")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            return_inferencedata=True,
        )

    alpha_draws = idata.posterior["alpha"].values.reshape(-1)
    a_draws = idata.posterior["a"].values.reshape(-1, I)
    b_draws = idata.posterior["b"].values.reshape(-1, J)
    n_draws = len(alpha_draws)
    if max_posterior_draws is not None and n_draws > max_posterior_draws:
        rng = np.random.default_rng(seed + 99)
        keep = np.sort(rng.choice(n_draws, size=max_posterior_draws, replace=False))
        alpha_draws = alpha_draws[keep]
        a_draws = a_draws[keep]
        b_draws = b_draws[keep]

    prob_draws = _sigmoid(alpha_draws[:, None, None] + a_draws[:, :, None] - b_draws[:, None, :])
    mu_hat = prob_draws.mean(axis=0)
    posterior_prompt_scores = _posterior_prompt_scores(experiment.Y, mask.observed, prob_draws)
    posterior_predictive_prompt_scores = _posterior_predictive_prompt_scores(
        experiment.Y,
        mask.observed,
        prob_draws,
        seed=seed + 199,
    )

    summary = az.summary(idata, var_names=["alpha", "sigma_a", "sigma_b", "a", "b"], hdi_prob=0.95)
    max_rhat = _finite_stat(summary.get("r_hat"), np.nanmax)
    min_ess = _finite_stat(summary.get("ess_bulk"), np.nanmin)
    divergences = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else 0
    scale = az.summary(idata, var_names=["alpha", "sigma_a", "sigma_b"], hdi_prob=0.95)
    diagnostics = {
        "model": "bayesian_additive_irt",
        "converged": bool(np.isfinite(max_rhat) and max_rhat < 1.05 and divergences == 0),
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": divergences,
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "target_accept": float(target_accept),
        "alpha_mean": float(scale.loc["alpha", "mean"]),
        "sigma_a_mean": float(scale.loc["sigma_a", "mean"]),
        "sigma_b_mean": float(scale.loc["sigma_b", "mean"]),
        "n_obs": int(y_obs.size),
    }
    return BayesianIRTPrediction(
        mu_hat,
        prob_draws,
        posterior_prompt_scores,
        posterior_predictive_prompt_scores,
        diagnostics,
    )


def _posterior_prompt_scores(Y, observed, prob_draws):
    reconstructed = prob_draws.copy()
    reconstructed[:, observed] = Y[observed]
    return reconstructed.mean(axis=2)


def _posterior_predictive_prompt_scores(Y, observed, prob_draws, *, seed: int):
    rng = np.random.default_rng(seed)
    reconstructed = (rng.random(prob_draws.shape) < prob_draws).astype(float)
    reconstructed[:, observed] = Y[observed]
    return reconstructed.mean(axis=2)


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _finite_stat(series, reducer) -> float:
    if series is None:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(reducer(arr))

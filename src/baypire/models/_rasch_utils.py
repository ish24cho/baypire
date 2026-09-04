from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return float(np.log(p / (1.0 - p)))


def fit_additive_logistic(
    Y,
    observed,
    *,
    prompt_penalty: float,
    item_penalty: float,
    intercept_penalty: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> tuple[np.ndarray, dict]:
    Y = np.asarray(Y, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    rows, cols = np.where(observed)
    y = Y[rows, cols].astype(float)
    I, J = Y.shape

    if y.size == 0:
        return np.full_like(Y, 0.5, dtype=float), {"converged": False, "reason": "no observed cells"}

    n_params = 1 + I + J
    theta0 = np.zeros(n_params, dtype=float)
    theta0[0] = logit((y.sum() + 0.5) / (len(y) + 1.0))

    def unpack(theta):
        return theta[0], theta[1 : 1 + I], theta[1 + I :]

    def objective(theta):
        alpha, a, b = unpack(theta)
        z = alpha + a[rows] - b[cols]
        loss = np.sum(np.logaddexp(0.0, z) - y * z)
        loss += 0.5 * prompt_penalty * float(np.dot(a, a))
        loss += 0.5 * item_penalty * float(np.dot(b, b))
        loss += 0.5 * intercept_penalty * float(alpha * alpha)

        p = sigmoid(z)
        residual = p - y
        grad = np.zeros(n_params, dtype=float)
        grad[0] = residual.sum() + intercept_penalty * alpha
        grad[1 : 1 + I] = np.bincount(rows, weights=residual, minlength=I) + prompt_penalty * a
        grad[1 + I :] = np.bincount(cols, weights=-residual, minlength=J) + item_penalty * b
        return float(loss), grad

    opt = minimize(
        fun=lambda theta: objective(theta),
        x0=theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": float(tol), "gtol": float(tol), "maxls": 50},
    )

    alpha_raw, a_raw, b_raw = unpack(np.asarray(opt.x, dtype=float))
    a_mean = float(np.mean(a_raw))
    b_mean = float(np.mean(b_raw))
    alpha = float(alpha_raw + a_mean - b_mean)
    a = a_raw - a_mean
    b = b_raw - b_mean
    mu_hat = sigmoid(alpha + a[:, None] - b[None, :])
    loss, grad = objective(np.asarray(opt.x, dtype=float))

    info = {
        "converged": bool(opt.success and np.isfinite(loss)),
        "reason": str(opt.message),
        "iterations": int(opt.nit),
        "final_loss": float(loss),
        "gradient_norm": float(np.linalg.norm(grad)),
        "alpha": alpha,
        "prompt_effect_sd": float(np.std(a, ddof=1)) if I > 1 else 0.0,
        "item_difficulty_sd": float(np.std(b, ddof=1)) if J > 1 else 0.0,
        "n_params": int(n_params),
        "n_obs": int(y.size),
    }
    return np.clip(mu_hat, 1e-8, 1.0 - 1e-8), info


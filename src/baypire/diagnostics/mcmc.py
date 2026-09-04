from __future__ import annotations

import numpy as np


def summarize_mcmc(idata) -> dict:
    if idata is None:
        return {"max_rhat": np.nan, "min_ess_bulk": np.nan, "divergences": np.nan}

    import arviz as az

    summary = az.summary(idata, hdi_prob=0.95)
    rhat = np.asarray(summary["r_hat"], dtype=float) if "r_hat" in summary else np.asarray([])
    ess = np.asarray(summary["ess_bulk"], dtype=float) if "ess_bulk" in summary else np.asarray([])
    rhat = rhat[np.isfinite(rhat)]
    ess = ess[np.isfinite(ess)]
    divergences = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else 0
    return {
        "max_rhat": float(np.max(rhat)) if rhat.size else np.nan,
        "min_ess_bulk": float(np.min(ess)) if ess.size else np.nan,
        "divergences": divergences,
    }


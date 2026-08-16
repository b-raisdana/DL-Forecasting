# mypy: ignore-errors
"""LightGBM tabular specialist + deep-model/GBM calibrated ensemble, per Tier-1_000.hand-made.jsonc's
dependency_modeling.auxiliary_model / ensemble.model_combination.

Trained outside the TensorFlow graph ("integration.external_ensemble.inside_tensorflow_graph": false)
on `auxiliary_features` — the same flattened last-candle-per-branch snapshot the deep model's MLP head
consumes. Predicts quantiles (q10/q50/q90) of `mfe`: the jsonc names the outputs but not which target
they quantile — `mfe` is the natural choice since it's the one continuous target both the deep model's
mean/std head and a GBM point-ish estimate can meaningfully compete/blend on (documented choice, not a
spec value).

"validation_optimized_scalar" ensemble weighting isn't spelled out mechanically anywhere in the docs
(confirmed against 03-Model & Architecture Engineering.md's late_ensemble() pseudocode, which combines
whole Keras sub-architectures, not a Keras-model + LightGBM pair) — implemented here as the simplest
reading: a single scalar alpha blending the deep model's mfe_mean against the GBM's q50, fit by
minimizing validation MSE against realized mfe.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

QUANTILES: dict[str, float] = {"q10": 0.10, "q50": 0.50, "q90": 0.90}


@dataclass
class GBMEnsemble:
    boosters: dict[str, lgb.LGBMRegressor]
    blend_alpha: float  # final_mfe = alpha * deep_mfe_mean + (1 - alpha) * gbm_q50

    def predict_quantiles(self, auxiliary_features: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[np.float64]]:
        return {name: booster.predict(auxiliary_features) for name, booster in self.boosters.items()}

    def blend(
        self, deep_mfe_mean: npt.NDArray[np.float64], gbm_q50: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return self.blend_alpha * deep_mfe_mean + (1.0 - self.blend_alpha) * gbm_q50


def train_gbm_quantile_model(
    auxiliary_features: npt.NDArray[np.float32], mfe: npt.NDArray[np.float32], n_estimators: int = 200
) -> dict[str, lgb.LGBMRegressor]:
    boosters: dict[str, lgb.LGBMRegressor] = {}
    for name, alpha in QUANTILES.items():
        booster = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=n_estimators,
            num_leaves=31,
            min_child_samples=20,
            verbosity=-1,
        )
        booster.fit(auxiliary_features, mfe)
        boosters[name] = booster
    return boosters


def fit_blend_alpha(
    deep_mfe_mean: npt.NDArray[np.float64], gbm_q50: npt.NDArray[np.float64], true_mfe: npt.NDArray[np.float64]
) -> float:
    """Closed-form scalar-weighted-least-squares fit: minimize ||alpha*d + (1-alpha)*g - y||^2 over
    alpha, i.e. fit y - g = alpha*(d - g) by OLS through the origin."""
    diff = deep_mfe_mean - gbm_q50
    denom = float(np.dot(diff, diff))
    if denom < 1e-12:
        return 0.5  # deep and GBM predictions are ~identical on this validation set — blend is moot
    alpha = float(np.dot(true_mfe - gbm_q50, diff) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def build_gbm_ensemble(
    train_auxiliary_features: npt.NDArray[np.float32],
    train_mfe: npt.NDArray[np.float32],
    val_auxiliary_features: npt.NDArray[np.float32],
    val_mfe: npt.NDArray[np.float32],
    val_deep_mfe_mean: npt.NDArray[np.float64],
) -> GBMEnsemble:
    boosters = train_gbm_quantile_model(train_auxiliary_features, train_mfe)
    val_q50 = boosters["q50"].predict(val_auxiliary_features)
    alpha = fit_blend_alpha(val_deep_mfe_mean, val_q50, val_mfe)
    return GBMEnsemble(boosters=boosters, blend_alpha=alpha)

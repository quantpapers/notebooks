# Equal risk contributions with implied volatility and volatility targeting

For a portfolio of global equity indices with implied volatility (IV) available
for each index, build a forward covariance estimate, solve for equal risk
contributions (ERC), and scale the entire equity allocation to a chosen
portfolio volatility target.

ERC determines the relative mix of indices. Volatility targeting determines total
equity exposure. Both calculations use the same covariance forecast so that
exposure scaling preserves the estimated equal risk shares.

The motivating case is a portfolio with roughly 8% realized volatility in normal
years and 14–18% in crisis years, with little difference between ordinary and
inverse-volatility weighting. Those are the user's observations, not measurements
made here. The 8% target below is an illustration, not a selected parameter.

## Forward covariance from per-index IV

Let the supplied annualized volatility estimates be \(\widehat\sigma_{i,t}\).
They must refer to a common forecast horizon and the exposures actually held.
Form

\[
D_t=\operatorname{diag}(\widehat\sigma_{1,t},\ldots,\widehat\sigma_{N,t}),
\qquad
\widehat\Sigma_t=D_t\widehat C_tD_t.
\]

Estimate correlation from an aligned trailing panel of simple total returns:

1. Demean each asset's returns and standardize to unit population variance.
2. Fit Ledoit–Wolf to the standardized panel.
3. Normalize the fitted matrix to correlation and replace the historical
   volatilities with the supplied forward estimates.

The resulting correlation estimate is, up to floating-point normalization,

\[
\widehat C_t=(1-\delta_t)C_{\text{sample},t}+\delta_t I.
\]

Ledoit–Wolf selects the shrinkage intensity from the data. Applying it to
standardized returns prevents the covariance target from pulling high- and
low-volatility indices toward a common variance. This is a practical correlation
shrinkage variant; it does not inherit a claim that the rescaled forward
covariance is universally optimal. [Ledoit–Wolf documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.covariance.LedoitWolf.html)

The supplied volatilities are already annualized, so the forward covariance is
already annualized. Do **not** multiply it by a trading-day annualization factor.

Historical correlations remain a modeling assumption. Shrinkage reduces sample
noise; it does not forecast a sudden correlation increase. Individual-index IV
does not supply the missing cross-index correlations.

## Raw IV versus a realized-volatility forecast

Using raw IV produces an implied-risk model. An 8% target under that model is
not necessarily an 8% realized-volatility target. Option prices contain risk
premia; for example, SPX implied volatility tends to trade above subsequent
realized volatility. [Cboe](https://www.cboe.com/tradable-products/vix)

For a realized-risk objective, supply calibrated forecasts instead:

\[
\widehat v_{i,t}
=f_{i,t}\!\left(\mathrm{IV}_{i,t}^{\,2}\right),
\qquad
\widehat\sigma_{i,t}=\sqrt{\widehat v_{i,t}}.
\]

Here \(f_{i,t}\) is a caller-selected positive variance forecasting model, not a
fixed haircut or a model fitted by this recipe. Calibrate it against subsequent
realized variance over the IV horizon. Use only observations whose entire forward
realization period has finished by the fitting date. Compare calibration choices
out of sample; do not fit on future crisis outcomes or full-sample averages.
Calibration does not guarantee future realized risk, and marginal variance
calibration does not fix a misspecified correlation forecast.

Use a common IV maturity and consistent annualization units. If IV is quoted in
percentage points, convert it to decimal volatility before calling the recipe.
For unhedged foreign indices, local-currency index IV is not automatically the
volatility of a base-currency holding: incorporate FX risk and its covariance,
or supply forecasts for the actual hedged/base-currency exposures. Ensure IV
timestamps were available before execution.

## Full ERC and the exposure overlay

For fully invested equity weights \(u_t>0\), with \(\sum_i u_{i,t}=1\), solve

\[
\frac{u_{i,t}(\widehat\Sigma_tu_t)_i}
     {u_t^\top\widehat\Sigma_tu_t}
=\frac{1}{N}.
\]

These are variance-contribution shares; they equal each asset's fraction of
portfolio volatility contribution. The implementation solves the positive
risk-budgeting problem using a quadratic term and a logarithmic barrier, working
in correlation units for conditioning. It does not invert covariance or replace
ERC with inverse-volatility weights. [Risk-budgeting formulation](https://arxiv.org/pdf/1311.4057)

Forecast the fully invested equity basket's volatility:

\[
\widehat\sigma_{p,t}=\sqrt{u_t^\top\widehat\Sigma_tu_t}.
\]

For a chosen positive annual target \(\sigma^*\) and an explicit positive finite
maximum exposure \(a_{\max}\), use

\[
a_t=\min\!\left(a_{\max},\frac{\sigma^*}{\widehat\sigma_{p,t}}\right),
\qquad
w_t=a_tu_t,\qquad
w_{\mathrm{cash},t}=1-a_t.
\]

An exposure cap of 1 means no leverage. A negative cash weight represents
borrowing or equivalent financed exposure. Since the equity weights are positive,
their gross exposure equals \(a_t\).

For any positive common multiplier, the \(a_t^2\) terms cancel:

\[
\frac{w_i(\widehat\Sigma_tw)_i}{w^\top\widehat\Sigma_tw}
=\frac{u_i(\widehat\Sigma_tu)_i}{u^\top\widehat\Sigma_tu}.
\]

Thus the overlay preserves ERC under the model. Forecast volatility after scaling
is \(a_t\widehat\sigma_{p,t}\), which equals the target unless the exposure cap
binds. Cash/financing is assumed to have negligible volatility. If the reserve
asset has material price or FX risk, its variance and covariances must enter the
portfolio calculation instead.

For the illustrative 8% target and a no-leverage policy:

| Fully invested basket's current annual volatility forecast | Equity exposure | Cash | Scaled forecast volatility |
|---|---:|---:|---:|
| 8% | 100% | 0% | 8% |
| 14% | 57.1% | 42.9% | 8% |
| 18% | 44.4% | 55.6% | 8% |

These are current forecasts, not the crisis year's eventual realized volatility.
When the unscaled forecast is below 8%, a no-leverage portfolio stays fully
invested and its forecast stays below the target.

If every index's forward volatility doubles while correlations remain unchanged,
covariance quadruples and ERC weights remain unchanged. The uncapped exposure
multiplier halves. If only one index's forecast rises, both the ERC allocation
and portfolio exposure can change. Risk parity across equity indices cannot
eliminate their common equity-market exposure.

## Complete Python recipe

The runnable source is [risk_parity_recipe.py](risk_parity_recipe.py). It uses the
installed NumPy, pandas, SciPy, and scikit-learn dependencies. The implementation
below is copied from that source; the source also contains the executable
analytical self-check.

~~~python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns):
    """Return variance-preserving shrunk covariance and fitted intensity."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.shape[0] < 2 or returns.shape[1] == 0:
        raise ValueError("Need at least two observations and one asset")
    if (not returns.columns.is_unique or not returns.index.is_unique
            or not returns.index.is_monotonic_increasing):
        raise ValueError("Asset labels must be unique; dates unique and sorted")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Use a complete, finite, aligned return panel")
    centered = values - values.mean(axis=0)
    volatility = centered.std(axis=0, ddof=0)
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Every asset must have positive finite volatility")
    fitted = LedoitWolf(assume_centered=True, store_precision=False).fit(
        centered / volatility
    )
    covariance = fitted.covariance_ * np.outer(volatility, volatility)
    return (
        pd.DataFrame(covariance, index=returns.columns, columns=returns.columns),
        float(fitted.shrinkage_),
    )


def forward_covariance(returns, annualized_vol):
    """Combine shrunk historical correlations with labeled forward vols."""
    historical, shrinkage = estimate_covariance(returns)
    if not isinstance(annualized_vol, pd.Series):
        raise TypeError("annualized_vol must be a pandas Series")
    if (not annualized_vol.index.is_unique
            or len(annualized_vol) != len(historical)
            or not historical.columns.isin(annualized_vol.index).all()):
        raise ValueError("Forward volatilities must match the return asset labels")
    volatility = annualized_vol.reindex(historical.columns).to_numpy(dtype=float)
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Forward volatilities must be positive and finite")
    values = historical.to_numpy()
    historical_vol = np.sqrt(np.diag(values))
    correlation = values / np.outer(historical_vol, historical_vol)
    covariance = correlation * np.outer(volatility, volatility)
    return pd.DataFrame(
        covariance, index=historical.index, columns=historical.columns
    ), shrinkage


def equal_risk_weights(covariance):
    """Return fully invested positive weights and variance-contribution shares."""
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if (covariance.empty or not covariance.index.is_unique
            or not covariance.index.equals(covariance.columns)):
        raise ValueError("Covariance must have matching, unique asset labels")
    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.diag(values) <= 0).any():
        raise ValueError("Covariance must be finite with positive variances")
    volatility = np.sqrt(np.diag(values))
    correlation = values / np.outer(volatility, volatility)
    if not np.allclose(correlation, correlation.T):
        raise ValueError("Covariance must be symmetric")
    correlation = (correlation + correlation.T) / 2
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance must be positive definite") from error

    def objective(log_x):
        x = np.exp(log_x)
        marginal = correlation @ x
        return 0.5 * x @ marginal - log_x.sum(), x * marginal - 1

    result = minimize(
        objective, np.zeros(len(volatility)), jac=True, method="BFGS"
    )
    if not result.success:
        raise RuntimeError(f"Risk-parity optimization failed: {result.message}")
    weights = np.exp(result.x) / volatility
    weights /= weights.sum()
    # Use the same symmetric matrix as the optimization, in original units.
    values = correlation * np.outer(volatility, volatility)
    contributions = weights * (values @ weights)
    variance = contributions.sum()
    if (not np.isfinite(weights).all() or (weights <= 0).any()
            or not np.isfinite(variance) or variance <= 0):
        raise RuntimeError("Optimization produced invalid weights or variance")
    return pd.DataFrame(
        {"weight": weights, "risk_share": contributions / variance,
         "target_risk_share": 1 / len(weights)},
        index=covariance.index,
    )


def volatility_target(annualized_covariance, target_vol, max_exposure):
    """Return scaled ERC allocation and annual risk / financing summary."""
    target_vol, max_exposure = float(target_vol), float(max_exposure)
    if (not np.isfinite(target_vol) or target_vol <= 0
            or not np.isfinite(max_exposure) or max_exposure <= 0):
        raise ValueError("Target volatility and maximum exposure must be positive and finite")
    allocation = equal_risk_weights(annualized_covariance).rename(
        columns={"weight": "erc_weight"}
    )
    weights = allocation["erc_weight"].to_numpy()
    variance = weights @ annualized_covariance.to_numpy() @ weights
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Forecast portfolio variance must be positive and finite")
    unscaled_vol = float(np.sqrt(variance))
    exposure = min(max_exposure, target_vol / unscaled_vol)
    allocation["weight"] = allocation["erc_weight"] * exposure
    return allocation, {
        "target_vol": target_vol,
        "unscaled_forecast_vol": unscaled_vol,
        "exposure": exposure,
        "cash_weight": 1 - exposure,
        "forecast_vol": exposure * unscaled_vol,
    }
~~~

The covariance estimator uses population moments to match scikit-learn. Two
observations are the mathematical minimum for nonzero dispersion, not a
sufficient-history recommendation. Solver and numerical-comparison settings use
the installed libraries' defaults; inspect the achieved risk shares. Invalid,
singular/indefinite covariance and optimizer failures raise errors rather than
triggering an undocumented fallback.

## Use with your data

Run from the repository root. Supply:

- A sorted, uniquely dated price panel, adjusted for splits and distributions,
  in the portfolio's common currency and on a consistent observation calendar.
- A selected correlation lookback, with a complete finite return panel.
- A dated forward-volatility panel with one column per index, already converted
  to annualized decimal units. Use calibrated values for a realized-risk target,
  or raw IV when intentionally targeting implied risk.

The code does not calibrate IV, infer stale quotes, align forecast horizons, or
choose policy parameters. Those choices need the caller's data and convention.
The 8% target and cap of 1 in this example are explicitly illustrative.

~~~python
from docs.research.risk_parity_recipe import forward_covariance, volatility_target

# total_return_prices and forward_vols are caller-supplied DataFrames.
# signal_date and correlation_lookback are caller-selected.
history = total_return_prices.loc[:signal_date].tail(correlation_lookback + 1)
if len(history) != correlation_lookback + 1:
    raise ValueError("Insufficient history for the chosen correlation lookback")
returns = history.pct_change(fill_method=None).iloc[1:]
annualized_vol = forward_vols.loc[signal_date]

annual_covariance, shrinkage = forward_covariance(returns, annualized_vol)
allocation, summary = volatility_target(
    annual_covariance,
    target_vol=0.08,    # Illustrative target from the discussion.
    max_exposure=1.0,  # Illustrative no-leverage policy.
)

print("Correlation shrinkage:", shrinkage)
print(allocation)
print(summary)

# allocation["erc_weight"]: fully invested equity mix, sums to 1.
# allocation["weight"]: final equity weights relative to portfolio NAV.
# allocation["risk_share"]: each equity's share of forecast risk.
# summary["cash_weight"]: residual cash; negative means borrowing.
# summary["forecast_vol"]: annual forecast after applying the exposure cap.
~~~

Reject missing returns rather than forward-filling them or replacing them with
zero. Inspect corporate actions and quote errors before estimation. Avoid
pairwise covariance estimated on inconsistent sets of dates.

The function requires exact asset-label agreement but accepts different ordering
in the volatility Series; it reindexes by asset name. Label matching alone
cannot prove the economic exposures, timestamps, currencies, or IV horizons
match.

## Execution and evaluation

Form the forecast from information available at the signal timestamp, then trade
at a later eligible execution. In Apollo3, use the configured positive business-
day execution lag; a signal never executes on the close that produced it.

Recalculate forecast risk for the **unscaled current equity mix**. Estimating it
from the historically volatility-targeted portfolio's returns would mix changing
weights and exposure multipliers into the underlying equity risk estimate.

When ERC and exposure are updated together from the same covariance, equal risk
shares are preserved at that sizing point. If the mix is held fixed while a newer
covariance updates exposure, the overlay still targets that mix's volatility, but
its risk contributions need not remain equal. The function here recomputes both.

Select lookback, calibration history, and rebalance cadence using walk-forward
evidence rather than treating a conventional window as a universal default.
Evaluate forward forecast accuracy, risk-share stability, turnover, net returns,
drawdowns, and realized risk after transaction costs, cash interest, and financing.
A forecast target is not a guarantee of realized volatility or drawdown: jumps,
correlation changes, execution lag, and rapid rebounds remain relevant.
Volatility scaling can reduce exposure after a fall and miss part of a rebound.
[Volatility-targeting example and timing tradeoffs](https://www.man.com/insights/volatility-is-back-better-to-target-returns-or-target-risk)

## Runnable verification

~~~bash
./venv/bin/python docs/research/risk_parity_recipe.py
~~~

The single self-check verifies the earlier three-asset ERC example, the
inverse-volatility special case for diagonal covariance, historical variance
preservation, replacement with annualized forward variances, label reordering,
attainment of an uncapped volatility target, preservation of risk shares, halving
of exposure when all IV doubles, binding no-leverage limits, borrowing/cash
accounting, and rejection of invalid inputs.

The synthetic 8%, 14%, and 18% index volatilities used in that check are arithmetic
fixtures based on the discussion, not forecasts or portfolio defaults. Its
nonbinding exposure cap is derived from the exposure required by the fixture.


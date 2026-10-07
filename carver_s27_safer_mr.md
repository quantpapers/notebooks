# Strategy twenty-seven: Safer fast mean reversion

The pure mean reversion strategy (Strategy 26) has viciously negative skew and suffers during high volatility. Strategy 27 introduces two safety mechanisms: a trend filter and a volatility multiplier.

> **Strategy twenty-seven:** Trade a mean reverting forecast which does not oppose a trend forecast, and reduces positions when volatility is high.

## 1. The Trend Filter Overlay

We only trade mean reversion when it is in the same direction as the prevailing trend. This effectively turns the strategy into "buying the dip" in an uptrend or "selling the rally" in a downtrend.

If $\text{sign}(\text{Forecast}_{MR}) \ne \text{sign}(\text{Forecast}_{Trend})$, then $\text{Forecast}_{adj} = 0$.

Carver uses **EWMAC(16, 64)** as the trend filter (EWMAC16).

## 2. The Volatility Multiplier

As in Strategy 13, we scale down the forecast when relative volatility ($V$) is high.

$$V_{i,t} = \frac{\sigma_{\%i,t}}{\text{Average}(\sigma_{\%i,t-2560} \dots \sigma_{\%i,t})}$$

$$M_{i,t} = 2 - 1.5 \times Q_{i,t}$$

Where $Q_{i,t}$ is the historical quantile of the current relative volatility.

## 3. Forecast Scaling and Limit Orders

The combined forecast is scaled by a scalar of approximately **20** (doubled from Strategy 26 because the strategy is "off" half the time).

The limit order prices $p$ are adjusted for the volatility multiplier $M$:

$$p = \text{Equilibrium}_t - \left[ N_t \times \frac{\text{NotionalExposure}_{avg}}{M_{i,t} \times \text{ForecastScalar} \times \text{AveragePosition}} \right]$$

## Performance of safer mean reversion

**Table 132: Median instrument performance since 2013 across financial asset classes (Safer fast MR)**

| | Equity | Vol | FX | Bond |
| :--- | :--- | :--- | :--- | :--- |
| Mean annual return | 9.8% | 16.3% | 10.0% | 18.3% |
| Sharpe ratio | 0.43 | 0.60 | 0.54 | 1.00 |
| Skew | −0.77 | −2.41 | −0.27 | −0.11 |

**Table 133: Median instrument performance since 2013 across commodity asset classes (Safer fast MR)**

| | Metals | Energy | Ags | Median |
| :--- | :--- | :--- | :--- | :--- |
| Mean annual return | 8.7% | 9.2% | 8.5% | 10.2% |
| Sharpe ratio | 0.43 | 0.47 | 0.46 | 0.54 |
| Skew | −0.74 | −0.65 | −0.94 | −0.57 |

**Table 134: Performance of aggregate Jumbo portfolio since 2013: original vs safer mean reversion**

| | Multiple trend | Carry | Fast MR (S26) | Safer Fast MR (S27) |
| :--- | :--- | :--- | :--- | :--- |
| Mean annual return | 13.8% | 8.7% | 17.6% | 31.6% |
| Standard deviation | 17.5% | 15.3% | 22.0% | 14.8% |
| Sharpe ratio | 0.79 | 0.56 | 0.80 | **2.14** |
| Skew (monthly) | 0.20 | −0.31 | −1.46 | 1.36 |

## Strategy twenty-seven: Trading plan

All other stages are identical to strategy twenty-six.

---
**Footnotes:**
*   The trend filter (EWMAC16) prevents "catching a falling knife" and effectively acts as a dynamic stop-loss.
*   The volatility multiplier $M$ protects against "the wrong kind of volatility" (regime shifts).
*   The resulting strategy achieves a backtested Sharpe ratio over 2.0 (aggregate portfolio), though Carver cautions against trusting such high figures blindly.
*   Safer mean reversion has a positive monthly skew due to the diversification of negative skew events across the Jumbo portfolio.
*   Requires a fully automated trading system and significant capital for full diversification.
*   Correlation with trend remains low (~0.43), making it an excellent portfolio addition.

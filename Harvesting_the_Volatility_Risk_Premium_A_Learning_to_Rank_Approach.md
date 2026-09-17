# Harvesting the Volatility Risk Premium: A Learning-to-Rank Approach

## Implementation summary

**Author:** Maciej Wysocki, Department of Quantitative Finance and Machine Learning, Faculty of Economic Sciences, University of Warsaw, Quantitative Finance Research Group. **Version/date:** arXiv:2608.24786v1 [q-fin.CP], 25 August 2026. **Source:** supplied 68-page PDF, `tmp/Harvesting the Volatility Risk Premium-A LTR approach.pdf`. This is an implementation-oriented extraction of that version, not an independently reproduced backtest.

The paper applies LightGBM LambdaRank to a daily cross-section of eight delta-targeted short SPXW puts plus SKIP. A path-aware Sortino-on-minute-bars label rewards terminal profit while penalizing intraday downside. A calibrated confidence gate and seven margin-aware sizing methods turn scores into positions. The headline Edge Allocation method reports Sharpe **3.1048 in 2021–2024 walk-forward** and **5.7612 in the 2025 hold-out**, versus CBOE PUT **0.6795 / 0.1754** and the strongest internal baseline, Rolling-Sharpe, **0.5569 / 0.6128**. These Sharpes use geometric annualized return divided by annualized volatility; the headline arithmetic OOT Sharpe is 5.489. EA's OOT return is 10.48%, volatility 1.82%, and maximum drawdown 1.43%; its full-sample maximum drawdown is 2.28%. OOT PSR versus buy-and-hold is 0.9636, but DSR after 75 trials is 0.8563. The single hold-out year and fragile WF sensitivities limit the evidence.

| Core specification | Extracted contract |
|---|---|
| Inputs / data required | CBOE minute SPXW OHLC, bid/ask chains and PM settlement; SPX minute/closes, IV surface snapshots, VIX tenor family/VVIX/futures, put-call ratios, EFFR, jobless claims, NYSE calendar and macro release dates, SPX dividend yield for benchmark adjustment. 2017 warm-up; 2018–2025 data; WF 2021–2024; OOT 2025. |
| Outputs / signal format | Nine real-valued candidate scores per eligible day; top candidate and top-two gap; SKIP or a nonnegative integer short-put count. FOMC announcement days excluded. |
| Core equations | Nearest absolute delta; gross settlement payoff; minute downside score; frozen 10/40/60/90 percentile grades; LambdaRank gains [0,1,3,7,15]; gap gate; Reg-T margin; seven sizing equations; geometric Sharpe and daily-arithmetic PSR/DSR. All defined below. |
| Hyperparameters / defaults | Full Tables 1 and 3 reproduced; 50 TPE trials; 2000-round ceiling and 50-round patience; truncation 5; S1–S5 cutoffs/caps/stability rules; 16% sizing anchor; label epsilon $10^{-8}$. Gate grid values and realized fitted model/size parameters are not disclosed. |
| Training / validation | Expanding annual fits starting 2018–2020; last six full-window months reserved for the gate; previous six months for ranker tuning, recent three of those for early stopping. Feature selection reruns per window. Sizing calibrates on in-sample predictions. OOT gate uses the union calibration described below. |
| Rebalancing / execution | Daily 10:00 ET selection, mid entry, PM-expiry cash settlement; shortest listed expiry means some early-sample positions span one session. Annual model/calibration refresh; no 2025 refitting. Start NAV \$5M and chain windows. |
| Verification targets | Complete source Tables 1–10, B.1, C.1, D.1, D.2, E.1; headline and internal/external comparisons; feature-survival counts; terminal-NAV and drawdown scalar targets. |

### Known extraction gaps

- Source Figure 1 plots 2018–2025 benchmark growth and VIX; Figures 2–3 plot 2021–2025 daily NAV and drawdowns. Exact daily coordinates are unavailable in text; scalar endpoints and extrema reported in prose are retained. Consult those original figures for the full paths.
- Source Figure 4's regime bars are covered numerically by the complete regime tables below, including values clipped by the plot axis. No independent numerical gap remains for those bars.
- Source Figure 5 plots all sizing methods' feature-group ablation deltas. EA's complete table is recovered; exact untabulated non-EA bars remain figure-only.
- Source Figure F.1 shows top-60 feature survival across windows. All 27 always-selected feature identities and aggregate survival counts are recovered; the remaining per-window cells and complete fitted feature manifests are not reconstructed from its heatmap.

**Reproduction boundary:** all disclosed implementation details are retained, but the source does not fully define the internal baselines, gate grid, several estimators, source data cleaning, and early overnight accounting. Formula/prose conflicts and missing choices are identified where they matter. They must be settled before claiming exact replication; no undocumented defaults are supplied.

## Contents

- [Universe and ranking target](#universe-and-ranking-target)
- [Feature selection and ranker training](#feature-selection-and-ranker-training)
- [Confidence gate, schedule, and simulation](#confidence-gate-schedule-and-simulation)
- [Position sizing, margin, and fees](#position-sizing-margin-and-fees)
- [Data requirements and benchmark construction](#data-requirements-and-benchmark-construction)
- [Performance metrics and significance tests](#performance-metrics-and-significance-tests)
- [Empirical verification targets](#empirical-verification-targets)
- [Robustness](#robustness-structural-sensitivity-ablations-and-execution-drag)
- [Mechanism decomposition and calibration targets](#mechanism-decomposition-and-calibration-targets)
- [Complete feature catalog and availability](#complete-feature-catalog-and-availability)
- [Regime-conditional verification tables](#regime-conditional-verification-tables)
- [References](#references)

## Universe and ranking target

At 10:00 ET each eligible trading day, resolve the shortest available SPXW expiration and every listed strike for it. Exclude FOMC announcement days upstream. Form eight short-put candidates with absolute delta targets

$$
\Delta^{\mathrm{tgt}}\in\{0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.45\},
\qquad K^*=\arg\min_K\left|\left|\Delta^{\mathrm{BS}}(K)\right|-\Delta^{\mathrm{tgt}}\right|.
$$

Recover implied volatility from the 10:00 option mid quote. Use Black–Scholes, EFFR as the risk-free rate, no dividends, and NYSE-calendar trading time to expiration. Greeks use that same model. Strike ties, duplicate contracts across buckets, unavailable quotes, and the exact numerical IV solver are unspecified.

The ninth candidate is synthetic SKIP. Its **realized training score and gross P&L** are zero. It still receives a learned prediction: forcing its *inference* score to zero would conflict with the paper's statement that the model learns its rank. Non-applicable per-strategy features are missing for SKIP, except for the explicit zero ROM and rank-8.5 conventions in the feature catalog.

Contracts are held to official PM cash settlement. With contract count $Q$, entry premium $p_{\mathrm{entry}}$ in index points per share, strike $K$, and settlement index $S_{\mathrm{settle}}$,

$$
\mathrm{GrossPnL}_{\$}=100Q\left[p_{\mathrm{entry}}-\max(K-S_{\mathrm{settle}},0)\right].
$$

The historical resolver yields same-session expiry on 62.0% of days in 2018–2021, 87.2% in 2022, and all days from 2023 onward. The remaining observations use one-session expiry. Tuesday expirations began 18 April 2022 and Thursday expirations 11 May 2022, according to the source. **Source conflict:** the feature catalog allows a one- or two-day fallback, whereas the universe discussion says no sampled tenor exceeds one session. Overnight capital overlap, bar coverage, and P&L attribution for the early one-session positions are not fully specified by the otherwise daily event loop.

### Path-aware score

Construct a one-minute mark sequence from entry through settlement:

$$
m_0=p_{\mathrm{entry}},\quad m_i=a_i\ (1\le i<N),\quad
m_N=\max(K-S_{\mathrm{settle}},0),\qquad d_i=m_{i-1}-m_i,
$$

where $a_i$ is the ask quote (the short position's cost to close). The per-share outcome telescopes: $G_{s,t}=\sum_{i=1}^N d_i=p_{\mathrm{entry}}-m_N$. The label score is

$$
D_{s,t}=\sqrt{\sum_{i=1}^N\min(d_i,0)^2},\qquad
z_{s,t}=\frac{G_{s,t}}{D_{s,t}+10^{-8}},\qquad z_{\mathrm{SKIP},t}=0.
$$

This is a gross, path-aware trade score, not the daily portfolio Sortino ratio. Ask-marked intraday losses affect the denominator even when two trades have equal settlement profit.

For each training window, pool non-SKIP scores across training dates and compute absolute quantiles $\theta_p=Q_p(\{z_{s,t}:s\ne\mathrm{SKIP},t\in T_{\mathrm{train}}\})$, for $p\in\{0.10,0.40,0.60,0.90\}$. Freeze the thresholds; do not recompute them within each prediction-day cross-section.

$$
g_{s,t}=\begin{cases}
4&z_{s,t}>\theta_{0.90},\\
3&\theta_{0.60}<z_{s,t}\le\theta_{0.90},\\
2&\theta_{0.40}<z_{s,t}\le\theta_{0.60},\\
1&\theta_{0.10}<z_{s,t}\le\theta_{0.40},\\
0&z_{s,t}\le\theta_{0.10}.
\end{cases}
$$

Expected pooled grade proportions are 10%, 30%, 20%, 30%, 10%. SKIP's zero score falls in grade 1 in every reported training window; this is an empirical threshold result, not an instruction to hard-code its grade independently. Quantile interpolation is unspecified. The stated isolation of the gate hold-out must also be respected when defining any training-derived thresholds; the label section does not explicitly resolve that boundary.

## Feature selection and ranker training

Run selection independently at each annual training boundary, withholding the final six months for gate calibration. CS means one market-state value per day broadcast to all candidates; PS means candidate-specific. Approximately 190 candidate features enter the following pipeline.

| Stage | Exact rule |
|---|---|
| S1 | Drop a feature when its training real-candidate null rate exceeds 0.30. |
| S2 | Drop if variance is below $10^{-6}$ or modal frequency exceeds 0.99. |
| S3, CS | Compute Spearman correlation across dates between the feature and the day-mean label. |
| S3, PS | Compute within-day Spearman correlation between feature and label; take its median across training dates. |
| S3 filter | Drop if absolute relevance correlation is below 0.05. |
| S4 clustering | Build within-scope Spearman correlations and cluster at absolute correlation at least 0.85. Keep the largest absolute S3-relevance representative, breaking ties by catalog-group priority, data quality, then feature name. |
| S4 group caps | Retain at most position 8, vol_surface 4, vix 6, calendar 8, and 5 for every other group; remove lowest-relevance excess representatives. These are source-defined, untuned caps. |
| S5 stability | Rerun S1–S4 on three expanding-time folds inside the training window; retain features surviving at least two folds. |

The exact expanding-fold endpoints, clustering/linkage algorithm, group-priority order, data-quality tie metric, missing-correlation treatment, and whether S3's label is continuous or ordinal are not specified. The source reports roughly 50–60 survivors per window.

### LightGBM configuration and search (source Table 1)

One date is one query group of nine rows. Fit a GBDT with objective `lambdarank`, metric NDCG at 1 and 3, LambdaRank truncation level 5, and `label_gain = [0, 1, 3, 7, 15]`. Pairwise gradients are weighted by the NDCG change caused by swapping differently graded candidates. Only the highest predicted candidate is acted on.

| Parameter | Search space | Sampling |
|---|---|---|
| Number of leaves | 16–256 | Log |
| Learning rate | 0.01–0.10 | Log |
| Minimum samples per leaf | 20–200 | Log |
| Subsampling fraction | 0.5–1.0 | Uniform |
| Row subsampling fraction | 0.5–1.0 | Uniform |
| Bagging frequency | Integers 0–10 | Integer uniform |
| L1 regularization | $10^{-3}$–10 | Log |
| L2 regularization | $10^{-3}$–10 | Log |
| Boosting rounds | Up to 2000 | Early stopping, patience 50 |

The source names both “Subsampling fraction” and “Row subsampling fraction” without giving API keys; the likely column/row distinction is not a verified mapping. Selected per-window parameter values and random seeds are absent.

Use 50 Optuna TPE trials per window, maximizing NDCG@1. First remove the final six months of the full window for gate calibration. The last six months of the remaining ranker data form the search validation slice; each trial trains on earlier observations. Refit the winning configuration with the last three months of the ranker data reserved for early stopping. Those three months overlap the search validation slice; they are not an independent validation period. Table 1 also describes early stopping during the search on this recent three-month subset. The source does not report the realized boosting iterations or validation NDCG scores.

## Confidence gate, schedule, and simulation

For predicted candidate scores $\widehat y_{s,t}$, let $s_t^*=\arg\max_s\widehat y_{s,t}$ and let $\widehat y_{(1),t}\ge\widehat y_{(2),t}$ be the top two scores. Define

$$
c_t=\widehat y_{(1),t}-\widehat y_{(2),t},\qquad
\tau_w^*=\arg\max_{\tau\in\mathcal G_\tau}
\operatorname{Sortino}\bigl(\mathrm{NetPnL}^{\mathrm{top1}}_t\,\mathbf1\{c_t\ge\tau\}\bigr)_{t\in H_w}.
$$

Here $H_w$ is the final six months of the training window, withheld from feature selection and ranker training; this displayed optimization is a reconstruction of the stated procedure. The threshold grid $\mathcal G_\tau$, ties, and the exact capital/position convention for its NetPnL series are not given. At inference, SKIP winning or $c_t<\tau_w^*$ means $Q_t=0$ and zero P&L. Equality clears the gate, although the calibration-table note describes “exceeds.”

### Annual schedule (source Table 2)

| Window | Training years | Prediction year |
|---|---|---|
| WF 1 | 2018–2020 | 2021 |
| WF 2 | 2018–2021 | 2022 |
| WF 3 | 2018–2022 | 2023 |
| WF 4 | 2018–2023 | 2024 |
| OOT | 2018–2024 | 2025 |

Use 2017 only for rolling-feature warm-up. Repeat feature selection, 50-trial tuning, model refit, gate calibration and per-method sizing calibration at each boundary. Freeze the 2018–2024 fit for all of 2025. The held-out year is not used for model selection or hyperparameter search. **OOT gate exception:** the detailed calibration appendix says the 2025 threshold is calibrated on the union of the four WF gate hold-outs, rather than the 2018–2024 window's final six months. Preserve that specific statement; it differs from the generic annual procedure.

### Daily event loop

Start each sizing-method path with $\mathrm{NAV}_0=\$5{,}000{,}000$ and chain NAV continuously across 2021–2025.

1. Select the active annual model and score all nine candidates with information available at 10:00 ET.
2. Apply learned SKIP selection and the score-gap gate.
3. For a real pick, calculate its entry margin, desired integer size, and cap the count at $\lfloor\mathrm{NAV}_{t-1}/M_t\rfloor$.
4. Enter at $(\mathrm{bid}+\mathrm{ask})/2$ under the headline fill assumption. Hold through PM settlement, subtract entry fees, and update $\mathrm{NAV}_t=\mathrm{NAV}_{t-1}+\mathrm{NetPnL}_{\$,t}$.
5. Record zero return on abstention days. The simulated NAV chains option P&L without separately adding collateral interest. The source treats that P&L as excess return because collateral earns the risk-free rate; EFFR is subtracted from external total returns, not again from the strategy.

The paper's pseudocode calls a per-share field `NetPnL` before subtracting fees; use the explicitly defined gross settlement payoff to avoid double-counting fees. Do not reinterpret the pre-2023 one-session options as same-day liquidations: their accounting remains an unresolved source detail.

## Position sizing, margin, and fees

Each sizing method converts the confidence-gate-cleared daily top-ranked strategy into a nonnegative integer number of contracts. All seven methods have exactly one free scalar, calibrated separately for each walk-forward window. Sizing uses the selected contract's trade-day inputs and previous-day portfolio value $\mathrm{NAV}_{t-1}$; the fitted parameter and training-set statistics remain frozen during that window's out-of-sample period. The common margin cap and fee schedule apply after the desired count is computed. A skipped or gate-rejected day has zero contracts.

### Common definitions and collateral constraint

Let $P_t$ be the selected short put's entry premium in index points, $S_t$ its underlying spot, $K$ its strike, $\delta_t$ its delta, and $\Gamma_t$ its gamma. The dollar contract multiplier is 100. The paper's Reg-T per-contract margin is

$$
M_t=100\left[P_t+\max\left(0.15S_t-\max(0,S_t-K),\;0.10K\right)\right].
$$

The put's out-of-the-money amount is $\max(0,S_t-K)$. Every desired contract count below is rounded **down**, then capped by available collateral:

$$
Q_t=\min\left(Q_t^{\mathrm{des}},\left\lfloor\frac{\mathrm{NAV}_{t-1}}{M_t}\right\rfloor\right).
$$

The authors report that this cap rarely binds in the headline configuration. This is the margin schedule used by the paper, not a claim about a broker's current requirements.

The entry fee per contract, in dollars, depends on the entry premium before the 100 multiplier:

$$
c(P_t)=\begin{cases}
0.25,&P_t<0.05,\\
0.50,&0.05\le P_t<0.10,\\
0.65,&P_t\ge0.10.
\end{cases}
\qquad
F(Q_t,P_t)=\begin{cases}
0,&Q_t=0,\\
\max(1.00,Q_t c(P_t)),&Q_t>0.
\end{cases}
$$

The total fee has a \$1.00 minimum per non-empty trade. These are entry-only fees: positions are held to PM cash settlement, with no exit fee for that settlement. The minimum and zero-trade branches above make the prose fee rule explicit.

### Per-window equal-volatility calibration

For each method $M$, run the full training-window backtest at every point of its grid $\mathcal G_M$, including the common margin cap and fees. Select

$$
\theta_w^\star=\underset{\theta\in\mathcal G_M}{\arg\min}\;
\left|\sigma_{\mathrm{train},w}(\theta)-0.16\right|,
$$

where $\sigma_{\mathrm{train},w}(\theta)$ is the annualized standard deviation of the simulated daily NAV returns in training window $w$. In explicit notation, the return is $r_t(\theta)=\mathrm{NAV}_t(\theta)/\mathrm{NAV}_{t-1}(\theta)-1$ and annualization is $\sqrt{252}\,\operatorname{sd}(r_t(\theta))$. The metric appendix specifies sample standard deviation (denominator $N-1$); grid-objective tie-breaking is not specified. The absolute-value objective is confirmed by the grid-table note.

The 16% anchor reflects long-run realized SPX volatility. The 2018–2020 training window, including the March 2020 spike, has approximately 21% realized volatility; using that window-specific anchor would encourage sizing for persistently elevated volatility. The paper also tests 12% and 20% volatility anchors, plus Sharpe-maximizing and Sortino-maximizing calibration objectives.

**Calibration limitation:** sizing is calibrated on predictions for observations on which the ranker was trained, unlike the separately held-out confidence-gate calibration. Better in-sample fits can depress training volatility and increase the fitted position size out of sample. The paper identifies calibration on an inner validation slice as the consistent alternative, but does not use it for the headline procedure. The upper grid endpoint binds in 24 of 35 method/window cells. Mean realized training volatility is 0.139, with range 0.081–0.165, versus the 0.16 target. For EA, attaining the target would require $u_{\max}=1.07$–$1.58$ in four of five windows, exceeding a margin utilization fraction's upper bound of one. Consequently, methods do **not** attain a common risk budget; comparisons control for position size only partially.

### Fixed Margin Utilization (FMU)

FMU is the non-adaptive baseline and commits a fixed fraction of NAV to margin:

$$
Q_t^{\mathrm{des}}=\left\lfloor\frac{f_{\mathrm{FMU}}\mathrm{NAV}_{t-1}}{M_t}\right\rfloor,
\qquad \theta=f_{\mathrm{FMU}}\in(0,1).
$$

### Volatility Targeting (VT)

Let $\sigma_t^{\mathrm{intra},5d}$ be the five-day realized **intraday** SPX volatility, in daily rather than annualized units. Approximate dollar P&L volatility per contract by delta linearization, then target daily dollar P&L volatility:

$$
\sigma_t^{\$,c}=|\delta_t|S_t\sigma_t^{\mathrm{intra},5d}\,100,
\qquad
\sigma_t^{\$,\mathrm{tgt}}=f_{\mathrm{VT}}\frac{\sigma_{\mathrm{ann}}}{\sqrt{252}}\mathrm{NAV}_{t-1},
\qquad \sigma_{\mathrm{ann}}=0.16.
$$

$$
Q_t^{\mathrm{des}}=\left\lfloor\frac{\sigma_t^{\$,\mathrm{tgt}}}{\sigma_t^{\$,c}}\right\rfloor,
\qquad \theta=f_{\mathrm{VT}}.
$$

The free scalar is a volatility-target multiplier, not a margin fraction. **Source ambiguity:** this sizing section does not give the precise estimator or timestamp of the five-day realized intraday volatility input; a completed entry-day observation must not be assumed available for a 10:00 ET trade.

### Short-Richness Scaling (SRS)

Let $\sigma_{\mathrm{ATM},t}^{\mathrm{0DTE}}$ be annualized at-the-money-forward implied volatility of the 0DTE chain at **10:00 ET**, in decimal units. VIX9D is in index percentage points. Define richness and its bounded scale:

$$
\mathcal R_t=\frac{\sigma_{\mathrm{ATM},t}^{\mathrm{0DTE}}}{\mathrm{VIX9D}_t/100},
\qquad
\mathcal S_t=\min\left(C_{\mathrm{train}},\max\left(1,\frac{\mathcal R_t}{\operatorname{median}(\mathcal R_{\mathrm{train}})}\right)\right).
$$

$$
Q_t^{\mathrm{des}}=\left\lfloor\frac{f_{\mathrm{SRS}}\mathcal S_t\mathrm{NAV}_{t-1}}{M_t}\right\rfloor,
\qquad \theta=f_{\mathrm{SRS}}.
$$

$\mathcal S_t$ is the richness scale, distinct from spot $S_t$. The training median and $C_{\mathrm{train}}$, the training-set 95th-percentile cap, are estimated once per training window and are not tunable. The intended scale floor is one. **Source ambiguity:** the text does not explicitly state whether $C_{\mathrm{train}}$ is the 95th percentile of raw richness or median-normalized richness; preserve that unresolved choice before implementing the cap. The formula's floor of one also depends on $C_{\mathrm{train}}\ge1$.

### Edge Allocation (EA)

EA is the headline method. Define the relative edge of 0DTE ATM implied volatility over annualized five-day realized intraday volatility:

$$
\mathcal E_t=\frac{\sigma_{\mathrm{ATM},t}^{\mathrm{0DTE}}-\sigma_t^{\mathrm{intra},5d}\sqrt{252}}{\sigma_t^{\mathrm{intra},5d}\sqrt{252}}.
$$

Map the current edge to a percentile in the frozen training-window empirical edge distribution:

$$
u_t=u_{\max}\widehat F_{\mathcal E,\mathrm{train}}(\mathcal E_t),
\qquad
Q_t^{\mathrm{des}}=\left\lfloor\frac{u_t\mathrm{NAV}_{t-1}}{M_t}\right\rfloor,
\qquad\theta=u_{\max}\in(0,1).
$$

Here $u_t$ is the daily margin utilization fraction and $\widehat F_{\mathcal E,\mathrm{train}}$ is the empirical CDF fitted on training-window edges. Size therefore increases monotonically with the percentile rank of today's edge. The paper does not specify a percentile interpolation or tie convention.

### Gamma-Budgeted Sizing (GB)

Freeze the adverse fractional intraday spot move $k$ at the training-set 95th percentile of daily maximum adverse moves. The per-contract dollar scenario loss includes both linear and quadratic Greeks:

$$
L_t^c=\left(|\delta_t|kS_t+\frac12|\Gamma_t|(kS_t)^2\right)100,
\qquad
Q_t^{\mathrm{des}}=\left\lfloor\frac{b_{\mathrm{GB}}\mathrm{NAV}_{t-1}}{L_t^c}\right\rfloor,
\qquad\theta=b_{\mathrm{GB}}.
$$

$b_{\mathrm{GB}}$ is the fraction of NAV the budget permits losing under that scenario. Both terms are multiplied by 100. **Source ambiguity:** this section does not define the reference price and intraday observation interval used to calculate each day's maximum adverse move.

### Half-Kelly (HK) and Quarter-Kelly (QK)

Maintain a rolling **252-day** history of past per-day returns on margin for the strategy:

$$
r_t^c=\frac{\mathrm{NetPnL}_t\,100}{M_t}.
$$

Here $\mathrm{NetPnL}_t$ is the paper's per-share/per-index-point trade P&L quantity before multiplication by 100, not the portfolio-level dollar P&L after sizing. Let $\widehat\mu_t^r$ and $(\widehat\sigma_t^r)^2$ be the rolling mean and variance of the available **past** returns on margin; a current day's realized return must not enter its own sizing statistics. Then

$$
f_t^\star=\frac{\widehat\mu_t^r}{(\widehat\sigma_t^r)^2},
\qquad
f_t=\min\left(f_{\mathrm{CK}},\max(0,\alpha f_t^\star)\right),
\qquad
Q_t^{\mathrm{des}}=\left\lfloor\frac{f_t\mathrm{NAV}_{t-1}}{M_t}\right\rfloor.
$$

Use fixed $\alpha=0.5$ for HK and $\alpha=0.25$ for QK. Their common free parameter $\theta=f_{\mathrm{CK}}$ caps the deployed daily Kelly fraction; $\alpha$ is not tuned. **Source ambiguities:** the section does not specify rolling-history initialization, minimum observations, zero-variance handling, whether skipped dates enter as zeros, how the pre-sizing return stream is formed on non-traded dates, or whether the return-on-margin statistic incorporates the contract-count-dependent entry fee. These cannot be silently defaulted while claiming exact reproduction.

### Calibration grids (source Table 3)

Every grid includes both endpoints at the stated equal increment; every point receives a training-window backtest.

| Method | Free parameter | Inclusive grid | Increment |
|---|---|---|---|
| FMU | $f_{\mathrm{FMU}}$ | $\{0.02,0.04,\ldots,0.60\}$ | 0.02 |
| VT | $f_{\mathrm{VT}}$ | $\{0.1,0.2,\ldots,2.5\}$ | 0.1 |
| SRS | $f_{\mathrm{SRS}}$ | $\{0.02,0.04,\ldots,0.50\}$ | 0.02 |
| EA | $u_{\max}$ | $\{0.05,0.10,\ldots,0.80\}$ | 0.05 |
| GB | $b_{\mathrm{GB}}$ | $\{0.005,0.010,\ldots,0.100\}$ | 0.005 |
| HK, QK | $f_{\mathrm{CK}}$ | $\{0.05,0.10,\ldots,0.80\}$ | 0.05 |

The grids are the paper's calibration choices, not new recommended limits. HK and QK share the grid and differ only in their fixed fractional-Kelly multiplier. The paper specifies no general handling of invalid or zero denominators in VT, SRS, EA, or GB, and no insolvency rule for the common collateral expression.

## Data requirements and benchmark construction

The source uses CBOE one-minute SPXW OHLC bars and intraday bid/ask quotes, underlying SPX minute data, complete option-chain identifiers/expirations/strikes, and official PM cash settlement. Required surface observations include 09:35, 10:00, close, and intraday values for IV ranges. Other inputs are SPX/VIX option put-call volumes, the VIX spot-tenor family and VVIX, front three VIX futures settlements and expiry calendars, FRED EFFR, non-seasonally-adjusted weekly jobless claims, NYSE sessions, and historical macro release dates. Exact dataset product identifiers and quote-cleaning rules are not reported.

The daily source-data statistics cover 1 January 2018–31 December 2025. Initialize rolling features and candidate statistics using 2017; do not treat that warm-up as model-fitting or evaluation data. The evaluation period is 2021–2025. SPXW is European-style and cash-settled, with a \$100 multiplier; distinguish PM-settled SPXW from AM-settled standard monthly SPX.

External comparators are the downloaded CBOE PUT and WPUT index series and SPX buy-and-hold. PUT sells ATM monthly SPX puts on the third Friday, holds to expiry, rolls monthly, and earns matched-maturity Treasury-bill collateral yield. WPUT uses the same cash-secured construction on weekly Friday expirations/rolls. Those descriptions do not provide complete independently executable index replication rules (strike tie rules, exact fixing times and bill rebalancing are absent); the reported comparison can instead use the source benchmark series. For SPX price-index buy-and-hold, accrue the continuous dividend yield daily before the EFFR excess-return adjustment, as the headline-table note requires. The yield data source and exact accrual convention are not identified.

### Source Table 4: descriptive-statistic verification

These are **daily log returns** of closing index levels over 2018–2025, unlike the simple returns used in the strategy performance calculation. MD is maximum drawdown, VaR/CVaR are signed lower-tail return statistics, lag-1 autocorrelation is serial correlation, JB is Jarque–Bera, ADF is Augmented Dickey–Fuller with AIC-selected lag order and a constant, and LB(10) is Ljung–Box through ten lags. Stars retain source significance levels: * 0.1, ** 0.05, *** 0.01. These diagnostics are data checks, not model inputs or fresh trading claims.

| Statistic | S&P 500 | VIX | PUT | WPUT |
|---|---:|---:|---:|---:|
| Mean | 0.0005 | 0.0002 | 0.0003 | 0.0001 |
| Median | 0.0009 | −0.0073 | 0.0005 | 0.0007 |
| Daily standard deviation | 0.0124 | 0.0810 | 0.0089 | 0.0084 |
| Annual standard deviation | 0.20 | 1.29 | 0.14 | 0.13 |
| Skewness | −0.65 | 1.40 | −1.96 | −1.90 |
| Excess kurtosis | 14.77 | 8.88 | 45.04 | 31.52 |
| Minimum | −0.1277 | −0.4424 | −0.1218 | −0.1014 |
| Maximum | 0.0909 | 0.7682 | 0.0903 | 0.0836 |
| MD | −0.34 | −0.86 | −0.29 | −0.26 |
| VaR 95% | −1.83% | −10.81% | −1.16% | −1.28% |
| CVaR 95% | −3.05% | −15.23% | −2.31% | −2.34% |
| Autocorrelation (1) | −0.15 | −0.08 | −0.26 | −0.20 |
| JB | 18325*** | 7321*** | 170315*** | 83857*** |
| ADF | −14.15*** | −18.62*** | −14.04*** | −14.32*** |
| LB(10) | 231.37*** | 26.45*** | 400.51*** | 206.85*** |

The paper names, but does not give estimator/quantile conventions for, these descriptive diagnostics; exact software-level reproduction remains open. Their names are not a substitute for the strategy metric definitions that follow.

## Performance metrics and significance tests

### Return basis and annualization

Use **252 trading days per year**, including abstention days in the daily series. Let $P_i$ be closing portfolio equity on trading day $i$, $N$ the number of observations, $r_i$ the daily simple return, and $r_i^f$ that day's EFFR quoted as an annual rate in decimal form:

$$
r_i=\frac{P_i-P_{i-1}}{P_{i-1}},\qquad
\widetilde r_i=
\begin{cases}
r_i,&\text{strategy series},\\
r_i-r_i^f/252,&\text{external benchmark series}.
\end{cases}
$$

Strategy P&L per unit of cash collateral is already excess return: do not deduct EFFR again, which would turn an abstention day into a loss. The external CBOE PUT, CBOE WPUT, and SPX buy-and-hold series are total-return series and require the subtraction. Sharpe, Sortino, PSR, and DSR use excess returns. Drawdown measures and trade counts use unadjusted strategy returns.

For any daily return series $x=(x_1,\ldots,x_N)$, define:

$$
\overline x=\frac1N\sum_{i=1}^N x_i,\qquad
s_x=\sqrt{\frac1{N-1}\sum_{i=1}^N(x_i-\overline x)^2},
$$

$$
\operatorname{aRC}(x)=\left[\prod_{i=1}^N(1+x_i)\right]^{252/N}-1,
\qquad
\operatorname{aSD}(x)=\sqrt{252}\,s_x,
$$

$$
\operatorname{aDD}(x)=\sqrt{252}\sqrt{\frac1N\sum_{i=1}^N[\min(x_i,0)]^2}.
$$

Annualized downside deviation has zero target and averages over **all** $N$ days, not only negative-return days. Annualized compounded return (aRC) is geometric; annualized standard deviation (aSD) uses sample variance with denominator $N-1$.

For the equity curve through terminal day $T$, maximum drawdown is the positive peak-to-trough fraction:

$$
\operatorname{MD}=\max_{\tau\in[0,T]}
\frac{\max_{t\in[0,\tau]}P_t-P_\tau}{\max_{t\in[0,\tau]}P_t}.
$$

The reported ratios are:

$$
\operatorname{SR}=\frac{\operatorname{aRC}(\widetilde r)}{\operatorname{aSD}(\widetilde r)},\qquad
\operatorname{Sortino}=\frac{\operatorname{aRC}(\widetilde r)}{\operatorname{aDD}(\widetilde r)}.
$$

The numerator is compounded annual growth, not annualized arithmetic mean. For the headline out-of-time method, the paper reports **5.761** under this convention versus **5.489** under the arithmetic convention. PSR and DSR instead use the daily arithmetic Sharpe input:

$$
\widehat{\operatorname{SR}}=\frac{\overline{\widetilde r}}{s_{\widetilde r}}.
$$

### Probabilistic and deflated Sharpe ratios

With $\Phi$ the standard-normal CDF, $\widehat\gamma_3$ the sample skewness of daily excess returns, and $\widehat\gamma_4$ the sample **non-excess** kurtosis (Gaussian value 3):

$$
\widehat{\operatorname{PSR}}(\operatorname{SR}^*)=
\Phi\!\left(
\frac{(\widehat{\operatorname{SR}}-\operatorname{SR}^*)\sqrt{N-1}}
{\sqrt{1-\widehat\gamma_3\widehat{\operatorname{SR}}+
\frac{\widehat\gamma_4-1}{4}\widehat{\operatorname{SR}}^{\,2}}}
\right).
$$

Here $\operatorname{SR}^*$ is the named benchmark's realized **daily arithmetic** Sharpe over the same evaluation slice, with the excess-return adjustment above. The strategy and benchmark Sharpe inputs must share the daily horizon. The paper does not specify the sample skewness/kurtosis finite-sample bias correction convention.

For $N_t$ independent trials under the null, the expected maximum daily Sharpe is approximated by:

$$
M_{N_t}=\mathbb E\!\left[\max_{n=1,\ldots,N_t}\widehat{\operatorname{SR}}_n\right]
\approx\sqrt{\frac1{N-1}}\left[
(1-\gamma)\Phi^{-1}\!\left(1-\frac1{N_t}\right)
+\gamma\Phi^{-1}\!\left(1-\frac1{N_t e}\right)
\right],
$$

where $n$ indexes trials, $\gamma\approx0.5772$ is the Euler–Mascheroni constant, $e$ is the base of natural logarithms, $\Phi^{-1}$ is the standard-normal quantile function, and the null variance of the daily Sharpe estimator is $1/(N-1)$.

The printed final formula is:

$$
\widehat{\operatorname{DSR}}=
\widehat{\operatorname{PSR}}(\operatorname{SR}_{\mathrm{eff}}^*),
\qquad
\operatorname{SR}_{\mathrm{eff}}^*=\max(\operatorname{SR}^*,M_{N_t}).
$$

The empirical correction uses **$N_t=75$**: 28 reported configurations (headline, 10 structural sensitivities, 15 feature-group ablations, 2 execution-drag variants), 40 confidence-gate calibrations (8 trade-rate points in each of 5 windows), 6 non-headline sizing methods, and 1 abandoned all-options variant. The 250 hyperparameter trials and 740 size-grid evaluations are excluded because they are scored on NDCG@1 and distance from the 16% training-volatility anchor, respectively, rather than ranked by realized performance.

**Source conflict — lock before coding:** the printed DSR formula retains $\operatorname{SR}^*$ through the maximum, while the adjacent prose and empirical reporting call DSR benchmark-independent and describe its reference as $M_{N_t}$ alone. These descriptions coincide only when the retained benchmark does not exceed $M_{N_t}$; the paper does not state how that conflict is resolved. Do not silently substitute one interpretation for the other.

Report PSR as a value separately against each external benchmark. DSR is reported for Edge Allocation, Fixed Margin Utilization, and Short-Richness Scaling in both walk-forward and out-of-time slices. The source uses 0.95 as the conventional strong-evidence confidence level, while retaining numerical values rather than reducing them to pass/fail labels.

### Pairwise significance and family correction

For daily P&L $\pi_i^A$ and $\pi_i^B$ of a strategy and benchmark, respectively, define $d_i=\pi_i^A-\pi_i^B$ and $\overline d=N^{-1}\sum_i d_i$. The Diebold–Mariano statistic and two-sided normal-approximation p-value are:

$$
\operatorname{DM}=\frac{\overline d}{\sqrt{\widehat\sigma_d^2/N}},\qquad
p=2[1-\Phi(|\operatorname{DM}|)].
$$

Here $\widehat\sigma_d^2$ estimates the long-run variance of the P&L difference. At the one-step-ahead horizon $h=1$ used by the daily comparison, use sample variance:

$$
\widehat\sigma_d^2=\frac1{N-1}\sum_{i=1}^N(d_i-\overline d)^2.
$$

For $h>1$, the paper says to add the first $h-1$ Newey–West-style autocovariance terms, but does not specify their weights or estimator normalization. This longer-horizon extension is not fully specified.

For a jointly reported family of $m$ benchmark comparisons, apply Bonferroni:

$$
p_{\mathrm{adjusted}}=\min(1,mp),\qquad \alpha=0.05.
$$

The family size $m$ is the number of benchmark comparisons reported jointly (3 when the family consists of PUT, WPUT, and SPX buy-and-hold for a given method/slice). Evaluate the corrected p-values at the source's family-wise significance level 0.05.

## Empirical verification targets

### Evaluation basis and benchmark construction

The data workflow covers 2017–2025: 2017 supplies feature/statistic warm-up, 2018–2020 is the initial training period, 2021–2024 is the four-window annually retrained expanding-window walk-forward (WF), and calendar 2025 is the out-of-time (OOT) hold-out, excluded from training, hyperparameter search, and model selection. The ranker selects daily among eight delta-targeted short puts and SKIP, with entry at 10:00 ET. The seven sizing methods share the ranker, selection logic, and confidence gate; their position-size rules differ. Sizing calibration targets 16% annual volatility, whereas realized evaluation volatility can be substantially lower.

Abbreviations: EA = Edge Allocation; FMU = Fixed Margin Utilization; SRS = Short-Richness Scaling; VT = Volatility Targeting; GB = Gamma-Budgeted Sizing; HK = half-Kelly; QK = quarter-Kelly. Ann. return and Ann. vol. denote annualized return and volatility; Max DD is the positive magnitude of maximum drawdown. Table return, volatility, and drawdown entries are fractions, not percentages. WF and OOT metrics are computed separately.

Strategy returns are daily P&L per unit of collateral. Collateral is cash earning the risk-free rate, so the strategy P&L returns are already excess returns. PUT and WPUT use total-return index series. SPX is a price index: accrue its continuous dividend yield daily before converting to excess returns. Subtract EFFR from all three external benchmark return series to compare on the same excess-return basis.

The internal baselines use the strategy's option universe and execution but disable the ranker, confidence gate, or both:

| Baseline | Reconstructible selection rule | Unspecified implementation details |
|---|---|---|
| Random | Random candidate selection | Sampling distribution, whether SKIP participates, random seed, gating, and sizing rule are not stated. |
| Always-P25d | Always short the 25-delta put | Exact baseline gating and sizing rule are not stated. |
| Always-P45d | Always short the 45-delta put | Exact baseline gating and sizing rule are not stated. |
| Momentum | A momentum selector | Signal, lookback, decision rule, gating, and sizing are not stated. |
| Rolling-Sharpe | Select the candidate with the highest 30-day rolling Sharpe | Tie-breaking, whether SKIP participates, gating, and sizing are not stated. |

These omissions prevent exact reproduction of the internal baselines; do not invent their rules from their names. External comparators are SPX buy-and-hold (BH), CBOE PUT, and CBOE WPUT index return series, transformed as specified above.

### Headline performance — full source Table 5

| Method | WF Sharpe | WF Sortino | WF Ann. return | WF Ann. vol. | WF Max DD | OOT Sharpe | OOT Sortino | OOT Ann. return | OOT Ann. vol. | OOT Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Sizing methods** | | | | | | | | | | |
| EA | 3.1048 | 4.5234 | 0.1091 | 0.0351 | 0.0228 | 5.7612 | 7.0291 | 0.1048 | 0.0182 | 0.0143 |
| FMU | 2.5119 | 2.9925 | 0.1319 | 0.0525 | 0.0691 | 5.0632 | 6.0447 | 0.1795 | 0.0354 | 0.0196 |
| SRS | 2.7597 | 3.4259 | 0.1270 | 0.0460 | 0.0614 | 5.2622 | 6.9824 | 0.1755 | 0.0334 | 0.0172 |
| VT | 2.2102 | 2.4040 | 0.1489 | 0.0674 | 0.0900 | 4.6669 | 4.9068 | 0.2388 | 0.0512 | 0.0343 |
| GB | 2.0554 | 2.2676 | 0.1366 | 0.0665 | 0.0899 | 5.0147 | 5.8707 | 0.2995 | 0.0597 | 0.0338 |
| HK | 1.9464 | 2.2562 | 0.1200 | 0.0616 | 0.0860 | 4.3711 | 5.1981 | 0.2091 | 0.0478 | 0.0271 |
| QK | 1.8965 | 2.1857 | 0.1186 | 0.0625 | 0.0925 | 4.3084 | 5.1671 | 0.1993 | 0.0463 | 0.0273 |
| **Internal baselines** | | | | | | | | | | |
| Random | 0.0844 | 0.0912 | 0.0131 | 0.1550 | 0.3935 | 0.3415 | 0.3626 | 0.0570 | 0.1669 | 0.1091 |
| Always-P25d | -0.0191 | -0.0203 | -0.0030 | 0.1571 | 0.3438 | 0.0005 | 0.0005 | 0.0001 | 0.1663 | 0.1273 |
| Always-P45d | -0.0405 | -0.0456 | -0.0064 | 0.1581 | 0.3396 | 0.1577 | 0.1752 | 0.0248 | 0.1572 | 0.1146 |
| Momentum | -0.0051 | -0.0057 | -0.0008 | 0.1616 | 0.2894 | -0.3982 | -0.4276 | -0.0669 | 0.1681 | 0.1490 |
| Rolling-Sharpe | 0.5569 | 0.5991 | 0.0698 | 0.1254 | 0.2383 | 0.6128 | 0.6258 | 0.0545 | 0.0890 | 0.0830 |
| **External index benchmarks** | | | | | | | | | | |
| SPX buy-and-hold (BH) | 0.6221 | 0.8871 | 0.1011 | 0.1625 | 0.3086 | 0.4622 | 0.6787 | 0.0884 | 0.1912 | 0.2008 |
| CBOE PUT index | 0.6795 | 0.9091 | 0.0675 | 0.0993 | 0.1994 | 0.1754 | 0.2605 | 0.0252 | 0.1436 | 0.1642 |
| CBOE WPUT index | 0.1449 | 0.1879 | 0.0145 | 0.1003 | 0.2285 | 0.1312 | 0.1593 | 0.0151 | 0.1152 | 0.1339 |

Every sizing method beats every internal and external comparator on Sharpe in both slices. The best internal baseline is Rolling-Sharpe (0.5569 WF; 0.6128 OOT); the smallest model advantage is QK's approximately 1.34 WF and 3.70 OOT. EA, SRS, FMU are the top three methods, in that order, in both slices. EA's Sharpe advantage over external benchmarks ranges from approximately 2.43 (PUT, WF) to 5.63 (WPUT, OOT). The smallest model/external gap is QK versus PUT on WF (1.22) and QK versus BH on OOT (3.85).

### Statistical confidence — full source Table 6

PSR denotes Probabilistic Sharpe Ratio; DSR denotes Deflated Sharpe Ratio. PSR is benchmark-specific. DSR is reported as benchmark-independent (subject to the formula/prose conflict documented in the metric definitions), with $N_t=75$ performance-ranked trials. The source uses 0.95 as its reported confidence comparison level.

| Method | Slice | PSR vs BH | PSR vs PUT | PSR vs WPUT | DSR |
|---|---|---:|---:|---:|---:|
| EA | WF | 0.9999 | 0.9998 | 1.0000 | 0.9962 |
| EA | OOT | 0.9636 | 0.9710 | 0.9721 | 0.8563 |
| FMU | WF | 0.9837 | 0.9806 | 0.9967 | 0.9168 |
| FMU | OOT | 0.9836 | 0.9887 | 0.9893 | 0.8638 |
| SRS | WF | 0.9934 | 0.9919 | 0.9989 | 0.9561 |
| SRS | OOT | 0.9944 | 0.9966 | 0.9968 | 0.9128 |

For methods outside the top three, source-reported PSR against the “worst of the three benchmarks” is:

| Method | WF PSR | OOT PSR |
|---|---:|---:|
| VT | 0.9240 | 0.9493 |
| GB | 0.9063 | 0.9773 |
| HK | 0.9264 | 0.9722 |
| QK | 0.9203 | 0.9684 |

The 75-trial count consists of 28 reported configurations (headline + 10 structural sensitivity runs + 15 feature-group ablations + 2 post-processed execution-drag variants), 40 confidence-gate calibrations (eight trade-rate values, scored by Sortino in each of five windows), six alternative sizing methods, and one abandoned all-options variant. The 250 hyperparameter trials (50 per window) are excluded because selection uses validation NDCG@1. The 740 position-size grid evaluations are excluded because selection minimizes deviation from the 16% training-volatility anchor; proportional scaling also leaves Sharpe invariant.

All 18 top-three PSR cells exceed 0.95. Only EA and SRS have WF DSR above 0.95; none of the top three has OOT DSR above 0.95. The paper attributes the OOT shortfall to the single-year sample and 75-trial correction. The daily-P&L Diebold–Mariano comparisons give a different conclusion: after Bonferroni correction across three external benchmarks, no OOT method rejects equal mean daily P&L at conventional levels. Corrected minimum p-values are 1.000 for EA, FMU, and SRS; the smallest among all seven is 0.231 for GB. PSR compares risk-adjusted returns, whereas this DM application compares mean daily P&L levels.

**Source ambiguity — lock before coding:** “Worst benchmark” is not an unambiguous PSR implementation instruction. The top-three summary values quoted in the prose correspond to the minimum PSR across benchmarks (the most demanding comparison), not necessarily the lowest-Sharpe benchmark. Preserve the full three-benchmark table and resolve this wording before implementing the four-method summary calculation.

### Year and regime interpretation

Every sizing method's OOT Sharpe exceeds its aggregate WF Sharpe by more than 2.4. Every method's strongest year is 2021 (Sharpe 6.52–9.21), and weakest year is 2024: only EA (0.41) and VT (0.05) are positive. EA's yearly Sharpe is 6.52, 2.51, 1.79, 0.41, and 5.76 for 2021–2025. HK and QK abstain for all of 2023; the paper attributes this to a positive Kelly threshold without stating its value. Reported zero Sharpe denotes an undefined zero-variance ratio rather than a data error. They lose in 2024 (HK -0.58; QK -0.62) and recover in 2025 (4.37; 4.31). A single strong hold-out year does not separate generalization from a favorable volatility regime.

VIX regime labels use **entry-day closing VIX**: low $VIX<15$, mid $15\leq VIX\leq25$, high $VIX>25$. They are retrospective evaluation labels, not information available at the 10:00 ET entry. Regime statistics include all days assigned to a regime; SKIP and gate-blocked days have zero return. Days and Trades are separate counts. WF sample counts are 223 low, 593 mid, and 148 high. EA's OOT counts are 21 low, 196 mid, and 20 high, totaling 237.

Realized-volatility regimes use 5-day intraday realized S&P 500 volatility measured at the entry-day close, divided into equal-quantile terciles **separately within each slice**. Thus WF and OOT terciles need not share numeric cutoffs and are retrospective labels.

WF annual return peaks in mid-VIX for six methods. EA instead rises from 0.50% to 13.90% to 15.40% from low to high, while its annual volatility rises 1.91%, 3.28%, 5.57%. Mid-VIX Sharpe is highest for five methods, ranging 3.59 (QK)–4.48 (GB); VT and GB instead peak in low VIX at Sharpe 5.46 and 6.62, with volatility 1.33% and 0.81%. EA's OOT mid-VIX Sharpe is 4.94. The outer OOT regimes have only 20/21 observations, so their annualized statistics are unstable; the prose reports low-VIX OOT Sharpe 13.71 for EA and 83.29 for GB (GB annual return 23.78%).

EA's OOT annual return increases across RV terciles: 4.11%, 10.25%, 17.67%. WF top-tercile return exceeds bottom for all methods except VT (13.22% versus 13.53%); it exceeds bottom for every OOT method. WF Sharpe peaks in tercile 2 for six methods; EA instead rises 2.77, 2.98, 3.55. No OOT method has monotone Sharpe across all three RV terciles. OOT Sharpe exceeds WF in 34 of 42 method/regime cells; six of eight exceptions occur in middle RV, where all methods except EA invert.

### Figure-derived scalar targets and extraction gaps

The cumulative NAV and underwater drawdown plots cover 2021-01-04 through 2025-12-31, with common initial capital \$5,000,000 and WF/OOT split 2025-01-01. Full plotted daily paths are not recoverable from layout text. The prose supplies these scalar targets:

| Target | Source-reported value |
|---|---|
| EA terminal NAV | \$8.15M; 63.0% cumulative gain; geometric annualized return 10.8% |
| VT terminal NAV | \$10.38M; 107.7% cumulative gain |
| GB terminal NAV | \$10.49M; 109.5% cumulative gain |
| EA deepest full-sample drawdown | -2.28%, 2022-06-28 |
| Deepest drawdown across sizing methods | QK -9.25%, 2024-08-07 |

The per-regime grouped bar plot compares seven sizing methods across VIX and RV regimes, with WF and OOT panels. Its OOT VIX vertical axis is capped at 16 and values above the cap are hatched. No bar heights or time-series coordinates have been inferred from text-extracted axes.

**Figure/table reconciliation:** The regime-plot caption reports a maximum OOT low-VIX Sharpe of 88.8: the full regime table identifies this as VT (88.7880), distinct from GB (83.2863). The low/high OOT day counts are 21/20.

**Source claim requiring resolution — lock before coding:** The prose describes passive short-volatility drawdowns of -30% to -50% “over the same sample,” but the headline table shows WF Max DD of 19.94% for PUT and 22.85% for WPUT (internal Always-P25d/P45d are 34.38%/33.96%); no table entry establishes a 50% drawdown. Do not treat that range as a verified external-index result.

## Robustness: structural sensitivity, ablations, and execution drag

WF denotes walk-forward 2021–2024; OOT denotes the held-out calendar year 2025. All values below reproduce the paper's compounded-return Sharpe convention. Each delta is the perturbed Sharpe minus the corresponding headline Sharpe. PSR is reported against the worst of the three external benchmarks for each slice. Printed deltas are preserved, including last-digit differences caused by rounding the displayed Sharpe inputs.

### Structural sensitivity settings and verification matrix

Change one dimension at a time from the headline: Edge Allocation (EA), 16% training-volatility anchor, equal-volatility size calibration, expanding walk-forward training with a 3-year initial training window, 50 hyperparameter trials per window, and correlation-clustering threshold 0.85. The ten structural perturbations are a 2-year training window, rolling 3-year training, volatility anchors 12% and 20%, trial counts 25 and 100, correlation thresholds 0.95 and 0.90, and replacement of equal-volatility calibration with in-sample Sharpe or Sortino maximization. The execution rows are separate post-processed repricings with selection fixed.

**Table 7 — EA sensitivities.**

| Perturbation | Sharpe WF | Sharpe OOT | Δ Sharpe WF | Δ Sharpe OOT | PSR WF | PSR OOT |
|---|---:|---:|---:|---:|---:|---:|
| Headline, volatility anchor 16% | 3.105 | 5.761 | 0.000 | 0.000 | 1.000 | 0.964 |
| Training window 2 years | −0.157 | 5.178 | −3.262 | −0.583 | 0.050 | 0.952 |
| Rolling 3-year window | 0.564 | 5.227 | −2.541 | −0.534 | 0.454 | 0.953 |
| Volatility anchor 12% | 3.036 | 5.739 | −0.069 | −0.023 | 1.000 | 0.964 |
| Volatility anchor 20% | 3.105 | 5.761 | 0.000 | 0.000 | 1.000 | 0.964 |
| Hyperparameter trials 25 | 3.308 | 5.736 | 0.203 | −0.025 | 1.000 | 0.963 |
| Hyperparameter trials 100 | 0.706 | 5.800 | −2.399 | 0.039 | 0.530 | 0.959 |
| Correlation threshold 0.95 | 0.976 | 4.817 | −2.129 | −0.944 | 0.632 | 0.993 |
| Correlation threshold 0.90 | 0.281 | 0.000 | −2.824 | −5.761 | 0.281 | 0.000 |
| Calibration: Sharpe maximization | 2.947 | 5.615 | −0.158 | −0.146 | 1.000 | 0.966 |
| Calibration: Sortino maximization | 2.902 | 5.292 | −0.202 | −0.469 | 1.000 | 0.960 |
| Execution: 75% spread coverage | 2.826 | 5.426 | −0.279 | −0.335 | 0.999 | 0.957 |
| Execution: sell at bid | 2.725 | 5.311 | −0.380 | −0.450 | 0.999 | 0.954 |

The correlation-0.90 OOT row is a **no-trade year for all seven sizing methods**: the confidence gate rejects every day, cash excess returns are zero, and Sharpe is mathematically undefined. The displayed 0.000 is the paper's reporting convention, not measured risk-adjusted performance. Among structural cells that trade, EA OOT Sharpe spans 4.817–5.800; WF spans −0.157–3.308. The no-trade cell must be tracked separately when reproducing this range.

The 20% anchor does not move the selected sizing parameter $\theta^\star$ in 24 of 35 method-window cells, including all five EA windows, because the headline already selects the top of the grid. Its identical EA result is therefore not evidence that EA is insensitive to a changed anchor. The 12% arm changes EA's selected parameter from 0.80 to 0.70 in one window. Where the parameter moves, approximate proportional exposure rescaling largely preserves Sharpe. Sharpe- and Sortino-maximizing size calibration preserve OOT method ordering across all seven methods.

The source diagnoses the 0.90 failure as sensitivity of top-1 minus top-2 confidence to redundant feature representations. For its union calibration across four WF held-out slices (965 days), it reports:

| Gate configuration | Selected threshold $\tau^\star$ | Calibrated admission fraction | Realized 2025 admission fraction |
|---|---:|---:|---:|
| Headline | 0.0000 | 1.0000 | 1.0000 |
| Correlation threshold 0.90 | 0.0689 | 0.3005 | 0.0000 |

These are union-calibration diagnostics, distinct from per-window gate parameters. The paper illustrates a signal-only check through consecutive abstentions: under independent daily admission probability $p=0.3005$,

$$
\Pr(k\text{ consecutive abstentions})=(1-p)^k.
$$

Its reported probabilities are 0.028 for $k=10$ and below 0.001 for $k=20$. Independence is an approximation because confidence is serially correlated; these examples are source diagnostics, not a specified production alarm policy.

### Feature-group ablation protocol

Drop one of the fifteen groups from the candidate catalog **before** the feature-selection pipeline, rerun the pipeline, and compare against the no-drop headline. Drop both native features and every derived feature source-tagged to the removed group's primary inputs.

| Scope | Groups |
|---|---|
| Cross-sectional, one daily value broadcast to all candidates | Calendar/event flags; morning session; SPX index; VIX family/futures; macroeconomic indicators; implied-volatility surface; realized-minus-implied volatility (RV/IV); realized higher moments; VIX term-structure curvature; trend |
| Per-strategy, one value per day/candidate | Position greeks; per-strategy rolling statistics; entry liquidity; intra-strategy term context; regime-conditional sensitivities |

The derived families included in source-tagged deletion are within-day ordinal ranks of six per-strategy inputs and three tail-risk signals: the relative gap between 5-day realized volatility and preceding-week 1-week ATMF implied volatility; the strategy's 5-day drawdown at the prior close; and the VIX 60-day z-score. Regime sensitivities comprise seven multiplicative interactions between per-strategy quantities and cross-sectional regimes. The position ablation removes eight native greeks plus five sourced sensitivities.

**Table 8 — EA feature-group ablations.**

| Ablated group | Sharpe WF | Sharpe OOT | Δ Sharpe WF | Δ Sharpe OOT | PSR WF | PSR OOT |
|---|---:|---:|---:|---:|---:|---:|
| Headline, no group dropped | 3.105 | 5.761 | 0.000 | 0.000 | 1.000 | 0.964 |
| Volatility surface | −0.051 | 5.481 | −3.156 | −0.281 | 0.092 | 0.977 |
| Intra-strategy | 0.081 | 3.611 | −3.024 | −2.150 | 0.165 | 0.970 |
| Regime sensitivities | 0.196 | 5.043 | −2.909 | −0.718 | 0.229 | 0.933 |
| RV/IV | 0.264 | 5.224 | −2.841 | −0.538 | 0.277 | 0.952 |
| Morning | 0.389 | 3.660 | −2.716 | −2.101 | 0.348 | 0.895 |
| VIX | 0.469 | 5.299 | −2.635 | −0.462 | 0.402 | 0.976 |
| VIX curvature | 0.735 | 5.887 | −2.370 | 0.126 | 0.539 | 0.963 |
| Entry liquidity | 1.124 | 5.102 | −1.981 | −0.660 | 0.731 | 0.987 |
| Higher moments | 1.673 | 4.162 | −1.431 | −1.599 | 0.912 | 0.915 |
| SPX index | 2.042 | 4.609 | −1.062 | −1.152 | 0.964 | 0.924 |
| Position greeks | 2.121 | 5.148 | −0.984 | −0.614 | 0.971 | 1.000 |
| Per-strategy statistics | 2.145 | 4.791 | −0.959 | −0.970 | 0.958 | 0.935 |
| Calendar | 2.199 | 3.265 | −0.906 | −2.496 | 0.969 | 0.922 |
| Trend | 3.105 | 5.761 | 0.000 | 0.000 | 1.000 | 0.964 |
| Macro | 4.188 | 4.975 | 1.083 | −0.786 | 0.998 | 0.931 |

The trend ablation is an exact no-op: all four trend features fail the relevance threshold (cross-day Spearman correlation with daily mean label below 0.05) in every one of the four WF windows and the OOT window. Macro removal raises EA WF Sharpe but lowers OOT PSR from 0.964 to 0.931. For the other six sizing methods, removing macro instead raises OOT Sharpe by 1.13–1.98 and leaves OOT PSR at or above 0.95. This is a method-specific result, not evidence for universally removing macro features.

Across sizing methods, ablation-group ordering correlates with EA at Spearman 0.73–0.94 in WF and 0.08–0.41 OOT; individual OOT ablation effects differ from EA by up to 4.33 Sharpe units. The volatility surface, intra-strategy context, and regime sensitivities cause the largest EA WF losses; calendar causes the largest EA OOT loss.

**Known figure-only gap:** Figure 5 contains the complete WF/OOT ablation delta bars for all seven sizing methods. Exact non-EA method-by-group values are not tabulated or printed on those bars and are not reconstructed here; the original figure is required for the full visual comparison. The EA numerical matrix is fully recovered above.

### Execution repricing protocol and verification matrix

Reprice the headline's realized entries under three assumptions, holding trained model selections, gate decisions, and each day's headline position size $Q$ fixed. Do not retrain or recalibrate sizing. Let $b$ and $a$ be entry bid and ask, and $c$ the fraction of the spread surrendered from the ask:

$$
p_{\mathrm{entry}}(c)=c b+(1-c)a,\qquad
c\in\{0.50,0.75,1.00\}.
$$

Thus the headline is midprice $(a+b)/2$, the intermediate stress is $0.75b+0.25a$, and the extreme stress sells at bid $b$. Recompute daily P&L with those premiums and the unchanged settlement outcomes. Fixed $Q$ prevents the stressed results from benefiting from a sizing recalibration under worse fills. For each method and stressed fill $c$, the reported OOT percentage drag is:

$$
100\left(\frac{\operatorname{SR}_{\mathrm{OOT}}(c)}
{\operatorname{SR}_{\mathrm{OOT}}(0.50)}-1\right).
$$

**Table 9 — Execution sensitivity across sizing methods.** EA = Edge Allocation; FMU = Fixed Margin Utilization; SRS = Short-Richness Scaling; VT = Volatility Targeting; GB = Gamma-Budgeted Sizing; HK = Half-Kelly; QK = Quarter-Kelly.

| Method | Mid WF | Mid OOT | 75% WF | 75% OOT | Bid WF | Bid OOT | OOT drag 75% (%) | OOT drag bid (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EA | 3.105 | 5.761 | 2.826 | 5.426 | 2.725 | 5.311 | −5.8 | −7.8 |
| FMU | 2.512 | 5.063 | 2.207 | 4.736 | 2.104 | 4.616 | −6.5 | −8.8 |
| SRS | 2.760 | 5.262 | 2.453 | 5.017 | 2.349 | 4.918 | −4.7 | −6.5 |
| VT | 2.210 | 4.667 | 1.882 | 4.286 | 1.772 | 4.158 | −8.2 | −10.9 |
| GB | 2.055 | 5.015 | 1.776 | 4.657 | 1.682 | 4.527 | −7.1 | −9.7 |
| HK | 1.946 | 4.371 | 1.675 | 4.070 | 1.583 | 3.960 | −6.9 | −9.4 |
| QK | 1.896 | 4.308 | 1.627 | 4.018 | 1.537 | 3.911 | −6.7 | −9.2 |

Every method's OOT Sharpe degrades monotonically as fills worsen, and method ordering is preserved. The 75% stress costs 4.7–8.2% of OOT Sharpe; bid fills cost 6.5–10.9%. These are consequences of the stated fill assumptions, not externally estimated market-impact coefficients.

## Mechanism decomposition and calibration targets

### Two-by-two risk-control ablation (source Table 10)

Toggle the confidence gate and the three tail-risk features (`rv_surprise_5d`, `prior_day_strategy_drawdown_max_5d`, `vix_z_60d`) while retaining the training split, selection pipeline, EA sizing and fills. Disable the gate by constraining its calibration to trade rate one, preserving the held-out training discipline. Of the three tail signals, only prior-close drawdown survives headline selection, in four of five windows; dropping the family can still alter selection of other candidates.

| Configuration / component | Sharpe WF | Sharpe OOT |
|---|---:|---:|
| Neither control | 0.623 | 5.221 |
| Tail-risk features only | 0.160 | 5.873 |
| Confidence gate only | 0.759 | 5.233 |
| Both controls, headline | 3.105 | 5.761 |
| Baseline component | 0.623 | 5.221 |
| Tail-feature main effect | −0.463 | +0.652 |
| Gate main effect | +0.136 | +0.012 |
| Interaction | +2.809 | −0.124 |
| Total | 3.105 | 5.761 |
| CBOE PUT comparator | 0.680 | 0.175 |

For configuration Sharpe $S_{ab}$ with gate toggle $a$ and tail-feature toggle $b$, main effects are $S_{10}-S_{00}$ and $S_{01}-S_{00}$; the interaction is $S_{11}-S_{10}-S_{01}+S_{00}$. Rounded displayed numbers may differ at the last digit from the printed decomposition.

OOT, the base pipeline supplies 5.046 of the headline's 5.586 Sharpe gap over PUT; the two controls jointly add 0.540. This base includes selection, sizing, margin and execution, so the ablation does not isolate the ranker alone. In WF the controls' interaction supplies most of the difference. The OOT gate threshold is zero in all four configurations; the small nonzero gate effect is reported without a full mechanical reconciliation.

### Confidence-gate calibration (source Table E.1)

| Prediction window | Threshold $\tau^*$ | Reported trade/admission rate | Objective |
|---|---:|---:|---|
| 2021 | 0.0000 | 1.0000 | Sortino |
| 2022 | 0.2013 | 0.5041 | Sortino |
| 2023 | 0.1613 | 0.3058 | Sortino |
| 2024 | 0.0137 | 0.4000 | Sortino |
| OOT 2025 | 0.0000 | 1.0000 | Sortino |
| WF held-out union | 0.0000 | 1.0000 | Sortino |

WF rows are calibrated on each window's last six held-out months. The union row combines the four WF hold-outs and supplies the 2025 threshold; the OOT row's rate is measured on 2025. Admission rate is the fraction passing the score-gap test, not necessarily the executed-trade rate after SKIP and integer sizing. The multiple-testing inventory reports eight trade-rate points per window but does not list their values or the exact mapping to score thresholds.

### Cross-window feature survival

Across five selection windows, 85 distinct features survive at least once: 27 in five windows, 10 in four, 16 in three, 13 in two, 19 in one. The mean pairwise Jaccard similarity $|A\cap B|/|A\cup B|$ is 0.58, ranging from 0.47 (2022 versus 2025) to 0.67 (2023 versus 2024).

The 27 always-selected identifiers are:

- `delta`, `gamma`, `theta`, `vega`, `dollar_delta`, `gamma_exposure`, `log_moneyness`, `leverage`;
- `atmf_iv_1dte_percentile_252d`, `atmf_iv_10dte_intraday_range`, `iv_to_spot_change_ratio_1dte_close`;
- `delta_distance_from_target`, `iv_at_strike_minus_atmf`, `iv_at_strike_minus_atmf_pct`;
- `entry_bid_ask_spread`, `entry_bid_ask_spread_pct`, `entry_log_premium`;
- `VIX_5d_pct_change`, `VIX_21d_pct_change`, `spx_1d_returns`, `spx_252d_rv`;
- `is_post_holiday_session`, `days_until_next_pce`;
- `delta_x_VIX`, `gamma_x_morning_spx_rv_annualized`, `iv_at_strike_minus_atmf_wd_rank`, `win_rate_30d`.

These are empirical verification targets, not a fixed feature list to substitute for per-window selection.

### Calendar-year Sharpe (source Table C.1)

| Method | 2021 | 2022 | 2023 | 2024 | 2025 OOT |
|---|---:|---:|---:|---:|---:|
| EA | 6.5184 | 2.5066 | 1.7852 | 0.4074 | 5.7612 |
| FMU | 7.3126 | 1.5074 | 2.5067 | −0.2873 | 5.0632 |
| SRS | 7.3571 | 2.0167 | 2.6270 | −0.2115 | 5.2622 |
| VT | 7.5322 | 1.3736 | 2.8968 | 0.0462 | 4.6669 |
| GB | 9.2101 | 1.4893 | 3.9302 | −0.1927 | 5.0147 |
| HK | 7.3126 | 1.5266 | 0.0000 | −0.5776 | 4.3711 |
| QK | 7.3126 | 1.5266 | 0.0000 | −0.6205 | 4.3084 |

Each cell uses only that calendar year's daily excess returns and the compounded-return Sharpe convention. HK and QK hold no positions during 2023; zero is the reporting convention for undefined zero-variance Sharpe. The note attributes this to an estimated Kelly fraction below an abstention threshold, but the sizing equations specify clipping/flooring rather than an additional numerical Kelly threshold.

## Complete feature catalog and availability

The catalog distinguishes CS (one day-level value broadcast to candidates) and PS (candidate-level value). Feature names below retain the paper's identifiers. Computation dates and prediction dates must be kept distinct: a daily-close statistic computed on $u$ first belongs to entry row $u+1$, using trading-day shifts. A lag is applied once to the completed statistic, not repeatedly to its already-lagged constituents.

| Block | Available | Entry lag |
|---|---|---|
| Calendar/events | In advance | None |
| Morning SPX, 09:30–09:59 | 10:00 ET | None |
| Morning 09:35-to-10:00 surface changes | 10:00 ET | None |
| Morning VIX/VVIX close-based variables | Day close | At least one trading day |
| SPX index | Day close | At least one trading day |
| Spot VIX family | Day close | At least one trading day |
| VIX futures front complex | Settlement | At least one trading day |
| Macroeconomic indicators | Release | At least one trading day |
| IV surface, 10:00 snapshot | 10:00 ET | None |
| IV surface, closing snapshot/intraday dynamics | Day close | At least one trading day |
| RV-minus-IV | Day close | At least one trading day |
| Realized higher moments | Day close | At least one trading day |
| VIX curvature | Day close | At least one trading day |
| Trend | Day close | At least one trading day |
| Position Greeks/exposures | 10:00 ET | None |
| Per-strategy rolling statistics | Trade outcome | At least one trading day |
| Entry liquidity | 10:00 ET | None |
| Intra-strategy term context | 10:00 ET | None |
| Regime-conditional products | Day close | At least one trading day |

This reproduces source Table B.1, including its lag on **all regime-interaction products**, even those whose factors are available at entry. The narrative's description of morning VIX changes and daily RV as intraday is superseded by the explicit feature definitions below. In contrast, realized-volatility regime attribution later uses a five-day *intraday* statistic; the paper does not reconcile that estimator with the close-to-close feature.

Let $P_u$ be the SPX close, $S$ an intraday mid quote, $r_{N,u}=P_u/P_{u-N}-1$, and $\ell_{N,u}=\ln(P_u/P_{u-N})$. Volatilities are annualized decimals; use 252 trading days per year. DTE denotes days to expiration; $T$ in Black–Scholes denotes the corresponding year fraction. EFFR is decimal, and dividend yield is zero. The paper leaves holiday adjustments to OPEX, missing data, minimum rolling observations, zero-denominator handling (except where stated), sample-moment bias corrections, and some month/year-to-session conversions unspecified.

PS features are missing on SKIP unless an explicit exception is stated. Features with `_wd_rank` are within-day ordinal ranks across the eight real candidates; assign SKIP rank 8.5. The ordering direction and real-candidate tie policy are not specified. These ranks are feature inputs, separate from the model's output scores.

### Calendar and event flags

Use the NYSE trading calendar and historical release-date tables.

| Feature | Definition |
|---|---|
| `day_of_week` | Monday=1 through Friday=5 |
| `day_of_month`, `month_of_year` | Integers 1–31 and 1–12 |
| `is_monthly_opex` | Third Friday of any month |
| `is_quarterly_opex` | Third Friday of March, June, September, December |
| `is_post_holiday_session`, `is_pre_holiday_session` | First session after / last session before a market holiday |
| `days_until_next_fomc`, `days_since_last_fomc` | Trading-day counts to/from the FOMC announcement |
| `is_cpi_day`, `days_until_next_cpi` | BLS CPI release indicator and forward count |
| `is_nfp_day`, `days_until_next_nfp` | BLS Employment Situation release indicator and forward count |
| `is_pce_day`, `days_until_next_pce` | BEA PCE release indicator and forward count |

FOMC announcement days are excluded from the candidate universe upstream.

### Morning session

Use one-minute SPX mid prices $S_{09:30},\ldots,S_{09:59}$, with the 09:59 closing mid representing the 10:00 mark so the entry-minute bar is excluded. Let $H,L$ be the maximum/minimum over that window, and $P_{u-1}$ the prior 16:00 close.

| Feature | Definition |
|---|---|
| `morning_spx_log_return` | $\ln(S_{09:59}/S_{09:30})$ |
| `morning_spx_range_pct` | $(H-L)/S_{09:30}$ |
| `morning_spx_directionality` | $(S_{09:59}-S_{09:30})/(H-L)$; zero if $H=L$ |
| `morning_gap_size` | $(S_{09:30}-P_{u-1})/P_{u-1}$ |
| `morning_gap_filled` | Indicator $L\le P_{u-1}\le H$ |
| `morning_atmf_iv_change` | 1-DTE ATMF IV at 10:00 minus its 09:35 value |
| `morning_atmf_iv_pct_change` | That change divided by the 09:35 IV |
| `morning_skew_change` | 10:00 minus 09:35 change in 1-DTE $\sigma^C_{25}-\sigma^P_{25}$ |
| `morning_vix_level` | Prior trading-day VIX close |
| `morning_vix_change`, `morning_vvix_change` | One-day changes of VIX/VVIX closes, lagged one day |

The displayed source formula for `morning_spx_rv_annualized` is

$$
\sigma_{\mathrm{morn}}=\sqrt{252\cdot390\sum_j\left(\ln\frac{S_{t_j}}{S_{t_{j-1}}}\right)^2}.
$$

Here $j$ indexes successive minute observations in the morning window and 390 is the regular-session minute count. **Source formula verified visually:** there is no division by the number of observed minute returns. This differs from annualizing their mean square; do not silently normalize it. Surface changes use actual 09:35 and 10:00 snapshots, separately from the 09:59 SPX price convention.

### SPX index and put-call ratios

All values are lagged at least one trading day before entry.

- `spx_{N}_returns`, $N\in\{1d,5d,10d,1m,3m,6m,1y\}$: simple return $r_{N,u}$. The overview calls them log returns, but the detailed definition explicitly uses simple returns.
- `spx_returns_roll_avg_30d`, `spx_returns_roll_std_30d`: 30-day mean and standard deviation of the one-day simple return.
- `spx_{N}d_rv`, $N\in\{5,21,63,252\}$, uses daily log returns:

$$
\mathrm{RV}_{N,u}=\sqrt{\frac{252}{N}\sum_{i=1}^{N}\left(\ln\frac{P_{u-i+1}}{P_{u-i}}\right)^2}.
$$

- `spx_1y_percentile`: fraction of the previous 252 closes strictly below the current close.
- `spx_position_52w`: $(P_u-\min_{252}P)/(\max_{252}P-\min_{252}P)$.
- `spx_put_call_ratio`, `vix_put_call_ratio`: daily put/call option volume ratios for the respective underlyings.
- `spx_pcr_21d_avg`, `vix_pcr_21d_avg`: their 21-day means.
- `spx_pcr_5d_pct_change`, `spx_pcr_21d_pct_change`, `vix_pcr_5d_pct_change`, `vix_pcr_21d_pct_change`: their five-/21-day percentage changes.

### VIX family and futures

Use daily closes for spot indices and settlements for futures, all lagged at least one trading day.

| Features | Definition |
|---|---|
| `VIX`, `VIX1D`, `VIX9D`, `VIX3M`, `VIX6M`, `VVIX` | 30-day, one-day, nine-day, three-month, six-month implied-volatility indices and volatility of VIX |
| `VIX_VIX1D_ratio`, `VIX_VIX9D_ratio`, `VIX_VIX3M_ratio`, `VIX_VIX6M_ratio` | VIX divided by the named tenor |
| Corresponding `_spread` features | VIX minus the named tenor |
| `{X}_5d_pct_change`, `{X}_21d_pct_change`, $X\in\{\mathrm{VIX,VIX1D,VVIX}\}$ | Five-/21-day relative changes |
| `{X}_1y_percentile`, same $X$ set | 252-day within-series percentile ranks |
| `vix_futures_front_price` | Nearest monthly future's settlement $M_1$ |
| `vix_front_slope` | $M_2-M_1$ |
| `vix_front_curvature` | $2M_2-M_1-M_3$ |
| `front_roll_yield` | $(M_2-M_1)/D_1$, with $D_1$ trading days to front-contract expiry |
| `vix_z_60d` | $(\mathrm{VIX}_u-\overline{\mathrm{VIX}}_{60,u})/s_{\mathrm{VIX},60,u}$ |

$M_i$ is the $i$th nearest monthly settlement. `vix_z_60d` is a tail-risk feature. Native VIX index levels are in volatility points, while computed IV is decimal; the sizing layer explicitly divides VIX9D by 100 where needed. Futures rollover details are not given.

### Macroeconomic indicators

Use FRED EFFR and seasonally non-adjusted weekly initial unemployment claims, forward-filled to the trading calendar and lagged at least one trading day from release.

- `effr_rate`: decimal annual EFFR, e.g. 0.0525 for 5.25%.
- `effr_rate_1y_pct_change`, `effr_rate_1y_percentile`: 252-day relative change and rolling percentile.
- `jobless_claims`: released weekly level.
- `jobless_claims_1y_pct_change`, `jobless_claims_1y_percentile`: 252-day relative change and percentile of the calendar-aligned series.

The release-time lag is specified, but the source does not identify revision-vintage handling or exact source series identifiers.

### Implied-volatility surface features (CS; source B.7)

These cross-sectional features use SPXW surface snapshots at **09:35 ET, 10:00 ET, and the official close**. The 10:00 snapshot is usable at the same day's entry; closing snapshots and full-day intraday-change features require **at least one trading day's lag**. Formulas below describe each observation at its measurement date, before this availability lag.

For each timestamp and DTE target $\tau$, select the chain expiration closest to $\tau$ and fit an IV-versus-delta interpolation. ATMF IV is the IV of the put closest to the forward strike

$$
F=S e^{rT},
$$

where $S$ is contemporaneous spot, $r$ is the EFFR risk-free rate, and $T$ is maturity in years. The paper uses Black–Scholes with no dividends and annualized decimal IV. Let $\sigma_{\mathrm{ATM},\tau,t}$ denote ATMF IV for target $\tau$, and $\sigma_{C\delta,\tau,t}$ and $\sigma_{P\delta,\tau,t}$ the call and put IVs at delta magnitude $\delta/100$, for $\delta\in\{10,25\}$.

**Delta-sign ambiguity:** the source prints $|\Delta_{\mathrm{BSM}}\mp\delta/100|=0$ “for puts and calls respectively”; taking those signs in that order conflicts with signed Black–Scholes deltas. The economically consistent interpretation is put delta $-\delta/100$ and call delta $+\delta/100$. This is a reconstruction clarification, not a silently corrected source formula. Interpolation family, interpolation/extrapolation boundaries, nearest-expiry/nearest-forward-strike tie-breaking, and missing-chain handling are not specified.

#### ATMF and per-delta levels

| Exact feature name or template | Target grid | Definition |
|---|---|---|
| `atmf_iv_{τ}dte_close` | $\tau\in\{1,5,20,30,60,90\}$ | Closing ATMF IV at the named DTE target. |
| `atmf_iv_1dte_1000` | 1 DTE | 10:00 ET ATMF IV; entry-time anchor. |
| `atmf_iv_5dte_1000` | 5 DTE | 10:00 ET ATMF IV; entry-time anchor. |
| `call_25d_iv_{τ}dte_close` | $\tau\in\{5,30\}$ | Closing 25-delta call IV. |
| `put_25d_iv_{τ}dte_close` | $\tau\in\{5,30\}$ | Closing 25-delta put IV. |

#### IV percentile

`atmf_iv_{τ}dte_percentile_252d`, for $\tau\in\{1,5,30\}$, is the fraction of the **past 252 closes** strictly below today's ATMF IV:

$$
\mathrm{IVPct}_{252,\tau,t}=\frac{1}{252}\sum_{i=1}^{252}
\mathbf 1\{\sigma_{\mathrm{ATM},\tau,t-i}<\sigma_{\mathrm{ATM},\tau,t}\}.
$$

The current observation is excluded from the reference sample; equal values do not count as below. The feature is a fraction, not a percentage multiplied by 100. The source does not specify incomplete-window handling.

#### Skew

All quantities in a row use the same maturity target and closing timestamp.

| Exact feature template | Target grid | Formula |
|---|---|---|
| `risk_reversal_delta25_{τ}dte_close` | $\tau\in\{1,5,10,20,30\}$ | $\sigma_{C25,\tau,t}-\sigma_{P25,\tau,t}$ |
| `risk_reversal_delta10_{τ}dte_close` | $\tau\in\{0,1,5,10,20,30\}$ | $\sigma_{C10,\tau,t}-\sigma_{P10,\tau,t}$ |
| `put_skew_delta25_{τ}dte_close` | $\tau\in\{1,5,20,30\}$ | $(\sigma_{\mathrm{ATM},\tau,t}-\sigma_{P25,\tau,t})/\sigma_{\mathrm{ATM},\tau,t}$ |
| `call_skew_delta25_{τ}dte_close` | $\tau\in\{1,5,20,30\}$ | $(\sigma_{\mathrm{ATM},\tau,t}-\sigma_{C25,\tau,t})/\sigma_{\mathrm{ATM},\tau,t}$ |

A more negative risk reversal means steeper downside skew: puts are more expensive than calls at the matching delta magnitude.

#### Forward volatility and realized-volatility surprise

Assuming additive total variance, for DTE targets $a<b$ with year fractions $t_i=\mathrm{DTE}_i/252$:

$$
\sigma_{a\to b}^{2}=\frac{\sigma^2(t_b)t_b-\sigma^2(t_a)t_a}{t_b-t_a},
\qquad
\sigma_{a\to b}=\sqrt{\sigma_{a\to b}^{2}}.
$$

The source displays the variance formula; taking its square root makes explicit the stated **volatility** feature.

| Exact feature template | Maturity-pair grid $(a,b)$ | IV inputs |
|---|---|---|
| `atmf_forward_vol_{a}d_{b}d_close` | $\{(5,10),(5,30),(10,20),(20,30),(30,60)\}$ | Closing ATMF IV at both legs. |
| `put_25d_forward_vol_{a}d_{b}d_close` | $\{(5,30),(10,20)\}$ | Closing 25-delta put IV at both legs; follows downside-skew forward volatility. |

**Source ambiguities:** no rule is given for negative implied forward variance; do not invent clipping to zero. The text also does not settle whether $t_a,t_b$ use the named target DTEs or the actual DTEs of the nearest selected expirations when those differ.

`rv_surprise_5d`, one of the three designated tail-risk features, compares five-day realized volatility with the five-day rolling mean of closing 5-DTE ATMF IV:

$$
\overline\sigma_{\mathrm{ATMF},5\mathrm{DTE},5d,t}
=\frac15\sum_{j=0}^{4}\sigma_{\mathrm{ATM},5,t-j},
\qquad
\mathrm{RVSurprise}_{5d,t}
=\frac{\mathrm{RV}_{5,t}-\overline\sigma_{\mathrm{ATMF},5\mathrm{DTE},5d,t}}
{\overline\sigma_{\mathrm{ATMF},5\mathrm{DTE},5d,t}}.
$$

The rolling-mean sum makes the stated five-day rolling comparator explicit. For self-containment, the paper's SPX realized-volatility convention is

$$
\mathrm{RV}_{5,t}=\sqrt{252}\sqrt{\frac15\sum_{i=1}^{5}
\left[\ln\left(\frac{P_{t-i+1}}{P_{t-i}}\right)\right]^2},
$$

where $P_t$ is the SPX close. This is an annualized root-mean-square log-return estimator, without subtracting a sample mean. Both sides of the surprise ratio are in annualized decimal volatility units.

#### IV-to-spot change ratio

`iv_to_spot_change_ratio_{τ}dte_close`, for $\tau\in\{0,1,5,10,20,30\}$, divides the daily percentage change in ATMF IV by the daily percentage change in the SPX close:

$$
\mathrm{IVSpotRatio}_{\tau,t}
=\frac{\Delta_{\%}\sigma_{\mathrm{ATM},\tau,t}}{\Delta_{\%}P_t}
=\frac{\sigma_{\mathrm{ATM},\tau,t}/\sigma_{\mathrm{ATM},\tau,t-1}-1}{P_t/P_{t-1}-1}.
$$

Consistent use of fractions or percentages gives the same ratio. Strongly negative values reflect IV rising when spot falls. Zero spot-return and other zero-denominator handling are unspecified.

#### Intraday ATMF IV dynamics

For every $\tau\in\{0,10,30\}$:

- `atmf_iv_{τ}dte_intraday_pct_change`: close-versus-09:35 ATMF IV percentage change, $\sigma_{\mathrm{ATM},\tau,t,\mathrm{close}}/\sigma_{\mathrm{ATM},\tau,t,09{:}35}-1$ in fractional-return notation. The source does not explicitly settle whether storage multiplies this value by 100.
- `atmf_iv_{τ}dte_intraday_range`: daily high-minus-low ATMF IV, $\max_h\sigma_{\mathrm{ATM},\tau,t,h}-\min_h\sigma_{\mathrm{ATM},\tau,t,h}$, where $h$ indexes the intraday observations. The source does not specify whether extrema use only the named snapshots or a denser intraday surface series.

Both become available at the close and must be lagged at least one trading day for 10:00 ET entry decisions.

### Realized-minus-implied volatility

Lag these daily-close-derived values at least one trading day.

| Feature | Definition |
|---|---|
| `rv_iv_spread_5d` | $\mathrm{RV}_5-\sigma_{\mathrm{ATMF},5}$ |
| `rv_iv_ratio_5d` | $\mathrm{RV}_5/\sigma_{\mathrm{ATMF},5}$ |
| `rv_iv_spread_21d` | $\mathrm{RV}_{21}-\sigma_{\mathrm{ATMF},20}$ |
| `rv_iv_ratio_21d` | $\mathrm{RV}_{21}/\sigma_{\mathrm{ATMF},20}$ |
| `iv_term_slope_5d_30d` | $\sigma_{\mathrm{ATMF},30}-\sigma_{\mathrm{ATMF},5}$ |
| `rv_term_slope_5d_21d` | $\mathrm{RV}_{21}-\mathrm{RV}_5$ |
| `iv_minus_rv_term_slope` | IV slope minus RV slope above |

IV subscripts are DTE targets; 20 DTE is the source's closest match to 21 trading days. The overview mentions a 63-day RV/IV spread family, but the detailed catalog provides only the entries above.

### Realized higher moments

`spx_realized_skew_21d`, `spx_realized_kurtosis_21d`, `spx_realized_skew_63d`, and `spx_realized_kurtosis_63d` are rolling sample skewness and excess (Fisher) kurtosis of daily SPX log returns over 21 and 63 days. Apply at least a one-day lag. The source does not specify the finite-sample estimator correction.

### VIX term curvature

Compute from spot-family closes and lag at least one trading day:

- `vix_curvature_9d_30d_3m`: $\mathrm{VIX9D}-2\mathrm{VIX}+\mathrm{VIX3M}$.
- `vix_curvature_1d_9d_30d`: $\mathrm{VIX1D}-2\mathrm{VIX9D}+\mathrm{VIX}$.
- `vix_curvature_1d_30d_3m`: $\mathrm{VIX1D}-2\mathrm{VIX}+\mathrm{VIX3M}$.

### Trend

With $\mathrm{MA}_{N,u}$ the trailing $N$-day simple moving average of SPX closes, compute then lag at least one day:

- `spx_distance_from_50dma_pct`: $(P_u-\mathrm{MA}_{50,u})/\mathrm{MA}_{50,u}$.
- `spx_distance_from_200dma_pct`: $(P_u-\mathrm{MA}_{200,u})/\mathrm{MA}_{200,u}$.
- `spx_50_200_dma_signal`: $\operatorname{sign}(\mathrm{MA}_{50,u}-\mathrm{MA}_{200,u})$.
- `spx_200dma_slope_21d_pct`: $\mathrm{MA}_{200,u}/\mathrm{MA}_{200,u-21}-1$.

### Position Greeks and exposures

At the entry snapshot, recover Black–Scholes Greeks using EFFR and no dividends. Let $K$ be the chosen strike, $S_{\mathrm{entry}}$ the SPX mid, and $P_{\mathrm{entry}}$ the option mid premium.

| Feature | Source definition |
|---|---|
| `delta`, `gamma`, `theta`, `vega` | Option Greeks $\Delta,\Gamma,\Theta,V$ at entry |
| `dollar_delta` | $100\lvert\Delta\rvert$ |
| `gamma_exposure` | $-100\lvert\Gamma\rvert S_{\mathrm{entry}}$ |
| `log_moneyness` | $\ln(K/S_{\mathrm{entry}})$ |
| `leverage` | $\lvert S_{\mathrm{entry}}\Delta/P_{\mathrm{entry}}\rvert$ |

These are the paper's explicit exposure definitions, regardless of the conventional meaning of “dollar delta” or “gamma exposure.” The paper does not fully specify theta/vega scaling or whether every stored Greek is option-signed versus position-signed. The sizing formulas use absolute gamma/delta where stated.

### Per-strategy rolling statistics (PS; source B.13)

Compute each candidate strategy's own realized history separately, then lag outcome-derived statistics by **one trading day** for use at entry. In the definitions below, $t$ is the day the trade outcome is observed: the feature used for entry on day $t$ is the statistic calculated through $t-1$. Do not apply an additional second lag to a feature already defined as prior-day.

Let $G_t$ be the candidate trade's gross P&L and $P_{\mathrm{entry},t}$ its entry premium, expressed in matching units. The paper defines

$$
\mathrm{ROM}_t=\frac{G_t}{P_{\mathrm{entry},t}}.
$$

**Preserve the denominator:** although the source calls this “return on margin” and a “notional-of-margin convention,” it explicitly divides by **entry premium**, not the Reg-T margin used in Kelly position sizing. These are distinct series. For the SKIP candidate, this section explicitly sets ROM to zero, even though the general feature convention says per-strategy features have no value for SKIP.

Define a trailing window $W_N(t)=\{t-N+1,\ldots,t\}$ and, for a generic series $x$,

$$
\overline x_{N,t}=\frac1N\sum_{i\in W_N(t)}x_i,
\qquad
s_{x,N,t}=\sqrt{\frac{1}{N-1}\sum_{i\in W_N(t)}(x_i-\overline x_{N,t})^2}.
$$

These make the stated rolling mean and sample-standard-deviation operations explicit. Let $P_t^{\mathrm{SPX}}$ be the SPX close, so $r_{1d,t}^{\mathrm{SPX}}=P_t^{\mathrm{SPX}}/P_{t-1}^{\mathrm{SPX}}-1$.

| Exact feature name | Definition at the outcome-observation date |
|---|---|
| `returns_on_margin` | $\mathrm{ROM}_t$. |
| `rom_diff_to_spx` | $\mathrm{ROM}_t-r_{1d,t}^{\mathrm{SPX}}$. |
| `rom_roll_avg_30d` | $\overline{\mathrm{ROM}}_{30,t}$. |
| `rom_roll_std_30d` | $s_{\mathrm{ROM},30,t}$, explicitly a sample standard deviation. |
| `rom_roll_skew_30d` | Skewness of $\{\mathrm{ROM}_i:i\in W_{30}(t)\}$; the source does not specify the finite-sample bias correction. |
| `sharpe_ratio_rom_60d` | $H_t=\overline{\mathrm{ROM}}_{60,t}/\sigma_{\mathrm{ROM},60,t}$, a **daily**, unannualized rolling Sharpe. The denominator is the rolling ROM standard deviation; this bullet does not specify its degrees of freedom. No risk-free subtraction is stated. |
| `sharpe_ratio_rom_std_60d` | $s_{H,60,t}$, the 60-day rolling **sample** standard deviation of the rolling Sharpe series itself. This is a nested window. |
| `win_rate_30d` | $\frac1{30}\sum_{i\in W_{30}(t)}\mathbf1\{\mathrm{ROM}_i>0\}$; zero returns are not wins. |

#### Cumulative-gross-P&L drawdowns and distances

Let

$$
C_t=\sum_{i\le t}G_i,
\qquad
C^{\max}_{N,t}=\max_{i\in W_N(t)} C_i.
$$

The source computes these on **cumulative gross P&L**, not on an equity curve including initial capital:

$$
D_{N,t}=\frac{C^{\max}_{N,t}-C_t}{C^{\max}_{N,t}},
\qquad
A_{N,t}=C_t-C^{\max}_{N,t}.
$$

| Exact feature name or template | Window grid | Definition |
|---|---|---|
| `drawdown_{N}` | $N\in\{63,126,252\}$ | $D_{N,t}$. The source prints this template without a `d` suffix. |
| `distance_from_max_{N}` | $N\in\{63,126,252\}$ | $A_{N,t}$ in dollars. |
| `prior_day_strategy_drawdown_max_5d` | 5 trading days | $D_{5,t-1}$ for the row dated $t$, by the stated five-day rolling drawdown definition. One of the three designated tail-risk features. |

**Source discrepancies and omissions:** “absolute distance” means a dollar-unit distance in the prose, but the displayed formula $C_t-C^{\max}_{N,t}$ is signed and nonpositive; preserve that formula rather than taking its absolute value. Cumulative P&L may have a zero or negative rolling maximum, and the paper specifies no handling for those denominators. It does not specify the cumulative-series starting point. The prior-day feature's name includes `max`, but its prose specifies five-day rolling drawdown, not a second maximum over previously computed drawdowns.

#### Stability and tails

For `stability_coef_60d`, take Pearson correlation between the ROM values and a within-window time index $j\in\{1,\ldots,60\}$, then square it:

$$
\mathrm{StabCoef}_{60,t}=\left[\operatorname{corr}\left(
(\mathrm{ROM}_{t-59},\ldots,\mathrm{ROM}_t),(1,\ldots,60)
\right)\right]^2.
$$

The source describes this as a smoothness proxy, with values near one indicating steadily increasing ROM and values near zero erratic ROM. **Formula qualification:** squaring removes the sign, so steadily decreasing ROM can also produce a value near one.

For `tail_ratio_rom_30d`, use absolute values around both tail quantiles:

$$
\mathrm{TailRatio}_{30,t}
=\frac{\left|Q_{0.95}(\{\mathrm{ROM}_i:i\in W_{30}(t)\})\right|}
{\left|Q_{0.05}(\{\mathrm{ROM}_i:i\in W_{30}(t)\})\right|}.
$$

$Q_p$ is the empirical $p$ quantile. Quantile interpolation, zero lower-tail denominator, and constant-series correlation handling are unspecified.

#### Within-day rank variants

Rank each named source across the eight delta-bucket candidates on the same day. The source's general convention assigns SKIP the rank **8.5**, below all eight alternatives; it does not specify rank orientation or tie treatment.

| Exact feature name | Ranked source |
|---|---|
| `iv_at_strike_minus_atmf_wd_rank` | `iv_at_strike_minus_atmf`: selected-strike entry IV minus ATMF IV at the same DTE. |
| `returns_on_margin_wd_rank` | `returns_on_margin`. |
| `sharpe_ratio_rom_60d_wd_rank` | `sharpe_ratio_rom_60d`. |
| `drawdown_63d_wd_rank` | `drawdown_63d`, as spelled in the source. |

The `drawdown_63d` rank source conflicts with the earlier printed `drawdown_{N}` template; resolve the canonical name before coding. Outcome-derived ranks must use lagged available statistics. The IV-difference rank is listed inside this lagged-statistics section although its source is an entry-time IV feature; the paper's general timing table makes the underlying entry-time term-context source available at 10:00 ET, so the rank's intended lag is ambiguous. Incomplete rolling-window behavior and missing candidate-history treatment are not specified.

### Entry liquidity

At 10:00 ET use `entry_bid_ask_spread` $=\mathrm{ask}-\mathrm{bid}$, `entry_bid_ask_spread_pct` $=(\mathrm{ask}-\mathrm{bid})/\mathrm{mid}$, and `entry_log_premium` $=\ln(\mathrm{mid})$. The `_wd_rank` variants are `entry_bid_ask_spread_pct_wd_rank` and `entry_log_premium_wd_rank`, with the catalog's within-day ranking convention. These are unlagged entry observations.

### Intra-strategy term context

At entry, unlagged:

- `delta_distance_from_target`: realized entry delta minus target delta, signed. The universe specifies positive *absolute* delta targets; the catalog does not reconcile the sign convention of this subtraction.
- `iv_at_strike_minus_atmf`: selected-strike IV minus ATMF IV at the same DTE.
- `iv_at_strike_minus_atmf_pct`: that difference divided by ATMF IV.
- `dte_of_selected_option`: selected option's DTE. The catalog says typically zero with one- or two-day fallback; the sampled-universe discussion says at most one session.

### Regime-conditional sensitivities

Each feature is the named PS source multiplied by the named CS source (or DTE). Source Table B.1 requires at least a one-trading-day lag on this block.

| Feature | Product |
|---|---|
| `delta_x_VIX` | $\Delta\,\mathrm{VIX}$ |
| `gamma_x_morning_spx_rv_annualized` | $\Gamma\sigma_{\mathrm{morn}}$ |
| `vega_x_VIX_5d_pct_change` | $V$ times five-day VIX percentage change |
| `iv_at_strike_minus_atmf_x_VIX_VIX9D_spread` | Strike-minus-ATMF IV times VIX-minus-VIX9D |
| `theta_x_atmf_iv_5dte_percentile_252d` | $\Theta$ times five-DTE IV percentile |
| `delta_distance_from_target_x_VVIX_1y_percentile` | Delta-target gap times VVIX percentile |
| `gamma_x_dte` | $\Gamma$ times selected-option DTE |

## Regime-conditional verification tables

WF covers 2021–2024; OOT is calendar 2025, held out from training, model selection, and hyperparameter search. Annual returns and annual volatility below are fractions. Days counts every day assigned to the regime, whereas Trades counts only days carrying a position. Compute returns once on the ordered slice, attribute each return to the day it was earned, then partition by regime. Include abstention and gate-blocked days with zero returns; do not recompute returns from discontinuous subsets of prices.

For entry-day closing VIX $v_t$, the VIX regime is

$$
R_t^{\mathrm{VIX}}=
\begin{cases}
\mathrm{low}, & v_t<15,\\
\mathrm{mid}, & 15\leq v_t\leq25,\\
\mathrm{high}, & v_t>25.
\end{cases}
$$

For the S&P 500's 5-day intraday realized volatility $x_t$ measured at entry-day close, let $q_{1,s}$ and $q_{2,s}$ be the empirical $1/3$ and $2/3$ quantiles computed from all observations in slice $s$ (WF or OOT), separately:

$$
q_{1,s}=Q_{1/3}(\{x_t:t\in s\}),\qquad
q_{2,s}=Q_{2/3}(\{x_t:t\in s\}).
$$

Tercile 1 is the lowest band below $q_{1,s}$, tercile 2 is between $q_{1,s}$ and $q_{2,s}$, and tercile 3 is the highest band above $q_{2,s}$. **Not specified:** empirical-quantile interpolation and allocation of observations exactly equal to a cut point. Do not silently choose either when reproducing counts. Equal-quantile partitioning does not imply equal reported day counts: WF has 322/320/322 and OOT 80/79/78. A tercile label spans different numeric volatility levels across WF and OOT.

Both regime axes use entry-day closing information, which arrives **after the 10:00 ET entry** and is unavailable to the trading decision. These are retrospective performance classifications, not implementable entry signals. The paper describes cells with fewer than about thirty days as too imprecise to interpret; this is the paper's own assessment of its small-sample cells, not a new trading filter. In particular, OOT low/high VIX contain 21/20 days.

### Headline performance by VIX regime — full source Table D.1

| Regime | Method | Split | Sharpe | Ann. return | Ann. vol. | Days | Trades |
|---|---|---|---:|---:|---:|---:|---:|
| low | EA | WF | 0.2611 | 0.0050 | 0.0191 | 223 | 61 |
| mid | EA | WF | 4.2354 | 0.1390 | 0.0328 | 593 | 337 |
| high | EA | WF | 2.7675 | 0.1540 | 0.0557 | 148 | 53 |
| low | EA | OOT | 13.7060 | 0.0549 | 0.0040 | 21 | 15 |
| mid | EA | OOT | 4.9387 | 0.0912 | 0.0185 | 196 | 187 |
| high | EA | OOT | 14.4589 | 0.3100 | 0.0214 | 20 | 18 |
| low | FMU | WF | 2.6184 | 0.0417 | 0.0159 | 223 | 68 |
| mid | FMU | WF | 4.1695 | 0.1901 | 0.0456 | 593 | 355 |
| high | FMU | WF | 0.5503 | 0.0527 | 0.0958 | 148 | 56 |
| low | FMU | OOT | 85.6325 | 0.1469 | 0.0017 | 21 | 21 |
| mid | FMU | OOT | 5.6371 | 0.1578 | 0.0280 | 196 | 196 |
| high | FMU | OOT | 5.3274 | 0.4568 | 0.0857 | 20 | 20 |
| low | SRS | WF | 1.7529 | 0.0319 | 0.0182 | 223 | 68 |
| mid | SRS | WF | 4.4101 | 0.1763 | 0.0400 | 593 | 355 |
| high | SRS | WF | 1.0494 | 0.0867 | 0.0826 | 148 | 56 |
| low | SRS | OOT | 63.9948 | 0.1258 | 0.0020 | 21 | 21 |
| mid | SRS | OOT | 5.8867 | 0.1454 | 0.0247 | 196 | 196 |
| high | SRS | OOT | 6.9412 | 0.5862 | 0.0845 | 20 | 20 |
| low | VT | WF | 5.4640 | 0.0726 | 0.0133 | 223 | 64 |
| mid | VT | WF | 3.8811 | 0.2084 | 0.0537 | 593 | 349 |
| high | VT | WF | 0.2859 | 0.0380 | 0.1329 | 148 | 54 |
| low | VT | OOT | 88.7880 | 0.2575 | 0.0029 | 21 | 21 |
| mid | VT | OOT | 5.5988 | 0.2621 | 0.0468 | 196 | 193 |
| high | VT | OOT | 0.1657 | 0.0164 | 0.0993 | 20 | 20 |
| low | GB | WF | 6.6201 | 0.0535 | 0.0081 | 223 | 68 |
| mid | GB | WF | 4.4769 | 0.1944 | 0.0434 | 593 | 355 |
| high | GB | WF | 0.3317 | 0.0481 | 0.1451 | 148 | 56 |
| low | GB | OOT | 83.2863 | 0.2378 | 0.0029 | 21 | 21 |
| mid | GB | OOT | 5.5620 | 0.2632 | 0.0473 | 196 | 196 |
| high | GB | OOT | 5.5906 | 0.8065 | 0.1443 | 20 | 20 |
| low | HK | WF | 1.8744 | 0.0366 | 0.0195 | 223 | 55 |
| mid | HK | WF | 3.6559 | 0.1725 | 0.0472 | 593 | 302 |
| high | HK | WF | 0.3839 | 0.0473 | 0.1232 | 148 | 55 |
| low | HK | OOT | 55.5100 | 0.1898 | 0.0034 | 21 | 20 |
| mid | HK | OOT | 4.6566 | 0.1756 | 0.0377 | 196 | 175 |
| high | HK | OOT | 5.3496 | 0.6196 | 0.1158 | 20 | 20 |
| low | QK | WF | 1.8152 | 0.0377 | 0.0208 | 223 | 55 |
| mid | QK | WF | 3.5873 | 0.1712 | 0.0477 | 593 | 302 |
| high | QK | WF | 0.3349 | 0.0419 | 0.1250 | 148 | 55 |
| low | QK | OOT | 55.3857 | 0.1846 | 0.0033 | 21 | 20 |
| mid | QK | OOT | 4.8011 | 0.1684 | 0.0351 | 196 | 174 |
| high | QK | OOT | 4.8847 | 0.5697 | 0.1166 | 20 | 20 |

### Headline performance by realized-volatility tercile — full source Table D.2

| Tercile | Method | Split | Sharpe | Ann. return | Ann. vol. | Days | Trades |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | EA | WF | 2.7711 | 0.0965 | 0.0348 | 322 | 167 |
| 2 | EA | WF | 2.9750 | 0.0886 | 0.0298 | 320 | 147 |
| 3 | EA | WF | 3.5503 | 0.1416 | 0.0399 | 322 | 137 |
| 1 | EA | OOT | 1.5499 | 0.0411 | 0.0265 | 80 | 77 |
| 2 | EA | OOT | 13.9158 | 0.1025 | 0.0074 | 79 | 72 |
| 3 | EA | OOT | 12.2625 | 0.1767 | 0.0144 | 78 | 71 |
| 1 | FMU | WF | 2.6665 | 0.1021 | 0.0383 | 322 | 176 |
| 2 | FMU | WF | 3.6109 | 0.1273 | 0.0352 | 320 | 157 |
| 3 | FMU | WF | 2.2680 | 0.1689 | 0.0745 | 322 | 146 |
| 1 | FMU | OOT | 4.8249 | 0.1032 | 0.0214 | 80 | 80 |
| 2 | FMU | OOT | 3.1131 | 0.1159 | 0.0372 | 79 | 79 |
| 3 | FMU | OOT | 7.7448 | 0.3362 | 0.0434 | 78 | 78 |
| 1 | SRS | WF | 2.4899 | 0.0838 | 0.0336 | 322 | 176 |
| 2 | SRS | WF | 3.8341 | 0.1168 | 0.0305 | 320 | 157 |
| 3 | SRS | WF | 2.8145 | 0.1838 | 0.0653 | 322 | 146 |
| 1 | SRS | OOT | 4.8857 | 0.0864 | 0.0177 | 80 | 80 |
| 2 | SRS | OOT | 3.2211 | 0.1052 | 0.0327 | 79 | 79 |
| 3 | SRS | OOT | 8.2180 | 0.3567 | 0.0434 | 78 | 78 |
| 1 | VT | WF | 2.8840 | 0.1353 | 0.0469 | 322 | 171 |
| 2 | VT | WF | 5.2927 | 0.1786 | 0.0337 | 320 | 153 |
| 3 | VT | WF | 1.3061 | 0.1322 | 0.1012 | 322 | 143 |
| 1 | VT | OOT | 5.0290 | 0.1787 | 0.0355 | 80 | 80 |
| 2 | VT | OOT | 3.3516 | 0.2135 | 0.0637 | 79 | 78 |
| 3 | VT | OOT | 6.5120 | 0.3312 | 0.0509 | 78 | 76 |
| 1 | GB | WF | 3.6686 | 0.0917 | 0.0250 | 322 | 176 |
| 2 | GB | WF | 6.3623 | 0.1657 | 0.0260 | 320 | 157 |
| 3 | GB | WF | 1.4276 | 0.1558 | 0.1091 | 322 | 146 |
| 1 | GB | OOT | 4.7558 | 0.1685 | 0.0354 | 80 | 80 |
| 2 | GB | OOT | 2.9237 | 0.1872 | 0.0640 | 79 | 79 |
| 3 | GB | OOT | 8.1170 | 0.5881 | 0.0725 | 78 | 78 |
| 1 | HK | WF | 2.2623 | 0.0901 | 0.0398 | 322 | 160 |
| 2 | HK | WF | 3.7739 | 0.1232 | 0.0326 | 320 | 127 |
| 3 | HK | WF | 1.5791 | 0.1474 | 0.0933 | 322 | 125 |
| 1 | HK | OOT | 4.5273 | 0.1289 | 0.0285 | 80 | 77 |
| 2 | HK | OOT | 2.8110 | 0.1416 | 0.0504 | 79 | 73 |
| 3 | HK | OOT | 6.3287 | 0.3752 | 0.0593 | 78 | 65 |
| 1 | QK | WF | 2.3980 | 0.0948 | 0.0396 | 322 | 160 |
| 2 | QK | WF | 3.6736 | 0.1216 | 0.0331 | 320 | 127 |
| 3 | QK | WF | 1.4710 | 0.1398 | 0.0951 | 322 | 125 |
| 1 | QK | OOT | 8.1524 | 0.1469 | 0.0180 | 80 | 77 |
| 2 | QK | OOT | 2.5104 | 0.1271 | 0.0506 | 79 | 72 |
| 3 | QK | OOT | 5.6502 | 0.3370 | 0.0596 | 78 | 65 |

The OOT low-VIX maximum Sharpe is VT 88.7880 (rounded to 88.8 in the regime-plot caption); GB is 83.2863 (rounded to 83.29 in the prose). These are different methods, not conflicting estimates.

## References

Complete bibliography as supplied in the paper; incomplete source entries remain incomplete.

- Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, pages 2623–2631, New York, NY, USA. Association for Computing Machinery.

- Andersen, T. G., Fusari, N., and Todorov, V. (2015). The risk premia embedded in index options. Journal of Financial Economics, 117(3):558–584.

- Andersen, T. G., Fusari, N., and Todorov, V. (2017). Short-Term Market Risks Implied by Weekly Options. The Journal of Finance, 72(3):1335–1386.

- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. Journal of Risk, 15(2).

- Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. The Journal of Portfolio Management, 40(5):94– 107.

- Bakshi, G. and Kapadia, N. (2003). Delta-Hedged Gains and the Negative Market Volatility Risk Premium. The Review of Financial Studies, 16(2):527–566.

- Barak, S., Mousavi, A., and Hosseini, S. A. (2025). Deep Reinforcement Learning for Dynamic Learn to Rank: A Risk-Aware Framework for Cryptocurrency.

- Battalio, R., Shkilko, A., and Van Ness, R. (2016). To Pay or Be Paid? The Impact of Taker Fees and Order Flow Inducements on Trading Costs in U.S. Options Markets. The Journal of Financial and Quantitative Analysis, 51(5):1637–1662.

- Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3):637–54.

- Black, K. and Szado, E. (2022). 35-Year Performance Analysis of Cboe S&P 500 Option-Selling Indices. Journal of Beta Investment Strategies, 13(3).

- Bollerslev, T., Gibson, M., and Zhou, H. (2011). Dynamic estimation of volatility risk premia and investor risk aversion from option-implied and realized volatilities. Journal of Econometrics, 160(1).

- Bondarenko, O. (2014). Why Are Put Options So Expensive? The Quarterly Journal of Finance, 04(03):1450015.

- Bondarenko, O. (2019). Historical Performance of Put-Writing Strategies.

- Božović, M. (2025). Intraday Jumps and 0DTE Options: Pricing and Hedging Implications.

- Broadie, M., Chernov, M., and Johannes, M. (2009). Understanding Index Option Returns. The Review of Financial Studies, 22(11):4493–4529.

- Burges, C. J. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. Technical Report MSR-TR-2010-82, Microsoft.

- Burrello, J., Fabozzi, F. J., Liang, H., Sood, A., and Vatanen, K. (2024). Applications of stock index options for income enhancement. Journal of Asset Management, 25(6):579–588.

- Carr, P. and Wu, L. (2009). Variance Risk Premiums. The Review of Financial Studies, 22(3):1311– 1341.

- Chicago Board Options Exchange (2022). Cboe to Add Tuesday and Thursday Expirations for SPX Weeklys Options. Press release, Cboe Global Markets.

- Constantinides, G. M., Jackwerth, J. C., and Savov, A. (2013). The Puzzle of Index Option Returns. The Review of Asset Pricing Studies, 3(2):229–257.

- Cont, R. and da Fonseca, J. (2002). Dynamics of implied volatility surfaces. Quantitative Finance, 2(1):45–60.

- Corradi, V., Distaso, W., and Mele, A. (2013). Macroeconomic determinants of stock volatility and volatility premiums. Journal of Monetary Economics, 60(2):203–220.

- Da Fonseca, J. and Xu, Y. (2019). Variance and skew risk premiums for the volatility market: The VIX evidence. Journal of Futures Markets, 39(3):302–321.

- Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3):253–263.

- Do, B. H., Foster, A., and Gray, P. (2016). The Profitability of Volatility Spread Trading on ASX Equity Options. Journal of Futures Markets, 36(2):107–126.

- Duan, J. and Kashima, H. (2021). Learning to Rank for Multi-Step Ahead Time-Series Forecasting. IEEE Access, 9:49372–49386.

- Faias, J. A. and Santa-Clara, P. (2017). Optimal Option Portfolio Strategies: Deepening the Puzzle of Index Option Mispricing. Journal of Financial and Quantitative Analysis, 52(1):277–303.

- Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 4885–4894, Red Hook, NY, USA. Curran Associates Inc.

- Gunnarsson, E. S., Isern, H. R., Kaloudis, A., Risstad, M., Vigdel, B., and Westgaard, S. (2024). Prediction of realized volatility and implied volatility indices using AI and machine learning: A review. International Review of Financial Analysis, 93:103221.

- Hong, H., Sung, H.-C., and Yang, J. (2018). On profitability of volatility trading on S&P 500 equity index options: The role of trading frictions. International Review of Economics & Finance, 55:295–307.

- Huang, S.-C., Chiou, C.-C., Chiang, J.-T., and Wu, C.-F. (2020). A novel intelligent option price forecasting and trading system by multiple kernel adaptive filters. Journal of Computational and Applied Mathematics, 369:112560.

- Israelov, R. and Klein, M. (2016). Risk and Return of Equity Index Collar Strategies. The Journal of Alternative Investments, 19(1):41–54.

- Israelov, R. and Tummala, H. (2017). Which Index Options Should You Sell?

- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

- Kelly Jr., J. L. (1956). A New Interpretation of Information Rate. Bell System Technical Journal, 35(4):917–926.

- Linger, M., Mellouki, I., and Boulanger, G. (2025a). Unifying Asset Ranking and Portfolio Weighting through a Multi-Task Neural Network. The Journal of Financial Data Science, 7(2):84–104.

- Linger, M., Metz, T., Sbai, K., and Boulanger, G. (2025b). Enhancing Long–Short Portfolios: A Refined Approach Using Learn-to-Rank Algorithms. The Journal of Financial Data Science, 7(1):76–97.

- Malkiel, B. G., Rinaudo, A., and Saha, A. (2018). Option Writing: Using VIX to Improve Returns. The Journal of Derivatives, 26(2):38–49.

- Natenberg, S. (2015). Option Volatility and Pricing, chapter Chapter 20: Volatility Revisited. McGraw-Hill Education, New York, 2nd edition.

- Nogueira, S., Sechidis, K., and Brown, G. (2018). On the Stability of Feature Selection Algorithms. Journal of Machine Learning Research, 18(174):1–54.

- Patel, P., Raquel, A., and Chadwick, S. (2024). The cash-secured put-write strategy and the variance risk premium. Journal of Asset Management, 25(1):31–50.

- Poh, D., Lim, B., Zohren, S., and Roberts, S. (2021). Building Cross-Sectional Systematic Strategies by Learning to Rank. The Journal of Financial Data Science, 3(2):70–86.

- Santa-Clara, P. and Saretto, A. (2009). Option strategies: Good deals and margin calls. Journal of Financial Markets, 12(3):391–417.

- Schwalbach, J. B. M. and McClelland, D. (2018). An analysis of short put strategies and their role in asset allocation. Investment Analysts Journal, 47(3):272–283.

- Sharpe, W. F. (1994). The Sharpe Ratio. The Journal of Portfolio Management, 21(1):49–58.

- Sheu, H.-J. and Wei, Y.-C. (2011). Effective options trading strategies based on volatility forecasting recruiting investor sentiment. Expert Systems with Applications, 38(1):585–596.

- Song, Q., Liu, A., and Yang, S. Y. (2017). Stock portfolio selection using learning-to-rank algorithms with news sentiment. Neurocomputing, 264:20–28.

- Sortino, F. A. and Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. The Journal of Investing, 3(3):59–64.

- Tannous, G. F. and Lee-Sing, C. (2008). Expected Time Value Decay of Options: Implications for Put-Rolling Strategies. Financial Review, 43(2):191–218.

- Thorp, E. O. (2008). The kelly criterion in blackjack sports betting, and the stock market. In Handbook of Asset and Liability Management, pages 385–428. North-Holland, San Diego.

- Thorp, E. O. (2011). Understanding the Kelly Criterion. In The Kelly Capital Growth Investment Criterion, volume Volume 3 of World Scientific Handbook in Financial Economics Series, pages 509–523. World Scientific.

- Ungar, J. and Moran, M. T. (2009). The Cash-secured PutWrite Strategy and Performance of Related Benchmark Indexes. The Journal of Alternative Investments, 11(4):43–56.

- Vasquez, A. (2017). Equity Volatility Term Structures and the Cross Section of Option Returns. Journal of Financial and Quantitative Analysis, 52(6):2727–2754.

- Vasquez, A., Amaya, D., Pearson, N. D., and Garcia-Ares, P. A. (2025). 0DTE Index Options and Market Volatility: How Large is Their Impact?

- Whaley, R. E. (2002). Return and Risk of CBOE Buy Write Monthly Index. The Journal of Derivatives, 10(2):35–42.

- Wu, M.-C., Huang, S.-H., and Chen, A.-P. (2024). Momentum portfolio selection based on learning-to-rank algorithms with heterogeneous knowledge graphs. Applied Intelligence, 54(5):4189–4209.

- Zhang, X., Wu, L., and Chen, Z. (2022). Constructing long-short stock portfolio with a new listwise learn-to-rank algorithm. Quantitative Finance, 22(2):321–331.

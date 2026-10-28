# Theory notes

The methods behind the forecasts, in the order they are used. Code references point to
`src/wpforecast/`; decisions to [DECISIONS.md](DECISIONS.md).

## 1. Recursive least squares as a Kalman filter

The power curve is linear in its parameters, $y_t = \varphi_t^\top \beta_t + \varepsilon_t$,
with $\varphi_t$ the B-spline × direction basis and diurnal terms (64 values,
`models/power_curve.py`). Let the parameters drift as a random walk,
$\beta_t = \beta_{t-1} + w_t$. The Kalman filter for this state-space model is recursive least
squares (RLS):

$$K_t = \frac{P_{t-1}\varphi_t}{\lambda + \varphi_t^\top P_{t-1}\varphi_t},\qquad
\hat\beta_t = \hat\beta_{t-1} + K_t\,(y_t - \varphi_t^\top\hat\beta_{t-1}),\qquad
P_t = \frac{(I - K_t\varphi_t^\top)P_{t-1}}{\lambda}.$$

$P$ is proportional to the covariance of $\hat\beta$. Dividing by the forgetting factor
$\lambda \le 1$ inflates it at every step, which is what a random walk with
$\mathrm{Cov}(w_t) \propto (1/\lambda - 1)P$ does. Old observations then lose weight like
$\lambda^{\text{age}}$, so the memory is about $1/(1-\lambda)$ rows: λ = 0.9999 is about
10,000 rows, or 270 days at ~37 rows a day.

- **Start.** $P_0 = \delta I$ is a Gaussian prior $\beta \sim N(0, \delta\sigma^2 I)$. With
  λ = 1 the recursion therefore ends exactly at ridge regression with penalty $1/\delta$,
  which the frozen and walk-forward models use (unit-tested).
- **Windup.** With λ < 1, directions the data never excite (rare speed × direction cells)
  inflate without bound. The covariance is bounded per direction, $P \le \delta I$, by
  clipping its eigenvalues at δ once a day (Salgado, Goodwin & Middleton, 1988). Scaling all
  of P to a trace bound instead also shrinks the gain of the observed directions and stops
  the tracking (D-043).
- **Covariance resetting.** At a known break, P returns to $\delta I$ and β stays as the
  prior mean, so new data move β fast. A full reset forgets all 64 directions; Step 9 shows
  that it over-reacts.
- **Short horizons.** The error of the curve at the last known hour,
  $e_t = p_t - \varphi_t^\top\hat\beta$, persists. The point forecast
  $\hat p_{t+k|t} = \varphi_{t+k}^\top\hat\beta + a_k e_t$ estimates $a_k$ per horizon block
  by the same weighted least squares (WPPT, Nielsen et al.).

## 2. Quantile regression as a linear program

The τ-quantile minimises the expected pinball loss
$\rho_\tau(u) = u\,(\tau - \mathbb 1\{u<0\})$, with $u = y - q$. For a linear model
$q_\tau = x^\top b$ (here $x = [B(\hat p),\, B(\hat p)\,k/K]$, a B-spline in the point
forecast and its interaction with the horizon), the empirical problem is a linear program
(Koenker & Bassett, 1978):

$$\min_{b,\,u,\,v}\ \tau\,\mathbf 1^\top u + (1-\tau)\,\mathbf 1^\top v
\quad\text{s.t.}\quad Xb + u - v = y,\ \ u, v \ge 0.$$

Its dual has n variables in a box and only p constraints, and HiGHS solves it ~15× faster:

$$\max_{a}\ y^\top a\quad\text{s.t.}\quad X^\top a = (1-\tau)\,X^\top \mathbf 1,\ \ 0 \le a \le 1.$$

The coefficients b are the dual's constraint multipliers (`models/probabilistic.py`,
`quantile_lp`). The consequences:

- **Interpolation.** An optimum interpolates p data points exactly and can be non-unique, so
  IRLS and HiGHS agree on the loss and differ only in degenerate directions.
- **Identification.** A basis function needs data under it. Knots at quantiles of the
  training p̂ ensure that; uniform knots over [0, 1] left the top functions almost without
  data, and storm hours got nonsense quantiles (D-042).
- **Grid.** Each level is a separate fit, so quantiles can cross. They are fitted at 23
  levels, interpolated in τ to the 99-level grid and sorted.

## 3. Censored generalised logit-normal likelihood

Normalised power lives in [0, 1] and has point masses at its bounds: calm hours at 0, full
output near 1. The generalised logit (Pinson, 2012)

$$z = g_\nu(y) = \log\frac{y^\nu}{1 - y^\nu}$$

maps (0, 1) to the real line, and the model assumes $z \sim N(\mu, \sigma^2)$ with

$$\mu = \theta_0 + \theta_1 g_\nu(\hat p) + \theta_2 k/K,\qquad
\log\sigma = \theta_3 + \theta_4 g_\nu(\hat p) + \theta_5 g_\nu(\hat p)^2 + \theta_6 k/K
\ (+\ \theta_7 s),$$

where $s$ is the spread of the ensemble members' power (Step 8). Observations at or beyond
$\varepsilon$ or $1-\varepsilon$ are censored, so they enter the likelihood through
probabilities, not densities:

$$\ell(\theta) = \sum_{\text{inner}} \log\phi\!\left(\tfrac{z_i-\mu_i}{\sigma_i}\right) - \log\sigma_i
\;+\; \sum_{y_i \le \varepsilon} \log\Phi\!\left(\tfrac{g_\nu(\varepsilon)-\mu_i}{\sigma_i}\right)
\;+\; \sum_{y_i \ge 1-\varepsilon} \log\left[1-\Phi\!\left(\tfrac{g_\nu(1-\varepsilon)-\mu_i}{\sigma_i}\right)\right].$$

The Jacobian of $g_\nu$ does not depend on θ, so it drops out of the maximisation (L-BFGS-B,
monthly refits). The quantiles are $q_\tau = g_\nu^{-1}(\mu + \sigma\Phi^{-1}(\tau))$, set to 0
or 1 beyond the censoring points. The predictive distribution therefore carries probability
mass at both bounds, which a Gaussian on y cannot.

## 4. The single-price newsvendor

A producer sells a day-ahead bid b at $\lambda^{DA}$ and settles $E - b$ at the imbalance
price. Against a perfect bid, the cost is
$C = (\lambda^{DA} - \lambda^{imb})(E - b)$ (`market.py`).

- **Dual price.** A short MWh costs $\psi_\uparrow \ge 0$ and a long MWh $\psi_\downarrow \ge 0$.
  The expected cost $\psi_\uparrow \mathbb E(b-E)^+ + \psi_\downarrow \mathbb E(E-b)^+$ has the
  first-order condition $F(b) = \psi_\downarrow/(\psi_\uparrow+\psi_\downarrow)$. Bid that
  quantile (Bremnes, 2004; Pinson et al., 2007).
- **Single price** (DK1 since 2025-03-19). One price for both directions makes C linear in b:
  $\partial\,\mathbb E C/\partial b = \mathbb E[\lambda^{imb} - \lambda^{DA}]$, a constant.
  There is no interior optimum, only a corner that bets on the sign of the mean spread
  (Bruninx et al., 2025); balance rules forbid such bets.
- **Through correlation.** An asymmetry returns only if one's own error co-moves with the
  system. Write $s = \lambda^{DA} - \lambda^{imb}$ and $e = E - b$ over n hours:

$$\sum s\,e = \underbrace{\sum (s-\bar s)(e-\bar e)}_{\text{covariance}} + \underbrace{n\,\bar s\,\bar e}_{\text{bet}}.$$

  The effective penalties $\hat c_\uparrow = \mathbb E[\lambda^{imb}-\lambda^{DA}\mid \text{short}]$
  and $\hat c_\downarrow = \mathbb E[\lambda^{DA}-\lambda^{imb}\mid \text{long}]$ give
  $\tau = \hat c_\downarrow / (\hat c_\uparrow + \hat c_\downarrow)$. For a portfolio shaped
  like the DK1 fleet, the covariance part dominates: a good forecast shares the market's
  errors, so it pays when the system is out of balance in the same direction.

## 5. Scoring

- **CRPS.** The continuous ranked probability score is twice the pinball loss integrated over
  levels, $\mathrm{CRPS}(F, y) = 2\int_0^1 \rho_\tau(y - F^{-1}(\tau))\,d\tau$. It is estimated
  from the 99 quantiles; for a point forecast it reduces to the absolute error.
- **Calibration.** It is judged separately from CRPS, by reliability (observed against
  nominal frequency) and the randomised PIT. The aim is maximal sharpness subject to
  calibration (Gneiting, Balabdaoui & Raftery, 2007).
- **Significance.** Diebold–Mariano tests use daily mean score differences with a Newey–West
  variance, and bootstrap bands resample whole days, because errors within a day are
  strongly dependent.

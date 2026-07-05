# BLP Demand Estimation in MATLAB

Nevo (2000) canonical replication — BLP (1995) random-coefficients logit,
estimated by nested fixed point (NFP) GMM.

## Quick Start

1. Run `main.m` in MATLAB. The Nevo fake-cereal dataset is included
   in `data/` — no download required.
2. (Optional) Cross-check against PyBLP:

   ```bash
   pip install pyblp
   python verify_blp.py
   ```

Both implementations load the same CSV files (including the simulation
draws `nodes0`–`nodes3`), so any gap between them is algorithmic — not
simulation noise.

## Reference Solution (golden values)

Step-1 GMM with W = (Z'Z)⁻¹, starting from Nevo's published estimates.
These match the PyBLP tutorial replication of Nevo (2000).

| Parameter | Value |
|---|---|
| alpha (price) | −62.7299 |
| sigma_p | 3.3125 |
| pi_p_inc | 588.325 |
| pi_p_incSq | −30.192 |
| pi_p_child | 11.055 |
| Mean own-price elasticity | −3.618 |
| GMM objective | 0.0020220 |

## File Structure

| File | Description |
|---|---|
| `main.m` | Main script — full pipeline from data loading to supply side |
| `obj_ftn.m` | GMM objective (1/N)·ξ′ZWZ′ξ |
| `mean_utility.m` | BLP contraction mapping (inner loop, exp space) |
| `nlin_mu.m` | Nonlinear utility deviation μ_ijt |
| `ind_mkt_share.m` | Individual choice probabilities (cumsum trick, no market loop) |
| `tsls_est.m` | 2SLS with HC0 robust SEs and first-stage F |
| `share_deriv_rc.m` | ∂s_j/∂p_k matrix (RC logit, single market) |
| `elas_rc.m` / `elas_logit.m` | Price elasticity matrices (RC / plain logit benchmark) |
| `markup_rc.m` / `markup_logit.m` | Bertrand–Nash markups |
| `mc_rc.m` / `mc_logit.m` | Marginal costs (p − markup) |
| `solve_eqm.m` | Counterfactual equilibrium prices (e.g. post-merger, logit demand) |
| `verify_blp.py` | Cross-validation against PyBLP |
| `data/` | Nevo (2000) fake-cereal data in PyBLP CSV format |

## Model

### Utility

Consumer $i$'s indirect utility from product $j$ in market $t$ (a city–quarter pair):

```math
u_{ijt} = \underbrace{x^{(1)\prime}_{jt}\theta_1 + \xi_{jt}}_{\delta_{jt}\ \text{(mean utility)}} + \underbrace{\sum_{k} x^{(2)}_{jt,k}\left(\sigma_k \nu_{ik} + \pi_k' D_i\right)}_{\mu_{ijt}\ \text{(individual deviation)}} + \varepsilon_{ijt}
```

where $\varepsilon_{ijt}$ is i.i.d. type-I extreme value and the outside
good is normalized to $u_{i0t} = \varepsilon_{i0t}$. Each random
coefficient decomposes as
$\beta_{ik} = \beta_k + \sigma_k \nu_{ik} + \pi_k' D_i$: the mean $\beta_k$
lives in $\delta_{jt}$ (for sugar, mushy and the constant, absorbed by the
brand fixed effects inside $x^{(1)}_{jt}$), the deviations in $\mu_{ijt}$.

Because sugar, mushy and the constant do not vary within a brand, they are
perfectly collinear with the brand fixed effects: each estimated FE absorbs
their mean tastes together with the brand's average unobserved quality
$\bar{\xi}_j$,

```math
\mathrm{FE}_j = \beta_0 + \beta_s \text{ sugar}_j + \beta_m \text{ mushy}_j + \bar{\xi}_j
```

The means $(\beta_0, \beta_s, \beta_m)$ can be recovered after estimation
by Chamberlain minimum distance — GLS of the estimated FEs on the brand
characteristics, weighted by the inverse covariance of the FE estimates
(planned extension; requires the full GMM covariance of the FEs).

### Notation

| Symbol | Meaning | In the code |
|---|---|---|
| $j = 1,\dots,24$ | products (cereal brands) | `product_id` |
| $t = 1,\dots,94$ | markets (city × quarter) | `mktIdx` |
| $i = 1,\dots,20$ | simulated consumers per market | `S`, `const.n_sim` |
| $x_{jt}$ | observed product characteristics (price, sugar, mushy, const) | `prod` columns |
| $x^{(1)}_{jt}$ | linear characteristics: price + brand FE | `x1` |
| $x^{(2)}_{jt}$ | RC characteristics: $[1,\ p_{jt},\ \text{sugar},\ \text{mushy}]$ | `x2` (+ const) |
| $\xi_{jt}$ | unobserved demand shock, correlated with price → IV needed | `xi` |
| $s_{jt}$ | observed market share | `data.share` |
| $\delta_{jt}$ | mean utility, common across consumers | `deltaHat` |
| $\mu_{ijt}$ | individual utility deviation | `nlin_mu.m` |
| $\nu_{ik}$ | standard-normal RC draws (one per $x^{(2)}$ char) | `nodes0`–`nodes3` |
| $D_i$ | demographics: income, income², age, child | agent columns |
| $\theta_1 = (\alpha,\ \text{brand FE})$ | linear parameters (25×1), recovered by 2SLS | `theta1Hat` |
| $\theta_2 = (\sigma,\ \pi)$ | nonlinear parameters (13×1), searched by `fminsearch` | `theta2Hat` |
| $Z$ | instruments: const + brand FE + 20 excluded IVs (44 cols) | `Z` |
| $W$ | GMM weight matrix $(Z'Z)^{-1}$ | `data.invW` |

**Sign convention (PyBLP):** $\mu_{ijt} = +(\pi'D_i + \sigma\nu_i)\cdot x_{jt}$,
so parameters compare to PyBLP output directly, with no sign flips.

### Random-coefficient specification

$X^{(2)} = [1,\ p_{jt},\ \text{sugar},\ \text{mushy}]$, each with its own
$\sigma_k$; demographics interact following Nevo's Table I pattern:

| $X^{(2)}$ characteristic | $\sigma$ | Interacted demographics ($\pi$) |
|---|---|---|
| constant | $\sigma_c$ | income, age |
| price | $\sigma_p$ | income, income², child |
| sugar | $\sigma_s$ | income, age |
| mushy | $\sigma_m$ | income, age |

Income² enters price only and age never interacts with price — this
breaks the income/income² collinearity (corr ≈ 0.995) that makes an
all-on-price specification weakly identified.

Parameter vector (order used throughout the code):

```
theta2 = [sigma_c, pi_c_inc, pi_c_age,                          % constant
          sigma_p, pi_p_inc, pi_p_incSq, pi_p_child,            % price
          sigma_s, pi_s_inc, pi_s_age,                          % sugar
          sigma_m, pi_m_inc, pi_m_age]                          % mushy
```

### Market shares and inversion

Predicted share of product $j$ in market $t$, integrating over the $S$
simulated consumers:

```math
s_{jt}(\delta_t;\theta_2) = \frac{1}{S}\sum_{i=1}^{S} \frac{\exp\left(\delta_{jt} + \mu_{ijt}\right)}{1 + \sum_{k \in \mathcal{J}_t} \exp\left(\delta_{kt} + \mu_{ikt}\right)}
```

Given $\theta_2$, the mean utilities solve
$s_{jt}(\delta_t;\theta_2) = s_{jt}^{\text{obs}}$ via the BLP contraction
(`mean_utility.m`), iterated in exp space until convergence:

```math
\exp\left(\delta^{(r+1)}\right) = \exp\left(\delta^{(r)}\right)\cdot\frac{s^{\text{obs}}}{s\left(\delta^{(r)};\theta_2\right)}
```

### GMM estimation (nested fixed point)

The structural error is recovered as
$\xi_{jt}(\theta_2) = \delta_{jt}(\theta_2) - x^{(1)\prime}_{jt}\theta_1(\theta_2)$,
and estimation exploits the moment condition $E[Z_{jt}' \xi_{jt}] = 0$:

```math
\hat\theta_2 = \arg\min_{\theta_2} \frac{1}{N}\ \xi(\theta_2)' Z W Z' \xi(\theta_2), \qquad W = (Z'Z)^{-1}
```

The outer loop (`fminsearch`) searches over $\theta_2$; for each candidate,
the inner loop (`mean_utility`) inverts market shares, and $\theta_1$ is
concentrated out by 2SLS of $\delta$ on $x^{(1)}$ (price instrumented by
the 20 excluded IVs). A 2-step efficient GMM skeleton with
$W = (Z'\hat\xi\hat\xi'Z)^{-1}$ is included (Section 14, commented out).

### Post-estimation

Price elasticity matrices by market, a plain-logit benchmark (IIA
comparison), and Bertrand–Nash markups / marginal costs from the
multi-product FOC p = mc + Δ⁻¹s with ownership built from `firm_ids`.

## References

- Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices in
  Market Equilibrium. *Econometrica*, 63(4), 841–890.
- Nevo, A. (2000). A Practitioner's Guide to Estimation of
  Random-Coefficients Logit Models of Demand. *Journal of Economics &
  Management Strategy*, 9(4), 513–548.
- Conlon, C., & Gortmaker, J. (2020). Best Practices for Differentiated
  Products Demand Estimation with PyBLP. *RAND Journal of Economics*,
  51(4), 1108–1161.

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

Utility of consumer *i* for product *j* in market *t*:

    u_ijt = x_jt·beta_i + xi_jt + eps_ijt

- **X1 (linear):** price + brand fixed effects
  (constant, sugar and mushy are absorbed by the FEs)
- **X2 (random coefficients):** [1, price, sugar, mushy] — draws `nodes0`–`nodes3`
- **Demographic interactions** (Nevo's Table I pattern):

  | X2 characteristic | Interacted demographics |
  |---|---|
  | constant | income, age |
  | price | income, income², child |
  | sugar | income, age |
  | mushy | income, age |

  Income² enters price only and age never interacts with price — this
  breaks the income/income² collinearity (corr ≈ 0.995) that makes an
  all-on-price specification weakly identified.

- **Sign convention (PyBLP):** μ_ijt = +(π·D_i + σ·ν_i)·x_jt, so parameters
  compare to PyBLP output directly, with no sign flips.
- **theta2 (13×1):** `[sigma_c, pi_c_inc, pi_c_age, sigma_p, pi_p_inc,
  pi_p_incSq, pi_p_child, sigma_s, pi_s_inc, pi_s_age, sigma_m, pi_m_inc,
  pi_m_age]`

### Estimation

Nested fixed point: the outer loop (`fminsearch`) searches over theta2;
for each candidate, the inner loop (`mean_utility`) inverts market shares
via the BLP contraction, theta1 is concentrated out by 2SLS, and the GMM
objective is evaluated on ξ = δ − X1·theta1. A 2-step efficient GMM
skeleton is included (Section 14, commented out).

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

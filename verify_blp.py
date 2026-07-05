"""
verify_blp.py — Cross-check the MATLAB template (nevo_spec.m) with PyBLP.

Data: PyBLP-style CSVs (data/nevo_products.csv, data/nevo_agents.csv) —
the same files the MATLAB template loads, so demographics and simulation
draws (nodes0-3) match exactly: any gap between the two implementations
is algorithmic, not simulation noise.

Specification (Nevo 2000, canonical):
  X1 = price + brand FE (absorbed);  X2 = [1, prices, sugar, mushy]
  pi pattern: const x (income, age); price x (income, income^2, child);
              sugar x (income, age); mushy x (income, age)
  Both implementations use the pyblp sign convention mu = +(pi·D + sigma·nu)·x,
  so parameters compare directly (no sign flips).

MATLAB reference solution (nevo_spec.m, step-1 GMM, fminsearch):
  alpha = -62.7299   sigma_p = 3.3125
  pi_p_inc = 588.325   pi_p_incSq = -30.192   pi_p_child = 11.055
  mean own-price elasticity = -3.618

Run:  python verify_blp.py          (from the repo root)
      pip install pyblp              (if not yet installed)
"""

import os

import numpy as np
import pandas as pd
import pyblp

pyblp.options.digits = 4
pyblp.options.verbose = False
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # ensure relative paths work

product_data = pd.read_csv('data/nevo_products.csv')
agent_data   = pd.read_csv('data/nevo_agents.csv')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logit benchmark (theta2 = 0)
#    Matches MATLAB: tsls_est(lnShare, x1, Z)  (nevo_spec.m Section 12)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("1. LOGIT 2SLS  (price + brand FE)")
print("=" * 65)

logit_form    = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
results_logit = pyblp.Problem(logit_form, product_data).solve()
print(results_logit)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BLP RC Logit — Nevo's canonical specification
#    Starting values = Nevo's published estimates (as in the pyblp tutorial).
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("2. BLP RC LOGIT  (Nevo specification)")
print("=" * 65)

blp_formulations = (
    pyblp.Formulation('0 + prices', absorb='C(product_ids)'),   # X1
    pyblp.Formulation('1 + prices + sugar + mushy'),            # X2
)
agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')

problem = pyblp.Problem(
    blp_formulations,
    product_data,
    agent_formulation,
    agent_data,
)

init_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
init_pi = np.array([
    [ 5.4819,  0,       0.2037,  0     ],    # const:  income, age
    [15.8935, -1.2000,  0,       2.6342],    # price:  income, income^2, child
    [-0.2506,  0,       0.0511,  0     ],    # sugar:  income, age
    [ 1.2650,  0,      -0.8091,  0     ],    # mushy:  income, age
])

results = problem.solve(
    sigma=init_sigma,
    pi=init_pi,
    optimization=pyblp.Optimization('l-bfgs-b', {'gtol': 1e-8}),
    method='1s',    # 1-step GMM with the 2SLS-style W — matches MATLAB §6
)
print(results)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Summary vs MATLAB reference
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY  (PyBLP vs MATLAB nevo_spec.m reference)")
print("=" * 65)

matlab_ref = {
    'alpha': -62.7299,
    'sigma_p': 3.3125, 'pi_p_inc': 588.3251,
    'pi_p_incSq': -30.1920, 'pi_p_child': 11.0546,
    'mean_own_elas': -3.6181,
}

elasticities = results.compute_elasticities()
mean_own = float(np.mean(results.extract_diagonals(elasticities)))

rows = [
    ('alpha (price)', float(results.beta[0]),        matlab_ref['alpha']),
    ('sigma_p',       float(results.sigma[1, 1]),    matlab_ref['sigma_p']),
    ('pi_p_inc',      float(results.pi[1, 0]),       matlab_ref['pi_p_inc']),
    ('pi_p_incSq',    float(results.pi[1, 1]),       matlab_ref['pi_p_incSq']),
    ('pi_p_child',    float(results.pi[1, 3]),       matlab_ref['pi_p_child']),
    ('mean own elas', mean_own,                      matlab_ref['mean_own_elas']),
]
print(f"  {'':16s}  {'PyBLP':>10}   {'MATLAB':>10}")
for name, py_val, ml_val in rows:
    print(f"  {name:16s}  {py_val:>10.4f}   {ml_val:>10.4f}")

print("\nNote: both use the same data, draws and step-1 W; remaining gaps")
print("reflect optimizer stopping points on a common objective.")

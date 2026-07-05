% =========================================================================
%  BLP Demand Estimation — Nevo (2000) Canonical Replication Template
%
%  Data format: PyBLP-style CSV (data/nevo_products.csv, nevo_agents.csv)
%
%  Model (pyblp sign convention:  mu = +(pi·D + sigma·nu)·x):
%    X1 = [const, price, brand FE]   — sugar/mushy/const absorbed by FE
%    X2 = [1, price, sugar, mushy]   — RC on all four (nodes0-3)
%    pi pattern (Nevo Table I):
%      const: income, age | price: income, income^2, child
%      sugar: income, age | mushy: income, age
%
%  theta2 (13 x 1) = [sigma_c, pi_c_inc, pi_c_age,
%                     sigma_p, pi_p_inc, pi_p_incSq, pi_p_child,
%                     sigma_s, pi_s_inc, pi_s_age,
%                     sigma_m, pi_m_inc, pi_m_age]
%
%  Reference solution (step-1 GMM; = pyblp tutorial replication of Nevo):
%    alpha = -62.7299   sigma_p = 3.3125
%    pi_p_inc = 588.325   pi_p_incSq = -30.192   pi_p_child = 11.055
%    mean own-price elasticity = -3.618   objective = 0.0020220
%
%  Convention
%    functions (names & locals) : snake_case
%    script variables           : camelCase
%    ID variables               : *_id
%    loop indices               : t (market), j (product), i (consumer)
% =========================================================================

clear; clc; close all;
global counter exp_delta_init

%% ── 1. Load Data ──────────────────────────────────────────────────────────

prod  = readtable('data/nevo_products.csv');
agent = readtable('data/nevo_agents.csv');

%% ── 2. Sort and Extract Product-Level Variables ──────────────────────────
% Sort table first (ind_mkt_share cumsum trick needs consecutive market rows),
% then extract — no per-variable re-sorting needed.

prod  = sortrows(prod,  {'market_ids', 'product_ids'});
agent = sortrows(agent, 'market_ids');

market_id  = prod.market_ids;
product_id = prod.product_ids;
firm_id    = prod.firm_ids;
brand_id   = prod.brand_ids;
share      = prod.shares;
price      = prod.prices;
sugar      = prod.sugar;
mushy      = prod.mushy;

ivCols = prod.Properties.VariableNames( ...
    startsWith(prod.Properties.VariableNames, 'demand_instruments'));
iv = table2array(prod(:, ivCols));   % [N x 20]

[mktLabels, ~, mktIdx] = unique(market_id, 'stable');   % [N x 1]  1..T
T = max(mktIdx);

%% ── 3. Dimensions ────────────────────────────────────────────────────────

Jt         = accumarray(mktIdx, 1);   % [T x 1]  products per market ({J_t})
mktLastRow = cumsum(Jt);              % [T x 1]  row index of last product in market t
N          = height(prod);
F          = length(unique(firm_id));

% S: simulated consumers per market (inferred from agent file)
agentCnt = accumarray(findgroups(agent.market_ids), 1);   % [T x 1]
assert(all(agentCnt == agentCnt(1)), ...
    'agent file must have the same number of consumers in every market');
S = agentCnt(1);

fprintf('T=%d  max(J_t)=%d  F=%d  N=%d  S=%d\n', T, max(Jt), F, N, S);

%% ── 4. Logit Shares ──────────────────────────────────────────────────────

insideShare = accumarray(mktIdx, share);          % [T x 1]
s0          = 1 - insideShare(mktIdx);            % [N x 1] outside-good share
lnShare     = log(share ./ s0);                   % [N x 1] logit LHS

%% ── 5. Regressors and Instruments ───────────────────────────────────────
% Brand (product) fixed effects absorb the constant, sugar and mushy;
% only price remains as a non-FE linear regressor (Nevo's X1).
% tsls_est treats column 2 as endogenous → x1 = [cons, price, FE(2:end)].

cons = ones(N, 1);

[prodLabels, ~, prodIdx] = unique(product_id, 'stable');
prodFE = full(sparse((1:N)', prodIdx, 1, N, max(prodIdx)));   % [N x 24]

x1 = [cons, price, prodFE(:, 2:end)];   % [N x 25]  const + price + 23 FE
x2 = [price, sugar, mushy];             % (constant RC handled in nlin_mu)
Z  = [cons, prodFE(:, 2:end), iv];      % [N x 44]  exogenous + 20 excluded IVs

%% ── 6. Weight Matrix W ───────────────────────────────────────────────────
% Step-1 GMM: W = (Z'Z)^{-1}  (standard 2SLS weight matrix)
% For 2-step efficient GMM: update W = inv(Z'*xi*xi'*Z) after step-1
% converges (see Section 14 below).

W = inv(Z' * Z);

%% ── 7. Spread Agent Variables  →  [N x S] ───────────────────────────────
% nodes0-3 map to X2 = [1, price, sugar, mushy] in pyblp column order.
% Pack agent columns into [S*T x 9], reshape to [S x T x 9], then index
% each slice by mkt_id to get [N x S] per variable.

agentArr = table2array(agent(:, ...
    {'nodes0','nodes1','nodes2','nodes3', ...
     'income','income_squared','age','child','weights'}));
agentArr = reshape(agentArr, S, T, []);   % S x T x 9
agentExp = agentArr(:, mktIdx, :);        % S x N x 9

nu_c     = agentExp(:,:,1)';   % [N x S]  RC draw: constant
nu_p     = agentExp(:,:,2)';   %           price
nu_s     = agentExp(:,:,3)';   %           sugar
nu_m     = agentExp(:,:,4)';   %           mushy
income   = agentExp(:,:,5)';
incomeSq = agentExp(:,:,6)';
age      = agentExp(:,:,7)';
child    = agentExp(:,:,8)';
weight   = agentExp(:,:,9)';

%% ── 8. Data and Const Structs ───────────────────────────────────────────

data.share    = share;
data.x1       = x1;
data.x2       = x2;
data.Z        = Z;
data.invW     = W;
data.nu_c     = nu_c;
data.nu_p     = nu_p;
data.nu_s     = nu_s;
data.nu_m     = nu_m;
data.income   = income;
data.incomeSq = incomeSq;
data.age      = age;
data.child    = child;
data.weight   = weight;

const.mkt_id    = mktIdx;
const.mkt_index = mktLastRow;
const.n_sim     = S;

%% ── 9. BLP Estimation ───────────────────────────────────────────────────
% Start from Nevo's published estimates (pyblp tutorial starting values).

init = [0.3302,  5.4819,  0.2037, ...            % const:  sigma, inc, age
        2.4526, 15.8935, -1.2000, 2.6342, ...    % price:  sigma, inc, incSq, child
        0.0163, -0.2506,  0.0511, ...            % sugar:  sigma, inc, age
        0.2441,  1.2650, -0.8091];               % mushy:  sigma, inc, age

exp_delta_init = exp(lnShare);   % logit delta as contraction starting point

objFun  = @(theta2) obj_ftn(theta2, data, const);
% Nelder-Mead in 13 dims needs generous limits.
options = optimset('TolFun', 1e-6, 'TolX', 1e-8, ...
                   'MaxIter', 10000, 'MaxFunEvals', 20000);
counter = 0;
[theta2Hat, fval, flag] = fminsearch(objFun, init, options);
fprintf('\nOptimizer flag: %d   Objective: %.8f\n', flag, fval);

%% ── 10. Recover theta1 at Converged theta2 ──────────────────────────────

[deltaHat, shareIHat] = mean_utility(theta2Hat, data, const);
res       = tsls_est(deltaHat, x1, Z);
theta1Hat = res.beta;
xi        = deltaHat - x1 * theta1Hat;

fprintf('\n=== theta1 (price; brand FE omitted) ===\n');
fprintf('  price (alpha)  %8.4f  (%.4f)\n', theta1Hat(2), res.se(2));
fprintf('  First-stage F: %.1f\n', res.first_F);

fprintf('\n=== theta2 (Nevo spec) ===          start    estimate\n');
labels2 = {'sigma_c','pi_c_inc','pi_c_age', ...
           'sigma_p','pi_p_inc','pi_p_incSq','pi_p_child', ...
           'sigma_s','pi_s_inc','pi_s_age', ...
           'sigma_m','pi_m_inc','pi_m_age'};
for k = 1:13
    fprintf('  %-12s  %9.4f  %9.4f\n', labels2{k}, init(k), theta2Hat(k));
end

%% ── 11. Price Elasticities (average across markets) ─────────────────────

etaAll = cell(T, 1);   % [J_t x J_t] elasticity matrix per market
ownEta = zeros(N, 1);  % own-price elasticity per product-market row
for t = 1:T
    rows        = find(mktIdx == t);
    dm.x2       = data.x2(rows, :);
    dm.share    = data.share(rows);
    dm.income   = data.income(rows, :);
    dm.incomeSq = data.incomeSq(rows, :);
    dm.child    = data.child(rows, :);
    dm.nu_p     = data.nu_p(rows, :);
    etaAll{t}   = elas_rc(theta1Hat, theta2Hat, dm, shareIHat(rows,:), const);
    ownEta(rows) = diag(etaAll{t});
end

% Full-matrix average is only defined when every market has the same
% products in the same order (true for Nevo data: 24 brands x 94 markets).
if all(Jt == Jt(1))
    etaAvg = mean(cat(3, etaAll{:}), 3);   % [J x J] averaged over markets
end

[brandList, ~, brandIdx] = unique(brand_id, 'stable');
ownEtaAvg = accumarray(brandIdx, ownEta) ./ accumarray(brandIdx, 1);

fprintf('\n=== Average Own-Price Elasticities ===\n');
for j = 1:length(brandList)
    fprintf('  brand %2d:  %7.4f\n', brandList(j), ownEtaAvg(j));
end
fprintf('  mean over all products: %7.4f\n', mean(ownEta));

%% ── 12. Logit Benchmark (theta2 = 0) ────────────────────────────────────
% With theta2 = 0, delta has the closed form ln(s_j/s_0) — no contraction
% needed. Cross-price elasticities collapse to -alpha*p_k*s_k (IIA).

resLogit    = tsls_est(lnShare, x1, Z);
theta1Logit = resLogit.beta;

ownEtaLogit = zeros(N, 1);
for t = 1:T
    rows      = find(mktIdx == t);
    dmL.x2    = data.x2(rows, :);
    dmL.share = data.share(rows);
    etaLogit  = elas_logit(theta1Logit, dmL);
    ownEtaLogit(rows) = diag(etaLogit);
end
ownEtaLogitAvg = accumarray(brandIdx, ownEtaLogit) ./ accumarray(brandIdx, 1);

fprintf('\n=== Own-Price Elasticities: RC vs Logit (avg across markets) ===\n');
fprintf('  %-8s  %8s  %8s\n', 'brand', 'RC', 'logit');
for j = 1:length(brandList)
    fprintf('  brand %2d  %8.4f  %8.4f\n', ...
        brandList(j), ownEtaAvg(j), ownEtaLogitAvg(j));
end

%% ── 13. Supply Side: Markups and Marginal Costs (Bertrand-Nash) ─────────
% Multi-product Bertrand FOC:  p = mc + Delta^{-1} s,
%   Delta(j,k) = -ownership(j,k) * ds_j/dp_k,  ownership from firm_ids.
% For counterfactual equilibrium prices (e.g. post-merger), see solve_eqm.m.

markup = zeros(N, 1);
for t = 1:T
    rows = find(mktIdx == t);
    dm.x2       = data.x2(rows, :);
    dm.share    = data.share(rows);
    dm.income   = data.income(rows, :);
    dm.incomeSq = data.incomeSq(rows, :);
    dm.child    = data.child(rows, :);
    dm.nu_p     = data.nu_p(rows, :);
    ownership   = double(firm_id(rows) == firm_id(rows)');
    markup(rows) = markup_rc(theta1Hat, theta2Hat, dm, shareIHat(rows,:), ...
                             const, ownership);
end
mc     = price - markup;
lerner = markup ./ price;

fprintf('\n=== Markups and Marginal Costs by Brand (avg across markets) ===\n');
fprintf('  %-8s  %8s  %8s  %8s  %8s\n', 'brand', 'price', 'markup', 'mc', 'lerner');
avgP  = accumarray(brandIdx, price)  ./ accumarray(brandIdx, 1);
avgMu = accumarray(brandIdx, markup) ./ accumarray(brandIdx, 1);
avgMc = accumarray(brandIdx, mc)     ./ accumarray(brandIdx, 1);
avgL  = accumarray(brandIdx, lerner) ./ accumarray(brandIdx, 1);
for j = 1:length(brandList)
    fprintf('  brand %2d  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
        brandList(j), avgP(j), avgMu(j), avgMc(j), avgL(j));
end

%% ── 14. 2-Step Efficient GMM (uncomment to run) ─────────────────────────

% m          = Z .* xi;
% data.invW  = inv(m' * m);
% counter    = 0;
% exp_delta_init = exp(deltaHat);          % warm-start from step-1 solution
% [theta2_2s, fval_2s] = fminsearch(objFun, theta2Hat, options);


% =========================================================================
%  Function files (same directory):
%    obj_ftn.m        — GMM objective
%    mean_utility.m   — BLP contraction mapping
%    nlin_mu.m        — nonlinear utility deviation mu_ijt
%    ind_mkt_share.m  — individual choice probabilities (cumsum trick)
%    tsls_est.m       — 2SLS with HC0 robust SE
%    share_deriv_rc.m — ds_j/dp_k matrix (RC logit, single market)
%    elas_rc.m        — price elasticity matrix (RC logit)
%    elas_logit.m     — price elasticity matrix (plain logit benchmark)
%    markup_rc.m / markup_logit.m — Bertrand-Nash markups
%    mc_rc.m / mc_logit.m         — marginal costs (p - markup)
%    solve_eqm.m      — counterfactual equilibrium prices (logit demand)
%
%  Verification:
%    verify_blp.py    — cross-check against PyBLP (pip install pyblp)
% =========================================================================

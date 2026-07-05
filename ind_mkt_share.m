function F = ind_mkt_share(exp_delta, exp_mu, const)
% Individual choice probabilities P(i chooses j | market)  [N x n_sim]
%
% cumsum trick: avoids a market loop by using cumulative sums and
% differencing at market boundaries to get within-market denominators.
% Requires data sorted by market (products of t=1 first, t=2 next, ...).

    mkt_id    = const.mkt_id;
    mkt_index = const.mkt_index;

    numerator = exp_delta .* exp_mu;          % [N x n_sim]  exp(delta + mu)

    cs               = cumsum(numerator, 1);  % [N x n_sim]  running sum down rows
    mkt_cs           = cs(mkt_index, :);      % [T x n_sim]  cumsum at market boundaries
    mkt_cs(2:end, :) = diff(mkt_cs, 1, 1);    % [T x n_sim]  within-market sum of numerators
    denominator      = 1 + mkt_cs(mkt_id, :); % [N x n_sim]  1 + Σ_j exp(delta_j + mu_ij)

    F = numerator ./ denominator;
end

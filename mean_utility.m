function [delta, share_tilde_ijt] = mean_utility(theta2, data, const)
% BLP contraction mapping in exp space:
%   exp(delta)^{r+1} = exp(delta)^r * s_obs / s_sim(exp(delta)^r, theta2)
%
% Tolerance ramps from 1e-6 (outer iters <= 25) to 1e-13 thereafter,
% saving computation during early exploration without sacrificing precision.

    global counter exp_delta_init

    exp_mu = exp(nlin_mu(theta2, data, const));   % [N x n_sim]

    tol = 1e-13;
    if counter <= 25, tol = 1e-6; end

    exp_delta     = zeros(size(data.share));
    exp_delta_new = exp_delta_init;
    iter          = 0;

    while sum(abs(exp_delta_new - exp_delta)) >= tol && iter < 1000
        exp_delta       = exp_delta_new;
        iter            = iter + 1;
        share_tilde_ijt = ind_mkt_share(exp_delta, exp_mu, const);   % [N x n_sim]
        share_tilde_jt  = mean(share_tilde_ijt, 2);                  % [N x 1]
        exp_delta_new   = exp_delta .* data.share ./ share_tilde_jt;
    end

    delta           = log(exp_delta_new);
    share_tilde_ijt = ind_mkt_share(exp_delta_new, exp_mu, const);
end

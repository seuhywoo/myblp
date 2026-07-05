function p_new = solve_eqm(theta1, mc, p_orig, share_orig, ownership_new, tol, max_iter)
% Solve for counterfactual (e.g. post-merger) equilibrium prices, logit demand.
%
% Fixed-point iteration on the Bertrand-Nash FOC: p = mc + markup(p).
% Shares are recomputed at each iteration using the logit formula.
%
% Inputs:
%   theta1          - linear parameters [const; alpha; ...]
%   mc              - [J x 1] marginal costs
%   p_orig          - [J x 1] original (pre-merger) prices
%   share_orig      - [J x 1] original market shares
%   ownership_new   - [J x J] counterfactual ownership matrix
%   tol             - convergence tolerance (default 1e-12)
%   max_iter        - max iterations (default 1000)
%
% Output:
%   p_new           - [J x 1] counterfactual equilibrium prices

    if nargin < 6, tol = 1e-12; end
    if nargin < 7, max_iter = 1000; end

    alpha = theta1(2);

    % Back out the non-price component of delta:
    %   delta_orig = log(s_j / s_0) = delta_noprice + alpha * p_orig
    s0_orig       = 1 - sum(share_orig);
    delta_orig    = log(share_orig / s0_orig);
    delta_noprice = delta_orig - alpha * p_orig;

    p_new = p_orig;   % start from pre-merger prices

    for iter = 1:max_iter
        p_old = p_new;

        % Recompute logit shares at current prices
        exp_delta = exp(delta_noprice + alpha * p_new);
        share     = exp_delta / (1 + sum(exp_delta));

        % FOC: p = mc + markup at current shares
        dm_iter.share = share;
        p_new = mc + markup_logit(theta1, dm_iter, ownership_new);

        if max(abs(p_new - p_old)) < tol
            fprintf('solve_eqm converged in %d iterations.\n', iter);
            return
        end
    end

    fprintf('solve_eqm: max iterations (%d) reached. Max price diff = %.2e\n', ...
            max_iter, max(abs(p_new - p_old)));
end

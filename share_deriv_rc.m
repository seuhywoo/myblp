function D = share_deriv_rc(theta1, theta2, dm, share_i, const)
% [J x J] matrix of d(s_j)/d(p_k) for the RC logit model.
%   D(j,j) =  E_i[ alpha_i * s_ij * (1 - s_ij) ]   (< 0, alpha_i < 0)
%   D(j,k) = -E_i[ alpha_i * s_ij * s_ik ]         (j ~= k)
%
% alpha_i = theta1(2) + sigma_p*nu_p + pi_p_inc*income
%         + pi_p_incSq*incomeSq + pi_p_child*child     (pyblp sign convention)
%
% dm must contain rows for a single market only.

    n_sim = const.n_sim;

    sigma_p    = theta2(4);  pi_p_inc   = theta2(5);
    pi_p_incSq = theta2(6);  pi_p_child = theta2(7);

    income   = dm.income(1,   1:n_sim);   % [1 x n_sim]  same for all j in this market
    incomeSq = dm.incomeSq(1, 1:n_sim);
    child    = dm.child(1,    1:n_sim);
    nu_p     = dm.nu_p(1,     1:n_sim);

    % Full (mean + deviation) price coefficient for each consumer  [1 x n_sim]
    alpha_i = theta1(2) + sigma_p*nu_p + pi_p_inc*income ...
            + pi_p_incSq*incomeSq + pi_p_child*child;

    s_i = share_i;          % [J x n_sim]
    J   = size(s_i, 1);

    D = zeros(J);
    for j = 1:J
        for k = 1:J
            if j == k
                D(j,k) =  mean(alpha_i .* s_i(j,:) .* (1 - s_i(j,:)));
            else
                D(j,k) = -mean(alpha_i .* s_i(j,:) .* s_i(k,:));
            end
        end
    end
end

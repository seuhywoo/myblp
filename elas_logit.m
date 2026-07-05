function E = elas_logit(theta1, dm)
% [J x J] price elasticity matrix for a single market (plain logit).
%
%   Own-price:   E(j,j) =  alpha * p_j * (1 - s_j)
%   Cross-price: E(j,k) = -alpha * p_k * s_k        [same for all j — IIA]
%
% dm must contain rows for a single market only.

    alpha = theta1(2);      % price coefficient
    price = dm.x2(:, 1);    % [J x 1]
    share = dm.share;       % [J x 1]
    J     = length(share);

    % Cross-price: column k = -alpha * s_k * p_k  (identical down each column)
    E = -alpha * (ones(J,1) * (share .* price)');

    % Own-price: override diagonal
    E(1:J+1:end) = alpha .* price .* (1 - share);
end

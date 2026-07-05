function E = elas_rc(theta1, theta2, dm, share_i, const)
% [J x J] price elasticity matrix for a single market (RC logit).
%   E(j,k) = d(s_j)/d(p_k) * p_k / s_j
%
% Diagonal:     E_i[ alpha_i * s_ij * (1 - s_ij) ] * p_j / s_j   (< 0)
% Off-diagonal: -E_i[ alpha_i * s_ij * s_ik ] * p_k / s_j         (> 0)
%
% dm must contain rows for a single market only.

    D     = share_deriv_rc(theta1, theta2, dm, share_i, const);
    price = dm.x2(:, 1);    % [J x 1]
    share = dm.share;       % [J x 1]

    E = D .* (price' ./ share);   % [J x J]
end

function res = tsls_est(Y, X, Z)
% 2SLS with HC0 robust standard errors.
%   Column 2 of X is treated as endogenous.
%   Z contains exogenous regressors + excluded instruments.
%
% res.beta, res.se, res.alpha (price coeff), res.se_alpha, res.first_F

    ZtZ  = Z' * Z;
    Xhat = Z * (ZtZ \ (Z' * X));   % [N x k]  first-stage fitted values
    b    = (Xhat' * X) \ (Xhat' * Y);
    e    = Y - X * b;

    meat = (Xhat .* e)' * (Xhat .* e);
    XhX  = Xhat' * X;
    V    = XhX \ meat / XhX;

    res.beta     = b;
    res.se       = sqrt(diag(V));
    res.alpha    = b(2);
    res.se_alpha = res.se(2);

    % First-stage F for excluded instruments
    n        = size(X, 1);
    n_inst   = size(Z, 2) - (size(X, 2) - 1);
    gamma_fs = ZtZ \ (Z' * X(:, 2));
    ehat_fs  = X(:, 2) - Z * gamma_fs;
    Zexcl    = Z(:, end-n_inst+1:end);
    F_num    = (gamma_fs(end-n_inst+1:end)' * (Zexcl'*Zexcl) * ...
                gamma_fs(end-n_inst+1:end)) / n_inst;
    F_den    = (ehat_fs' * ehat_fs) / (n - size(Z, 2));
    res.first_F = F_num / F_den;
end

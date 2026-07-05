function mu = markup_rc(theta1, theta2, dm, share_i, const, ownership)
% Markup vector for the RC logit model (multi-product Bertrand-Nash).
% FOC: p = mc + mu,  where  Delta * mu = s,  Delta(j,k) = -ownership(j,k) * D(j,k)
%
% ownership(j,k) = 1 if products j and k are owned by the same firm.
% dm must contain rows for a single market only.

    D     = share_deriv_rc(theta1, theta2, dm, share_i, const);
    Delta = -ownership .* D;
    mu    = Delta \ dm.share;
end

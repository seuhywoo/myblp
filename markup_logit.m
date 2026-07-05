function mu = markup_logit(theta1, dm, ownership)
% Markup vector for the plain logit model (multi-product Bertrand-Nash).
% FOC: p = mc + mu,  where  Delta * mu = s,  Delta(j,k) = -ownership(j,k) * D(j,k)
%
% ownership(j,k) = 1 if products j and k are owned by the same firm.
% dm must contain rows for a single market only.

    share = dm.share;
    D     = share_deriv_logit(theta1(2), share);
    Delta = -ownership .* D;
    mu    = Delta \ share;
end

function D = share_deriv_logit(alpha, share)
% [J x J] matrix of d(s_j)/d(p_k) for the plain logit model.
%   D(j,j) =  alpha * s_j * (1 - s_j)
%   D(j,k) = -alpha * s_j * s_k       (j ~= k)

    D = -alpha * (share * share');
    J = length(share);
    D(1:J+1:end) = alpha * share .* (1 - share);
end

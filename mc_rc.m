function mc = mc_rc(theta1, theta2, dm, share_i, const, ownership)
% Backs out marginal costs under the RC logit model:  mc = price - markup
% dm must contain rows for a single market only.

    mu = markup_rc(theta1, theta2, dm, share_i, const, ownership);
    mc = dm.x2(:, 1) - mu;
end

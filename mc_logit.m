function mc = mc_logit(theta1, dm, ownership)
% Backs out marginal costs under the plain logit model:  mc = price - markup
% dm must contain rows for a single market only.

    mu = markup_logit(theta1, dm, ownership);
    mc = dm.x2(:, 1) - mu;
end

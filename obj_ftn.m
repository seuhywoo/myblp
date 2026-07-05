function F = obj_ftn(theta2, data, const)
% GMM objective  (1/N) * xi'*Z*W*Z'*xi
% Inner: mean_utility → delta; 2SLS → theta1; recover xi = delta - x1*theta1

    global counter

    [delta, ~] = mean_utility(theta2, data, const);

    if any(isnan(delta))
        F = 1e9;
        counter = counter + 1;
        return
    end

    res = tsls_est(delta, data.x1, data.Z);
    xi  = delta - data.x1 * res.beta;
    N   = length(data.share);
    F   = xi' * data.Z * data.invW * data.Z' * xi / N;

    counter = counter + 1;
    if mod(counter, 10) == 0
        fprintf('iter = %4d    obj = %.8f\n', counter, F);
    end
end

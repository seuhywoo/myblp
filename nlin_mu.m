function mu = nlin_mu(theta2, data, const)
% Nonlinear utility deviation, Nevo (2000) specification  [N x n_sim]
%
% X2 = [1, price, sugar, mushy] — random coefficient on all four.
% Sign convention (pyblp):  mu = +(pi·D_i + sigma·nu_i) * x_j
%
% theta2 (13 x 1), grouped by X2 characteristic:
%    1 sigma_c     2 pi_c_inc    3 pi_c_age                 (constant)
%    4 sigma_p     5 pi_p_inc    6 pi_p_incSq   7 pi_p_child (price)
%    8 sigma_s     9 pi_s_inc   10 pi_s_age                 (sugar)
%   11 sigma_m    12 pi_m_inc   13 pi_m_age                 (mushy)
%
% Nevo's interaction pattern: income^2 enters price only; age never
% interacts with price — this breaks the income/income^2 collinearity
% that plagues an all-on-price specification.

    n_sim = const.n_sim;

    sigma_c = theta2(1);  pi_c_inc = theta2(2);  pi_c_age   = theta2(3);
    sigma_p = theta2(4);  pi_p_inc = theta2(5);  pi_p_incSq = theta2(6);
    pi_p_child = theta2(7);
    sigma_s = theta2(8);  pi_s_inc = theta2(9);  pi_s_age   = theta2(10);
    sigma_m = theta2(11); pi_m_inc = theta2(12); pi_m_age   = theta2(13);

    income   = data.income(:,   1:n_sim);   % [N x n_sim]
    incomeSq = data.incomeSq(:, 1:n_sim);
    age      = data.age(:,      1:n_sim);
    child    = data.child(:,    1:n_sim);
    nu_c     = data.nu_c(:,     1:n_sim);   % nodes0 → constant RC
    nu_p     = data.nu_p(:,     1:n_sim);   % nodes1 → price RC
    nu_s     = data.nu_s(:,     1:n_sim);   % nodes2 → sugar RC
    nu_m     = data.nu_m(:,     1:n_sim);   % nodes3 → mushy RC

    price = data.x2(:, 1);   % [N x 1]
    sugar = data.x2(:, 2);
    mushy = data.x2(:, 3);

    mu = (sigma_c*nu_c + pi_c_inc*income + pi_c_age*age) ...
       + price .* (sigma_p*nu_p + pi_p_inc*income ...
                   + pi_p_incSq*incomeSq + pi_p_child*child) ...
       + sugar .* (sigma_s*nu_s + pi_s_inc*income + pi_s_age*age) ...
       + mushy .* (sigma_m*nu_m + pi_m_inc*income + pi_m_age*age);
end

function data = data_loader(seed, split, csv_path)
%DATA_LOADER Load the Concrete Compressive Strength dataset.
%   data = data_loader(seed, split, csv_path) reads the CSV, performs a
%   stratified (by target quantile) split into train/val/test, and
%   standardizes features and target using training-set statistics only.
%
%   Inputs:
%     seed      : integer, controls the split and row ordering
%     split     : 1x3 vector of ratios summing to 1, e.g. [0.70 0.15 0.15]
%     csv_path  : path to concrete_data.csv
%
%   Output struct:
%     X_train, X_val, X_test  (standardized)
%     y_train, y_val, y_test  (standardized)
%     mu_X, sigma_X, mu_y, sigma_y   (training-set stats)
%
%   The un-standardized target can be recovered as y_raw = sigma_y*y + mu_y.

    T = readmatrix(csv_path);
    X = T(:, 1:end-1);
    y = T(:, end);
    N = size(X, 1);

    rng(seed, 'twister');

    nb = 10;
    edges = quantile(y, linspace(0, 1, nb + 1));
    edges(1) = -Inf; edges(end) = Inf;
    bin = discretize(y, edges);

    idx_tr = false(N, 1);
    idx_va = false(N, 1);
    idx_te = false(N, 1);
    for b = 1:nb
        idx_b = find(bin == b);
        idx_b = idx_b(randperm(numel(idx_b)));
        n_b = numel(idx_b);
        n_tr = round(split(1) * n_b);
        n_va = round(split(2) * n_b);
        n_te = n_b - n_tr - n_va;
        idx_tr(idx_b(1:n_tr)) = true;
        idx_va(idx_b(n_tr+1:n_tr+n_va)) = true;
        idx_te(idx_b(n_tr+n_va+1:end)) = true;
        assert(n_te == numel(idx_b) - n_tr - n_va);
    end

    mu_X = mean(X(idx_tr, :), 1);
    sigma_X = std(X(idx_tr, :), 0, 1);
    sigma_X(sigma_X == 0) = 1;
    mu_y = mean(y(idx_tr));
    sigma_y = std(y(idx_tr));

    standardize = @(A, mu, sig) (A - mu) ./ sig;

    data.X_train = standardize(X(idx_tr, :), mu_X, sigma_X);
    data.X_val   = standardize(X(idx_va, :), mu_X, sigma_X);
    data.X_test  = standardize(X(idx_te, :), mu_X, sigma_X);
    data.y_train = (y(idx_tr) - mu_y) / sigma_y;
    data.y_val   = (y(idx_va) - mu_y) / sigma_y;
    data.y_test  = (y(idx_te) - mu_y) / sigma_y;

    data.mu_X = mu_X; data.sigma_X = sigma_X;
    data.mu_y = mu_y; data.sigma_y = sigma_y;
end

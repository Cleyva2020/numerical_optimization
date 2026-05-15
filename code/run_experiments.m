function cfg = run_experiments()
%RUN_EXPERIMENTS Run the complete optimizer comparison.
%
% This script:
%   1. Loads the experiment configuration.
%   2. Tunes LM, NCG, and Adam using the validation set.
%   3. Runs the final comparison across all random seeds.
%   4. Saves the trained parameters, history, configuration, and metrics.

    cfg = default_config();

    if ~exist('results', 'dir')
        mkdir('results');
    end

    % ------------------------------------------------------------
    % Hyperparameter tuning
    % ------------------------------------------------------------
    % Each optimizer is tuned using the first seed and validation RMSE.
    fprintf('=== Hyperparameter tuning on seed %d ===\n', cfg.seeds(1));

    cfg.lm   = tune_lm_grid(cfg);
    cfg.ncg  = tune_ncg_grid(cfg);
    cfg.adam = tune_adam_grid(cfg);

    % ------------------------------------------------------------
    % Final multi-seed experiment
    % ------------------------------------------------------------
    % After tuning, each optimizer is evaluated using the selected
    % hyperparameters across all seeds.
    for seed = cfg.seeds
        fprintf('\n=== Seed %d ===\n', seed);

        problem = build_problem(seed, cfg);
        theta0  = nn('init', cfg.arch, seed);

        fprintf('-- LM --\n');
        [theta, hist] = opt_lm(theta0, problem, fill_common(cfg.lm, cfg));
        save_result(seed, 'lm', theta, hist, cfg, problem);

        fprintf('-- NCG --\n');
        [theta, hist] = opt_ncg(theta0, problem, fill_common(cfg.ncg, cfg));
        save_result(seed, 'ncg', theta, hist, cfg, problem);

        fprintf('-- Adam --\n');
        [theta, hist] = opt_adam(theta0, problem, fill_common(cfg.adam, cfg));
        save_result(seed, 'adam', theta, hist, cfg, problem);
    end

    fprintf('\nAll experiments complete. See results/.\n');
end


function opts = fill_common(opts, cfg)
%FILL_COMMON Add shared stopping parameters to an optimizer option structure.

    opts.grad_tol = cfg.grad_tol;
end


function save_result(seed, method, theta, hist, cfg, problem)
%SAVE_RESULT Compute final metrics and save one experiment result.

    tr_rmse = rmse_mpa(theta, problem.X_train, problem.y_train, problem);
    va_rmse = rmse_mpa(theta, problem.X_val,   problem.y_val,   problem);
    te_rmse = rmse_mpa(theta, problem.X_test,  problem.y_test,  problem);

    meta.train_rmse = tr_rmse;
    meta.val_rmse   = va_rmse;
    meta.test_rmse  = te_rmse;
    meta.seed       = seed;
    meta.method     = method;

    fprintf('  train/val/test RMSE (MPa): %.3f / %.3f / %.3f\n', ...
            tr_rmse, va_rmse, te_rmse);

    fname = fullfile('results', sprintf('seed%d_%s.mat', seed, method));
    save(fname, 'theta', 'hist', 'cfg', 'meta');
end


function rmse = rmse_mpa(theta, X, y_std, problem)
%RMSE_MPA Compute RMSE in the original MPa scale.
%
% The network is trained on standardized targets. This function converts
% predictions and targets back to MPa before computing RMSE.

    yhat_std = nn('forward', theta, X, problem.arch);

    yhat_mpa = yhat_std * problem.sigma_y + problem.mu_y;
    y_mpa    = y_std   * problem.sigma_y + problem.mu_y;

    rmse = sqrt(mean((y_mpa - yhat_mpa).^2));
end


function problem = build_problem(seed, cfg)
%BUILD_PROBLEM Load data and create the problem structure for one seed.

    data = data_loader(seed, cfg.split, cfg.csv_path);

    problem.arch = cfg.arch;

    problem.X_train = data.X_train;
    problem.y_train = data.y_train;

    problem.X_val = data.X_val;
    problem.y_val = data.y_val;

    problem.X_test = data.X_test;
    problem.y_test = data.y_test;

    problem.mu_y = data.mu_y;
    problem.sigma_y = data.sigma_y;
end


function best_lm = tune_lm_grid(cfg)
%TUNE_LM_GRID Select LM hyperparameters using validation RMSE.

    seed = cfg.seeds(1);

    problem = build_problem(seed, cfg);
    theta0 = nn('init', cfg.arch, seed);

    best_val = inf;
    best_lm = cfg.lm;

    grid_out = struct( ...
        'lambda0', {}, ...
        'nu', {}, ...
        'rho_acc', {}, ...
        'val_rmse', {}, ...
        'best_val_loss', {} ...
    );

    fprintf('\n--- LM grid search ---\n');

    for a = 1:numel(cfg.lm.lambda0_grid)
        for b = 1:numel(cfg.lm.nu_grid)
            for c = 1:numel(cfg.lm.rho_acc_grid)

                opts = cfg.lm;
                opts.lambda0 = cfg.lm.lambda0_grid(a);
                opts.nu = cfg.lm.nu_grid(b);
                opts.rho_acc = cfg.lm.rho_acc_grid(c);
                opts.grad_tol = cfg.grad_tol;

                [theta, hist] = opt_lm(theta0, problem, opts);
                va = rmse_mpa(theta, problem.X_val, problem.y_val, problem);

                fprintf('  LM lambda0=%.0e, nu=%g, rho_acc=%.0e: val RMSE = %.3f MPa\n', ...
                        opts.lambda0, opts.nu, opts.rho_acc, va);

                grid_out(end+1).lambda0 = opts.lambda0;
                grid_out(end).nu = opts.nu;
                grid_out(end).rho_acc = opts.rho_acc;
                grid_out(end).val_rmse = va;

                if isfield(hist, 'best_val_loss')
                    grid_out(end).best_val_loss = hist.best_val_loss;
                else
                    grid_out(end).best_val_loss = min(hist.val_loss);
                end

                % Keep the LM configuration with the lowest validation RMSE.
                if va < best_val
                    best_val = va;
                    best_lm = opts;
                end
            end
        end
    end

    save(fullfile('results', 'lm_grid.mat'), 'grid_out', 'best_lm');

    fprintf('  -> best LM: lambda0=%.0e, nu=%g, rho_acc=%.0e\n', ...
            best_lm.lambda0, best_lm.nu, best_lm.rho_acc);
end


function best_ncg = tune_ncg_grid(cfg)
%TUNE_NCG_GRID Select NCG hyperparameters using validation RMSE.

    seed = cfg.seeds(1);

    problem = build_problem(seed, cfg);
    theta0 = nn('init', cfg.arch, seed);

    best_val = inf;
    best_ncg = cfg.ncg;

    grid_out = struct( ...
        'c1', {}, ...
        'c2', {}, ...
        'val_rmse', {}, ...
        'best_val_loss', {} ...
    );

    fprintf('\n--- NCG grid search ---\n');

    for a = 1:numel(cfg.ncg.c1_grid)
        for b = 1:numel(cfg.ncg.c2_grid)

            opts = cfg.ncg;
            opts.c1 = cfg.ncg.c1_grid(a);
            opts.c2 = cfg.ncg.c2_grid(b);
            opts.grad_tol = cfg.grad_tol;

            [theta, hist] = opt_ncg(theta0, problem, opts);
            va = rmse_mpa(theta, problem.X_val, problem.y_val, problem);

            fprintf('  NCG c1=%.0e, c2=%.2f: val RMSE = %.3f MPa\n', ...
                    opts.c1, opts.c2, va);

            grid_out(end+1).c1 = opts.c1;
            grid_out(end).c2 = opts.c2;
            grid_out(end).val_rmse = va;

            if isfield(hist, 'best_val_loss')
                grid_out(end).best_val_loss = hist.best_val_loss;
            else
                grid_out(end).best_val_loss = min(hist.val_loss);
            end

            % Keep the NCG configuration with the lowest validation RMSE.
            if va < best_val
                best_val = va;
                best_ncg = opts;
            end
        end
    end

    save(fullfile('results', 'ncg_grid.mat'), 'grid_out', 'best_ncg');

    fprintf('  -> best NCG: c1=%.0e, c2=%.2f\n', ...
            best_ncg.c1, best_ncg.c2);
end


function best_adam = tune_adam_grid(cfg)
%TUNE_ADAM_GRID Select Adam hyperparameters using validation RMSE.

    seed = cfg.seeds(1);

    problem = build_problem(seed, cfg);
    theta0 = nn('init', cfg.arch, seed);

    best_val = inf;
    best_adam = cfg.adam;

    grid_out = struct( ...
        'lr', {}, ...
        'batch', {}, ...
        'val_rmse', {}, ...
        'best_val_loss', {} ...
    );

    opts_base = cfg.adam;
    opts_base.grad_tol = cfg.grad_tol;

    % Use a shorter training budget during tuning to reduce total runtime.
    opts_base.max_epochs = max(200, round(cfg.adam.max_epochs / 4));

    fprintf('\n--- Adam grid search ---\n');

    for a = 1:numel(cfg.adam.lr_grid)
        for b = 1:numel(cfg.adam.batch_grid)

            opts = opts_base;
            opts.lr = cfg.adam.lr_grid(a);
            opts.batch = cfg.adam.batch_grid(b);

            [theta, hist] = opt_adam(theta0, problem, opts);
            va = rmse_mpa(theta, problem.X_val, problem.y_val, problem);

            fprintf('  Adam lr=%.0e, batch=%d: val RMSE = %.3f MPa\n', ...
                    opts.lr, opts.batch, va);

            grid_out(end+1).lr = opts.lr;
            grid_out(end).batch = opts.batch;
            grid_out(end).val_rmse = va;

            if isfield(hist, 'best_val_loss')
                grid_out(end).best_val_loss = hist.best_val_loss;
            else
                grid_out(end).best_val_loss = NaN;
            end

            % Keep the Adam configuration with the lowest validation RMSE.
            if va < best_val
                best_val = va;
                best_adam = opts;
            end
        end
    end

    save(fullfile('results', 'adam_grid.mat'), 'grid_out', 'best_adam');

    % Keep compatibility with older scripts that expect adam_lr_grid.mat.
    best_lr = best_adam.lr;
    save(fullfile('results', 'adam_lr_grid.mat'), 'grid_out', 'best_lr');

    fprintf('  -> best Adam: lr=%.0e, batch=%d\n', ...
            best_adam.lr, best_adam.batch);
end
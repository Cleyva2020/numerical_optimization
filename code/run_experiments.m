function cfg = run_experiments()
%RUN_EXPERIMENTS Multi-seed driver for LM, NCG, Adam on Concrete dataset.
%   Saves one .mat file per (seed, method) under results/.
%   For Adam, first tunes the learning rate on seed 0 over a grid and
%   caches the grid outcome in results/adam_lr_grid.mat.

    cfg = default_config();
    if ~exist('results', 'dir'), mkdir('results'); end

    fprintf('=== Adam LR tuning on seed 0 ===\n');
    best_lr = tune_adam_lr(cfg);
    cfg.adam.lr = best_lr;

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
    opts.grad_tol = cfg.grad_tol;
end


function save_result(seed, method, theta, hist, cfg, problem)
    tr_rmse = rmse_mpa(theta, problem.X_train, problem.y_train, problem);
    va_rmse = rmse_mpa(theta, problem.X_val,   problem.y_val,   problem);
    te_rmse = rmse_mpa(theta, problem.X_test,  problem.y_test,  problem);
    meta.train_rmse = tr_rmse;
    meta.val_rmse   = va_rmse;
    meta.test_rmse  = te_rmse;
    meta.seed       = seed;
    meta.method     = method;
    fprintf('  train/val/test RMSE (MPa): %.3f / %.3f / %.3f\n', tr_rmse, va_rmse, te_rmse);
    fname = fullfile('results', sprintf('seed%d_%s.mat', seed, method));
    save(fname, 'theta', 'hist', 'cfg', 'meta');
end


function rmse = rmse_mpa(theta, X, y_std, problem)
    yhat_std = nn('forward', theta, X, problem.arch);
    yhat_mpa = yhat_std * problem.sigma_y + problem.mu_y;
    y_mpa    = y_std   * problem.sigma_y + problem.mu_y;
    rmse = sqrt(mean((y_mpa - yhat_mpa).^2));
end


function problem = build_problem(seed, cfg)
    data = data_loader(seed, cfg.split, cfg.csv_path);
    problem.arch = cfg.arch;
    problem.X_train = data.X_train; problem.y_train = data.y_train;
    problem.X_val   = data.X_val;   problem.y_val   = data.y_val;
    problem.X_test  = data.X_test;  problem.y_test  = data.y_test;
    problem.mu_y    = data.mu_y;    problem.sigma_y = data.sigma_y;
end


function best_lr = tune_adam_lr(cfg)
    seed = cfg.seeds(1);
    problem = build_problem(seed, cfg);
    theta0  = nn('init', cfg.arch, seed);
    grid = cfg.adam.lr_grid;
    best_val = inf; best_lr = grid(1);
    grid_out = struct('lr', {}, 'val_rmse', {}, 'best_val_loss', {});
    opts_base = cfg.adam; opts_base.grad_tol = cfg.grad_tol;
    opts_base.max_epochs = max(200, round(cfg.adam.max_epochs / 4));  % cheaper tuning
    for i = 1:numel(grid)
        opts_base.lr = grid(i);
        [theta, hist] = opt_adam(theta0, problem, opts_base);
        va = rmse_mpa(theta, problem.X_val, problem.y_val, problem);
        fprintf('  lr = %.0e: val RMSE = %.3f MPa (best val loss = %.4e)\n', grid(i), va, hist.best_val_loss);
        grid_out(end+1).lr = grid(i);
        grid_out(end).val_rmse = va;
        grid_out(end).best_val_loss = hist.best_val_loss;
        if va < best_val
            best_val = va;
            best_lr = grid(i);
        end
    end
    save(fullfile('results', 'adam_lr_grid.mat'), 'grid_out', 'best_lr');
    fprintf('  -> best LR: %.0e\n', best_lr);
end

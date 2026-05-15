function cfg = default_config()
%DEFAULT_CONFIG Hyperparameters and experiment settings.
% This file defines the shared experiment setup and optimizer grids.

    % Common experiment settings
    cfg.arch     = [8 16 16 1];
    cfg.seeds    = 0:4;
    cfg.split    = [0.70 0.15 0.15];
    cfg.grad_tol = 1e-5;
    cfg.csv_path = fullfile('..', 'data', 'concrete_data.csv');

    % Levenberg-Marquardt settings
    cfg.lm.lambda0  = 1e-3;
    cfg.lm.nu       = 10;
    cfg.lm.rho_acc  = 1e-4;
    cfg.lm.max_iter = 500;
    cfg.lm.patience = 50;

    % LM validation grid
    cfg.lm.lambda0_grid = [1e-4 1e-3 1e-2];
    cfg.lm.nu_grid      = [5 10];
    cfg.lm.rho_acc_grid = [1e-4 1e-3];

    % Nonlinear Conjugate Gradient settings
    cfg.ncg.c1       = 1e-4;
    cfg.ncg.c2       = 0.1;
    cfg.ncg.max_iter = 500;
    cfg.ncg.patience = 50;

    % NCG validation grid
    cfg.ncg.c1_grid = [1e-4];
    cfg.ncg.c2_grid = [0.1 0.3 0.5];

    % Adam settings
    cfg.adam.batch      = 32;
    cfg.adam.beta1      = 0.9;
    cfg.adam.beta2      = 0.999;
    cfg.adam.eps        = 1e-8;
    cfg.adam.max_epochs = 2000;
    cfg.adam.patience   = 50;

    % Adam validation grid
    cfg.adam.lr_grid    = [1e-4 3e-4 1e-3 3e-3 1e-2];
    cfg.adam.batch_grid = [16 32 64];
end
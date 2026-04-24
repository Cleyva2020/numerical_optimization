function cfg = default_config()
%DEFAULT_CONFIG Hyperparameters and experiment settings.
%   Single source of truth. The Methodology section of the report mirrors
%   the values set here.

    cfg.arch     = [8 16 16 1];
    cfg.seeds    = 0:4;
    cfg.split    = [0.70 0.15 0.15];
    cfg.grad_tol = 1e-5;
    cfg.csv_path = fullfile('..', 'data', 'concrete_data.csv');

    cfg.lm.lambda0  = 1e-3;
    cfg.lm.nu       = 10;
    cfg.lm.rho_acc  = 1e-4;
    cfg.lm.max_iter = 500;

    cfg.ncg.c1       = 1e-4;
    cfg.ncg.c2       = 0.1;
    cfg.ncg.max_iter = 500;

    cfg.adam.lr_grid    = [1e-4 3e-4 1e-3 3e-3 1e-2];
    cfg.adam.batch      = 32;
    cfg.adam.beta1      = 0.9;
    cfg.adam.beta2      = 0.999;
    cfg.adam.eps        = 1e-8;
    cfg.adam.max_epochs = 2000;
    cfg.adam.patience   = 50;
end

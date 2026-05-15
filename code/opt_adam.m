function [theta, history] = opt_adam(theta0, problem, opts)
%OPT_ADAM Adam optimizer with mini-batches and validation early stopping.
%
% This optimizer minimizes the neural network training loss:
%
%       L(theta) = 1/2 * ||r(theta)||^2
%
% where r(theta) is the residual vector between the true outputs and the
% network predictions. Adam is a stochastic first-order method: it updates
% the parameters using mini-batch gradients and adaptive moment estimates.
%
% The Adam update is based on:
%
%       m_t = beta1*m_{t-1} + (1 - beta1)*g_t
%       v_t = beta2*v_{t-1} + (1 - beta2)*(g_t .* g_t)
%
% followed by bias correction and an adaptive parameter update.
%
% Inputs:
%   theta0  - Initial parameter vector containing all weights and biases.
%   problem - Structure containing data partitions and network architecture.
%   opts    - Structure with Adam hyperparameters:
%             lr, batch, beta1, beta2, eps, max_epochs, patience, grad_tol.
%
% Outputs:
%   theta   - Best parameter vector selected by validation loss.
%   history - Structure with optimization history for plots and analysis.

    % ------------------------------------------------------------
    % Problem data and hyperparameters
    % ------------------------------------------------------------
    arch = problem.arch;

    Xtr = problem.X_train;
    ytr = problem.y_train;

    Xva = problem.X_val;
    yva = problem.y_val;

    Ntr = size(Xtr, 1);
    n = numel(theta0);

    lr = opts.lr;
    bsz = opts.batch;

    b1 = opts.beta1;
    b2 = opts.beta2;
    eps = opts.eps;

    max_ep = opts.max_epochs;
    patience = opts.patience;
    grad_tol = getfielddef(opts, 'grad_tol', 0);

    % ------------------------------------------------------------
    % Adam state initialization
    % ------------------------------------------------------------
    theta = theta0;

    % First moment estimate: moving average of gradients.
    m = zeros(n, 1);

    % Second moment estimate: moving average of squared gradients.
    v = zeros(n, 1);

    % Adam update counter, used for bias correction.
    t = 0;

    % ------------------------------------------------------------
    % History and validation-based model selection
    % ------------------------------------------------------------
    history = init_history(max_ep);

    best_val = inf;
    best_theta = theta;
    stale = 0;

    t_start = tic;

    % ------------------------------------------------------------
    % Main Adam training loop
    % ------------------------------------------------------------
    % One epoch means one full pass through the training set using mini-batches.
    for ep = 1:max_ep

        % Shuffle the training data at the start of each epoch.
        % The seed depends on the epoch to make the shuffle reproducible.
        rng(1000 + ep, 'twister');
        perm = randperm(Ntr);

        % --------------------------------------------------------
        % Mini-batch updates
        % --------------------------------------------------------
        % Adam updates theta several times per epoch, once per mini-batch.
        for bs = 1:bsz:Ntr
            be = min(bs + bsz - 1, Ntr);
            idx = perm(bs:be);

            % Compute mini-batch residuals, Jacobian, and gradient.
            [r, J] = residual_jacobian(theta, Xtr(idx, :), ytr(idx), arch);
            g = J' * r;

            % ----------------------------------------------------
            % Adam moment updates
            % ----------------------------------------------------
            % m stores the moving average of gradients.
            % v stores the moving average of squared gradients.
            t = t + 1;
            m = b1 * m + (1 - b1) * g;
            v = b2 * v + (1 - b2) * (g .* g);

            % Bias correction for the first and second moments.
            mhat = m / (1 - b1^t);
            vhat = v / (1 - b2^t);

            % Adaptive parameter update.
            theta = theta - lr * mhat ./ (sqrt(vhat) + eps);
        end

        % --------------------------------------------------------
        % Full-batch monitoring after each epoch
        % --------------------------------------------------------
        % Even though Adam trains with mini-batches, performance is logged
        % using the full training and validation sets for a fair comparison.
        [loss_tr, grad_inf] = full_loss_and_grad(theta, Xtr, ytr, arch);
        loss_va = 0.5 * sum(residual_jacobian(theta, Xva, yva, arch).^2);

        history = append_history(history, ep, loss_tr, loss_va, ...
                                 grad_inf, toc(t_start), lr);

        % --------------------------------------------------------
        % Validation monitoring and early stopping
        % --------------------------------------------------------
        % The final model is selected using the lowest validation loss.
        % If validation loss does not improve for 'patience' epochs,
        % training stops early.
        if loss_va < best_val - 1e-12
            best_val = loss_va;
            best_theta = theta;
            stale = 0;
        else
            stale = stale + 1;

            if stale >= patience
                fprintf('opt_adam: early stop at epoch %d (patience %d).\n', ...
                        ep, patience);
                break;
            end
        end

        % Optional gradient-based stopping condition.
        if grad_tol > 0 && grad_inf < grad_tol
            fprintf('opt_adam: grad tol reached at epoch %d.\n', ep);
            break;
        end
    end

    % Return the model with the best validation loss, not necessarily the last epoch.
    theta = best_theta;

    history = trim_history(history);
    history.best_val_loss = best_val;
end


function [L, g_inf] = full_loss_and_grad(theta, X, y, arch)
%FULL_LOSS_AND_GRAD Compute full-batch loss and gradient infinity norm.
%
% This function is used for monitoring after each epoch. It evaluates the
% loss and gradient on the full dataset, not only on a mini-batch.

    [r, J] = residual_jacobian(theta, X, y, arch);
    L = 0.5 * (r' * r);
    g_inf = max(abs(J' * r));
end


function h = init_history(K)
%INIT_HISTORY Preallocate optimization history arrays.
%
% For Adam, the history is stored once per epoch.

    h.iter       = zeros(K, 1);
    h.loss       = zeros(K, 1);
    h.val_loss   = zeros(K, 1);
    h.grad_inf   = zeros(K, 1);
    h.time       = zeros(K, 1);
    h.lr         = zeros(K, 1);
    h.k          = 0;
end


function h = append_history(h, it, L, Lv, gi, t, lr)
%APPEND_HISTORY Store one optimization record in the history structure.

    h.k = h.k + 1;

    h.iter(h.k)     = it;
    h.loss(h.k)     = L;
    h.val_loss(h.k) = Lv;
    h.grad_inf(h.k) = gi;
    h.time(h.k)     = t;
    h.lr(h.k)       = lr;
end


function h = trim_history(h)
%TRIM_HISTORY Remove unused preallocated entries after early stopping.

    k = h.k;
    f = fieldnames(h);

    for i = 1:numel(f)
        fi = f{i};

        if isnumeric(h.(fi)) && numel(h.(fi)) >= k
            h.(fi) = h.(fi)(1:k);
        end
    end
end


function v = getfielddef(s, f, d)
%GETFIELDDEF Return a structure field if it exists, otherwise return default.

    if isfield(s, f)
        v = s.(f);
    else
        v = d;
    end
end
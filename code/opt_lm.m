function [theta, history] = opt_lm(theta0, problem, opts)
%OPT_LM Levenberg-Marquardt method for nonlinear least-squares training.
%
% This optimizer minimizes the neural network training loss:
%
%       L(theta) = 1/2 * ||r(theta)||^2
%
% where r(theta) is the residual vector between the true outputs and the
% network predictions. The Levenberg-Marquardt step is computed from:
%
%       (J'J + lambda I)p = -J'r
%
% where J is the residual Jacobian and lambda is the damping parameter.
%
% Inputs:
%   theta0  - Initial parameter vector containing all weights and biases.
%   problem - Structure containing data partitions and network architecture.
%   opts    - Structure with LM hyperparameters:
%             lambda0, nu, rho_acc, max_iter, grad_tol, patience.
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

    lambda   = opts.lambda0;
    nu       = opts.nu;
    rho_acc  = opts.rho_acc;
    max_iter = opts.max_iter;
    grad_tol = opts.grad_tol;
    patience = getfielddef(opts, 'patience', inf);

    % ------------------------------------------------------------
    % Initial objective, Jacobian, and gradient
    % ------------------------------------------------------------
    theta = theta0;

    [r, J] = residual_jacobian(theta, Xtr, ytr, arch);
    L = 0.5 * (r' * r);
    g = J' * r;

    n = numel(theta);
    I = speye(n);

    % ------------------------------------------------------------
    % History and validation-based model selection
    % ------------------------------------------------------------
    history = init_history(max_iter + 1);
    t_start = tic;

    Lv = val_loss(theta, Xva, yva, arch);
    history = append_history(history, 0, L, Lv, max(abs(g)), ...
                             toc(t_start), lambda, 1, 1.0);

    best_val = Lv;
    best_theta = theta;
    stale = 0;

    rej_streak = 0;

    % ------------------------------------------------------------
    % Main Levenberg-Marquardt loop
    % ------------------------------------------------------------
    for k = 1:max_iter

        % Stop if the first-order optimality condition is approximately met.
        if max(abs(g)) < grad_tol
            fprintf('opt_lm: grad tol at iter %d.\n', k - 1);
            break;
        end

        % Build the damped Gauss-Newton system:
        %       (J'J + lambda I)p = -g
        A = J' * J + lambda * I;

        % If the system is ill-conditioned, increase lambda to stabilize it.
        bump = 0;
        while rcond(full(A)) < 1e-14 && bump < 30
            lambda = lambda * nu;
            A = J' * J + lambda * I;
            bump = bump + 1;
        end

        % Compute the proposed LM step.
        p = A \ (-g);

        % Evaluate the candidate parameters before accepting the step.
        theta_trial = theta + p;
        r_trial = residual_jacobian(theta_trial, Xtr, ytr, arch);
        L_trial = 0.5 * (r_trial' * r_trial);

        % Predicted reduction from the quadratic Gauss-Newton model.
        pred_red = -(g' * p + 0.5 * (p' * (J' * J) * p));

        % Gain ratio: actual reduction divided by predicted reduction.
        if pred_red <= 0
            rho = -1;
        else
            rho = (L - L_trial) / pred_red;
        end

        % Accept or reject the step based on the gain ratio.
        accepted = rho > rho_acc;

        if accepted
            theta = theta_trial;

            [r, J] = residual_jacobian(theta, Xtr, ytr, arch);
            L = 0.5 * (r' * r);
            g = J' * r;

            % Successful step: trust the Gauss-Newton model more.
            lambda = max(lambda / nu, 1e-12);
            rej_streak = 0;
        else
            % Failed step: increase damping and try a more conservative step.
            lambda = lambda * nu;
            rej_streak = rej_streak + 1;
        end

        % --------------------------------------------------------
        % Validation monitoring and early stopping
        % --------------------------------------------------------
        Lv = val_loss(theta, Xva, yva, arch);
        history = append_history(history, k, L, Lv, max(abs(g)), ...
                                 toc(t_start), lambda, accepted, rho);

        if Lv < best_val - 1e-12
            best_val = Lv;
            best_theta = theta;
            stale = 0;
        else
            stale = stale + 1;

            if stale >= patience
                fprintf('opt_lm: early stop at iter %d (patience %d).\n', ...
                        k, patience);
                break;
            end
        end

        % Safety stop if the method repeatedly rejects steps and lambda explodes.
        if rej_streak >= 15 && lambda > 1e12
            fprintf('opt_lm: lambda blew up at iter %d, stopping.\n', k);
            break;
        end
    end

    % Return the model with the best validation loss, not necessarily the last iterate.
    theta = best_theta;

    history.best_val_loss = best_val;
    history = trim_history(history);
end


function v = val_loss(theta, X, y, arch)
%VAL_LOSS Compute least-squares loss on a validation set.

    r = residual_jacobian(theta, X, y, arch);
    v = 0.5 * sum(r.^2);
end


function h = init_history(K)
%INIT_HISTORY Preallocate optimization history arrays.

    h.iter     = zeros(K, 1);
    h.loss     = zeros(K, 1);
    h.val_loss = zeros(K, 1);
    h.grad_inf = zeros(K, 1);
    h.time     = zeros(K, 1);
    h.lambda   = zeros(K, 1);
    h.accepted = zeros(K, 1);
    h.rho      = zeros(K, 1);
    h.k        = 0;
end


function h = append_history(h, it, L, Lv, gi, t, lam, acc, rho)
%APPEND_HISTORY Store one optimization record in the history structure.

    h.k = h.k + 1;

    h.iter(h.k)     = it;
    h.loss(h.k)     = L;
    h.val_loss(h.k) = Lv;
    h.grad_inf(h.k) = gi;
    h.time(h.k)     = t;
    h.lambda(h.k)   = lam;
    h.accepted(h.k) = acc;
    h.rho(h.k)      = rho;
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
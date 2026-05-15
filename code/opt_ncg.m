function [theta, history] = opt_ncg(theta0, problem, opts)
%OPT_NCG Nonlinear Conjugate Gradient with Polak-Ribiere+ and strong Wolfe.
%
% This optimizer minimizes the neural network training loss:
%
%       L(theta) = 1/2 * ||r(theta)||^2
%
% where r(theta) is the residual vector between the true outputs and the
% network predictions. NCG is a first-order method: it uses gradients and
% a conjugate search direction instead of forming or solving a Hessian system.
%
% The method updates the parameters using:
%
%       theta_{k+1} = theta_k + alpha_k * p_k
%
% where p_k is the nonlinear conjugate gradient direction and alpha_k is
% selected using a strong Wolfe line search.
%
% Inputs:
%   theta0  - Initial parameter vector containing all weights and biases.
%   problem - Structure containing data partitions and network architecture.
%   opts    - Structure with NCG hyperparameters:
%             c1, c2, max_iter, grad_tol, patience.
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

    n = numel(theta0);

    grad_tol = opts.grad_tol;
    max_iter = opts.max_iter;
    patience = getfielddef(opts, 'patience', inf);

    % ------------------------------------------------------------
    % Initial objective, gradient, and search direction
    % ------------------------------------------------------------
    theta = theta0;

    [r, J] = residual_jacobian(theta, Xtr, ytr, arch);
    L = 0.5 * (r' * r);
    g = J' * r;

    % The first NCG direction is the steepest descent direction.
    p = -g;

    % ------------------------------------------------------------
    % History and validation-based model selection
    % ------------------------------------------------------------
    history = init_history(max_iter + 1);
    ls_failures = 0;
    t_start = tic;

    Lv = val_loss(theta, Xva, yva, arch);
    history = append_history(history, 0, L, Lv, max(abs(g)), ...
                             toc(t_start), 1.0, 0.0);

    best_val = Lv;
    best_theta = theta;
    stale = 0;

    % ------------------------------------------------------------
    % Main Nonlinear Conjugate Gradient loop
    % ------------------------------------------------------------
    for k = 1:max_iter

        % Ensure that the current direction is a descent direction.
        % If not, restart with the negative gradient.
        if g' * p >= 0
            p = -g;
        end

        % Define the one-dimensional line-search function:
        %
        %       phi(alpha) = L(theta + alpha*p)
        %
        % and its directional derivative:
        %
        %       phi'(alpha) = grad L(theta + alpha*p)' * p
        phi  = @(a) ls_loss(theta + a * p, Xtr, ytr, arch);
        dphi = @(a) ls_gradp(theta + a * p, Xtr, ytr, arch, p);

        phi0 = L;
        dphi0 = g' * p;

        % Strong Wolfe parameters control the accepted step length alpha.
        ls_opts.c1 = opts.c1;
        ls_opts.c2 = opts.c2;
        ls_opts.alpha0 = 1.0;
        ls_opts.max_iter = 25;

        % Compute a step length alpha satisfying the strong Wolfe conditions.
        [alpha, info] = line_search_wolfe(phi, dphi, phi0, dphi0, ls_opts);

        if ~info.success
            ls_failures = ls_failures + 1;
        end

        % --------------------------------------------------------
        % Parameter update
        % --------------------------------------------------------
        theta_new = theta + alpha * p;

        [r_new, J_new] = residual_jacobian(theta_new, Xtr, ytr, arch);
        g_new = J_new' * r_new;
        L_new = 0.5 * (r_new' * r_new);

        % --------------------------------------------------------
        % Polak-Ribiere+ update
        % --------------------------------------------------------
        % The beta coefficient determines how much of the previous direction
        % is reused in the next direction:
        %
        %       p_{k+1} = -g_{k+1} + beta * p_k
        %
        % The "+" version forces beta to be nonnegative, which helps restart
        % the method when the previous direction is no longer useful.
        num = g_new' * (g_new - g);
        den = g' * g;

        if den == 0
            beta = 0;
        else
            beta = max(num / den, 0);
        end

        % Periodic restart after n iterations, where n is the number of
        % parameters. This helps prevent loss of conjugacy in nonlinear problems.
        if mod(k, n) == 0
            beta = 0;
        end

        % Build the next NCG search direction.
        p = -g_new + beta * p;

        % Accept the new iterate.
        theta = theta_new;
        g = g_new;
        L = L_new;

        % --------------------------------------------------------
        % Validation monitoring and early stopping
        % --------------------------------------------------------
        Lv = val_loss(theta, Xva, yva, arch);
        history = append_history(history, k, L, Lv, max(abs(g)), ...
                                 toc(t_start), alpha, beta);

        if Lv < best_val - 1e-12
            best_val = Lv;
            best_theta = theta;
            stale = 0;
        else
            stale = stale + 1;

            if stale >= patience
                fprintf('opt_ncg: early stop at iter %d (patience %d).\n', ...
                        k, patience);
                break;
            end
        end

        % Stop if the first-order optimality condition is approximately met.
        if max(abs(g)) < grad_tol
            fprintf('opt_ncg: grad tol at iter %d.\n', k);
            break;
        end
    end

    % Return the model with the best validation loss, not necessarily the last iterate.
    theta = best_theta;

    history = trim_history(history);
    history.ls_failures = ls_failures;
    history.best_val_loss = best_val;
end


function L = ls_loss(theta, X, y, arch)
%LS_LOSS Compute least-squares loss for line search.

    r = residual_jacobian(theta, X, y, arch);
    L = 0.5 * sum(r.^2);
end


function d = ls_gradp(theta, X, y, arch, p)
%LS_GRADP Compute directional derivative for line search.
%
% This returns:
%
%       grad L(theta)' * p
%
% which is the derivative of phi(alpha) along the search direction p.

    [r, J] = residual_jacobian(theta, X, y, arch);
    d = (J' * r)' * p;
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
    h.alpha    = zeros(K, 1);
    h.beta     = zeros(K, 1);
    h.k        = 0;
end


function h = append_history(h, it, L, Lv, gi, t, a, b)
%APPEND_HISTORY Store one optimization record in the history structure.

    h.k = h.k + 1;

    h.iter(h.k)     = it;
    h.loss(h.k)     = L;
    h.val_loss(h.k) = Lv;
    h.grad_inf(h.k) = gi;
    h.time(h.k)     = t;
    h.alpha(h.k)    = a;
    h.beta(h.k)     = b;
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
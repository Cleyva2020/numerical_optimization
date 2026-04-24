function [theta, history] = opt_lm(theta0, problem, opts)
%OPT_LM Levenberg-Marquardt for nonlinear least squares.
%
%   opts: lambda0, nu, rho_acc, max_iter, grad_tol

    arch = problem.arch;
    Xtr = problem.X_train; ytr = problem.y_train;
    Xva = problem.X_val;   yva = problem.y_val;

    lambda  = opts.lambda0;
    nu      = opts.nu;
    rho_acc = opts.rho_acc;
    max_iter = opts.max_iter;
    grad_tol = opts.grad_tol;

    theta = theta0;
    [r, J] = residual_jacobian(theta, Xtr, ytr, arch);
    L = 0.5 * (r' * r);
    g = J' * r;
    n = numel(theta);
    I = speye(n);

    history = init_history(max_iter + 1);
    t_start = tic;
    history = append_history(history, 0, L, val_loss(theta, Xva, yva, arch), max(abs(g)), toc(t_start), lambda, 1, 1.0);

    rej_streak = 0;
    for k = 1:max_iter
        if max(abs(g)) < grad_tol
            fprintf('opt_lm: grad tol at iter %d.\n', k-1);
            break;
        end

        A = J' * J + lambda * I;
        bump = 0;
        while rcond(full(A)) < 1e-14 && bump < 30
            lambda = lambda * nu;
            A = J' * J + lambda * I;
            bump = bump + 1;
        end
        p = A \ (-g);

        theta_trial = theta + p;
        r_trial = residual_jacobian(theta_trial, Xtr, ytr, arch);
        L_trial = 0.5 * (r_trial' * r_trial);

        pred_red = -(g' * p + 0.5 * (p' * (J'*J) * p));
        if pred_red <= 0
            rho = -1;
        else
            rho = (L - L_trial) / pred_red;
        end

        accepted = rho > rho_acc;
        if accepted
            theta = theta_trial;
            [r, J] = residual_jacobian(theta, Xtr, ytr, arch);
            L = 0.5 * (r' * r);
            g = J' * r;
            lambda = max(lambda / nu, 1e-12);
            rej_streak = 0;
        else
            lambda = lambda * nu;
            rej_streak = rej_streak + 1;
        end

        history = append_history(history, k, L, val_loss(theta, Xva, yva, arch), max(abs(g)), toc(t_start), lambda, accepted, rho);

        if rej_streak >= 15 && lambda > 1e12
            fprintf('opt_lm: lambda blew up at iter %d, stopping.\n', k);
            break;
        end
    end

    history = trim_history(history);
end


function v = val_loss(theta, X, y, arch)
    r = residual_jacobian(theta, X, y, arch);
    v = 0.5 * sum(r.^2);
end


function h = init_history(K)
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
    k = h.k;
    f = fieldnames(h);
    for i = 1:numel(f)
        fi = f{i};
        if isnumeric(h.(fi)) && numel(h.(fi)) >= k
            h.(fi) = h.(fi)(1:k);
        end
    end
end

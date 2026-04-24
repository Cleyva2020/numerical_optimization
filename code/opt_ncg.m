function [theta, history] = opt_ncg(theta0, problem, opts)
%OPT_NCG Nonlinear conjugate gradient with Polak-Ribiere+ and strong Wolfe.
%
%   opts: c1, c2, max_iter, grad_tol

    arch = problem.arch;
    Xtr = problem.X_train; ytr = problem.y_train;
    Xva = problem.X_val;   yva = problem.y_val;
    n = numel(theta0);

    grad_tol = opts.grad_tol;
    max_iter = opts.max_iter;

    theta = theta0;
    [r, J] = residual_jacobian(theta, Xtr, ytr, arch);
    L = 0.5 * (r' * r);
    g = J' * r;
    p = -g;

    history = init_history(max_iter + 1);
    ls_failures = 0;
    t_start = tic;
    history = append_history(history, 0, L, val_loss(theta, Xva, yva, arch), max(abs(g)), toc(t_start), 1.0, 0.0);

    for k = 1:max_iter
        if g' * p >= 0
            p = -g;
        end

        phi  = @(a) ls_loss(theta + a*p, Xtr, ytr, arch);
        dphi = @(a) ls_gradp(theta + a*p, Xtr, ytr, arch, p);
        phi0 = L;
        dphi0 = g' * p;

        ls_opts.c1 = opts.c1; ls_opts.c2 = opts.c2;
        ls_opts.alpha0 = 1.0; ls_opts.max_iter = 25;
        [alpha, info] = line_search_wolfe(phi, dphi, phi0, dphi0, ls_opts);
        if ~info.success, ls_failures = ls_failures + 1; end

        theta_new = theta + alpha * p;
        [r, J] = residual_jacobian(theta_new, Xtr, ytr, arch);
        g_new = J' * r;
        L_new = 0.5 * (r' * r);

        num = g_new' * (g_new - g);
        den = g' * g;
        if den == 0
            beta = 0;
        else
            beta = max(num / den, 0);
        end

        if mod(k, n) == 0
            beta = 0;
        end

        p = -g_new + beta * p;
        theta = theta_new;
        g = g_new;
        L = L_new;

        history = append_history(history, k, L, val_loss(theta, Xva, yva, arch), max(abs(g)), toc(t_start), alpha, beta);
        if max(abs(g)) < grad_tol
            fprintf('opt_ncg: grad tol at iter %d.\n', k);
            break;
        end
    end

    history = trim_history(history);
    history.ls_failures = ls_failures;
end


function L = ls_loss(theta, X, y, arch)
    r = residual_jacobian(theta, X, y, arch);
    L = 0.5 * sum(r.^2);
end

function d = ls_gradp(theta, X, y, arch, p)
    [r, J] = residual_jacobian(theta, X, y, arch);
    d = (J' * r)' * p;
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
    h.alpha    = zeros(K, 1);
    h.beta     = zeros(K, 1);
    h.k        = 0;
end

function h = append_history(h, it, L, Lv, gi, t, a, b)
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
    k = h.k;
    f = fieldnames(h);
    for i = 1:numel(f)
        fi = f{i};
        if isnumeric(h.(fi)) && numel(h.(fi)) >= k
            h.(fi) = h.(fi)(1:k);
        end
    end
end

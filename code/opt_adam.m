function [theta, history] = opt_adam(theta0, problem, opts)
%OPT_ADAM Adam (Kingma & Ba, 2015) with mini-batches and val-loss early stopping.
%
%   opts fields: lr, batch, beta1, beta2, eps, max_epochs, patience,
%                grad_tol (optional, stops when full-train grad inf-norm < tol)

    arch = problem.arch;
    Xtr = problem.X_train; ytr = problem.y_train;
    Xva = problem.X_val;   yva = problem.y_val;
    Ntr = size(Xtr, 1);
    n = numel(theta0);

    lr = opts.lr;
    bsz = opts.batch;
    b1 = opts.beta1; b2 = opts.beta2; eps = opts.eps;
    max_ep = opts.max_epochs;
    patience = opts.patience;
    grad_tol = getfielddef(opts, 'grad_tol', 0);

    theta = theta0;
    m = zeros(n, 1);
    v = zeros(n, 1);
    t = 0;

    history = init_history(max_ep);
    best_val = inf; best_theta = theta; stale = 0;

    t_start = tic;
    for ep = 1:max_ep
        rng(1000 + ep, 'twister');
        perm = randperm(Ntr);
        for bs = 1:bsz:Ntr
            be = min(bs + bsz - 1, Ntr);
            idx = perm(bs:be);
            [r, J] = residual_jacobian(theta, Xtr(idx, :), ytr(idx), arch);
            g = J' * r;

            t = t + 1;
            m = b1 * m + (1 - b1) * g;
            v = b2 * v + (1 - b2) * (g .* g);
            mhat = m / (1 - b1^t);
            vhat = v / (1 - b2^t);
            theta = theta - lr * mhat ./ (sqrt(vhat) + eps);
        end

        [loss_tr, grad_inf] = full_loss_and_grad(theta, Xtr, ytr, arch);
        loss_va = 0.5 * sum(residual_jacobian(theta, Xva, yva, arch).^2);
        history = append_history(history, ep, loss_tr, loss_va, grad_inf, toc(t_start), lr);

        if loss_va < best_val - 1e-12
            best_val = loss_va;
            best_theta = theta;
            stale = 0;
        else
            stale = stale + 1;
            if stale >= patience
                fprintf('opt_adam: early stop at epoch %d (patience %d).\n', ep, patience);
                break;
            end
        end
        if grad_tol > 0 && grad_inf < grad_tol
            fprintf('opt_adam: grad tol reached at epoch %d.\n', ep);
            break;
        end
    end

    history = trim_history(history);
    theta = best_theta;
    history.best_val_loss = best_val;
end


function [L, g_inf] = full_loss_and_grad(theta, X, y, arch)
    [r, J] = residual_jacobian(theta, X, y, arch);
    L = 0.5 * (r' * r);
    g_inf = max(abs(J' * r));
end


function h = init_history(K)
    h.iter       = zeros(K, 1);
    h.loss       = zeros(K, 1);
    h.val_loss   = zeros(K, 1);
    h.grad_inf   = zeros(K, 1);
    h.time       = zeros(K, 1);
    h.lr         = zeros(K, 1);
    h.k          = 0;
end

function h = append_history(h, it, L, Lv, gi, t, lr)
    h.k = h.k + 1;
    h.iter(h.k)     = it;
    h.loss(h.k)     = L;
    h.val_loss(h.k) = Lv;
    h.grad_inf(h.k) = gi;
    h.time(h.k)     = t;
    h.lr(h.k)       = lr;
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

function v = getfielddef(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end

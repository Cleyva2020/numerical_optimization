function test_optimizers()
%TEST_OPTIMIZERS Sanity-check NCG and LM on a small NLS problem.
%   A 3-D quadratic bowl has r(x) = A*x - b with A SPD, so LM should
%   converge in ~1 iteration and NCG in <= dim(x) iterations.

    A = chol([5 2 1; 2 7 3; 1 3 9], 'lower');   % so A*A' is SPD = the given matrix
    b = [1; -2; 3];

    % synthetic residual problem: r_i = (A*x - b)_i, so 0.5||r||^2 is convex quadratic
    res_fun = @(x) A * x - b;
    jac_fun = @(x) A;
    grad_fun = @(x) A' * res_fun(x);
    loss_fun = @(x) 0.5 * sum(res_fun(x).^2);

    x0 = [0; 0; 0];
    x_star = A \ b;

    fprintf('Exact solution: '); disp(x_star');

    % NCG manually (using line_search_wolfe)
    x = x0; g = grad_fun(x); p = -g;
    for k = 1:20
        phi  = @(a) loss_fun(x + a*p);
        dphi = @(a) grad_fun(x + a*p)' * p;
        ls.c1 = 1e-4; ls.c2 = 0.1; ls.alpha0 = 1; ls.max_iter = 25;
        [alpha, ~] = line_search_wolfe(phi, dphi, loss_fun(x), g'*p, ls);
        x = x + alpha * p;
        g_new = grad_fun(x);
        beta = max((g_new' * (g_new - g)) / (g' * g), 0);
        p = -g_new + beta * p;
        g = g_new;
        if max(abs(g)) < 1e-8, break; end
    end
    fprintf('NCG: converged in %d iters, |x - x*|_inf = %.3e\n', k, max(abs(x - x_star)));
    assert(max(abs(x - x_star)) < 1e-6);

    % LM manually
    x = x0;
    lambda = 1e-3;
    for k = 1:20
        r = res_fun(x); J = jac_fun(x);
        g = J' * r;
        p = (J'*J + lambda*eye(3)) \ (-g);
        L0 = 0.5*(r'*r);
        r_tr = res_fun(x + p);
        L1 = 0.5*(r_tr'*r_tr);
        pred = -(g'*p + 0.5*p'*(J'*J)*p);
        rho = (L0 - L1) / max(pred, eps);
        if rho > 1e-4
            x = x + p;
            lambda = lambda / 10;
        else
            lambda = lambda * 10;
        end
        if max(abs(g)) < 1e-8, break; end
    end
    fprintf('LM : converged in %d iters, |x - x*|_inf = %.3e\n', k, max(abs(x - x_star)));
    assert(max(abs(x - x_star)) < 1e-6);

    fprintf('test_optimizers PASSED.\n');
end

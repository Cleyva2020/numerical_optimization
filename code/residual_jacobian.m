function [r, J] = residual_jacobian(theta, X, y, arch)
%RESIDUAL_JACOBIAN Residual and analytical Jacobian for NN regression.
%   r = y - f(X; theta),                 size N x 1
%   J(i, :) = -d f(x_i; theta) / d theta, size N x n
%
%   Computed by one forward pass plus one per-sample backward pass. Tanh
%   hidden activations, linear output. Returns r only if nargout < 2.

    [yhat, cache] = nn('forward', theta, X, arch);
    r = y - yhat;

    if nargout < 2
        return;
    end

    W = cache.W;
    a = cache.a;
    N = cache.N;
    L = numel(W);
    n = nn('numel', arch);
    J = zeros(N, n);

    delta_z = ones(N, 1);

    off_end = n;
    for l = L:-1:1
        n_l = arch(l+1);
        n_lm1 = arch(l);
        a_prev = a{l};

        off_start_b = off_end - n_l + 1;
        J(:, off_start_b:off_end) = -delta_z;

        off_end_W = off_start_b - 1;
        off_start_W = off_end_W - n_l * n_lm1 + 1;
        blk_W = zeros(N, n_l * n_lm1);
        for k = 1:n_lm1
            blk_W(:, (k-1)*n_l+1 : k*n_l) = a_prev(:, k) .* delta_z;
        end
        J(:, off_start_W:off_end_W) = -blk_W;

        off_end = off_start_W - 1;

        if l > 1
            delta_a = delta_z * W{l};
            delta_z = delta_a .* (1 - a_prev.^2);
        end
    end
end

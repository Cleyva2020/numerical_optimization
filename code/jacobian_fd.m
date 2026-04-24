function J = jacobian_fd(theta, X, y, arch, h)
%JACOBIAN_FD Centered finite-difference Jacobian of the residual.
%   Used only for correctness checks; O(n) forward passes, slow.

    if nargin < 5
        h = 1e-6;
    end
    n = numel(theta);
    N = numel(y);
    J = zeros(N, n);
    for j = 1:n
        tp = theta; tp(j) = tp(j) + h;
        tm = theta; tm(j) = tm(j) - h;
        rp = residual_jacobian(tp, X, y, arch);
        rm = residual_jacobian(tm, X, y, arch);
        J(:, j) = (rp - rm) / (2 * h);
    end
end

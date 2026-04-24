function test_jacobian()
%TEST_JACOBIAN Cross-check analytical Jacobian against centered FD.
%   Tests at three random theta values on a random small batch. Fails
%   (raises an error) if relative max error exceeds 1e-6.

    archs = {[8 16 16 1], [4 5 3 1], [3 2 1]};
    tol = 1e-6;

    for i = 1:numel(archs)
        arch = archs{i};
        N = 7;
        rng(100 + i, 'twister');
        X = randn(N, arch(1));
        y = randn(N, 1);
        for trial = 1:3
            theta = 0.3 * randn(nn('numel', arch), 1);
            [~, J_ana] = residual_jacobian(theta, X, y, arch);
            J_fd = jacobian_fd(theta, X, y, arch);
            err = max(abs(J_ana(:) - J_fd(:)));
            denom = max(1, max(abs(J_fd(:))));
            rel = err / denom;
            fprintf('  arch [%s] trial %d: max|J_ana - J_fd| = %.3e (rel %.3e)\n', ...
                num2str(arch), trial, err, rel);
            if rel > tol
                error('test_jacobian: relative error %.3e exceeds tol %.3e', rel, tol);
            end
        end
    end
    fprintf('test_jacobian PASSED.\n');
end

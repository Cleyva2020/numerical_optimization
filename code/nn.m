function varargout = nn(mode, varargin)
%NN Neural network utilities: forward pass, Glorot init, pack/unpack.
%   theta0 = nn('init', arch, seed)
%     Returns a flat Glorot-initialized parameter vector. Biases are zero.
%
%   [W, b] = nn('unpack', theta, arch)
%     Converts a flat theta to cell arrays of weight matrices and bias
%     vectors. W{l} is n_l x n_{l-1}, b{l} is n_l x 1.
%
%   [yhat, cache] = nn('forward', theta, X, arch)
%     Forward pass. X is N x d. Returns yhat (N x 1) and a cache struct
%     with the pre-activations z{l} and activations a{l} used by
%     residual_jacobian for backprop. Hidden activations are tanh, output
%     layer is linear.
%
%   n = nn('numel', arch)
%     Number of parameters implied by the architecture.

    switch mode
        case 'init'
            [varargout{1:nargout}] = nn_init(varargin{:});
        case 'unpack'
            [varargout{1:nargout}] = nn_unpack(varargin{:});
        case 'forward'
            [varargout{1:nargout}] = nn_forward(varargin{:});
        case 'numel'
            varargout{1} = nn_numel(varargin{:});
        otherwise
            error('nn: unknown mode "%s"', mode);
    end
end


function n = nn_numel(arch)
    L = numel(arch) - 1;
    n = 0;
    for l = 1:L
        n = n + arch(l+1) * arch(l) + arch(l+1);
    end
end


function theta = nn_init(arch, seed)
    rng(seed, 'twister');
    L = numel(arch) - 1;
    theta = zeros(nn_numel(arch), 1);
    off = 0;
    for l = 1:L
        fan_in = arch(l);
        fan_out = arch(l+1);
        lim = sqrt(6 / (fan_in + fan_out));
        nW = fan_out * fan_in;
        theta(off+1:off+nW) = (2*rand(nW, 1) - 1) * lim;
        off = off + nW + fan_out;
    end
end


function [W, b] = nn_unpack(theta, arch)
    L = numel(arch) - 1;
    W = cell(L, 1);
    b = cell(L, 1);
    off = 0;
    for l = 1:L
        nW = arch(l+1) * arch(l);
        W{l} = reshape(theta(off+1:off+nW), arch(l+1), arch(l));
        off = off + nW;
        b{l} = theta(off+1:off+arch(l+1));
        off = off + arch(l+1);
    end
end


function [yhat, cache] = nn_forward(theta, X, arch)
    [W, b] = nn_unpack(theta, arch);
    L = numel(W);
    N = size(X, 1);

    a = cell(L+1, 1);
    z = cell(L, 1);
    a{1} = X;
    for l = 1:L-1
        z{l} = a{l} * W{l}' + b{l}';
        a{l+1} = tanh(z{l});
    end
    z{L} = a{L} * W{L}' + b{L}';
    a{L+1} = z{L};

    yhat = a{L+1};
    assert(size(yhat, 2) == 1, 'nn_forward expects scalar output');
    yhat = yhat(:);

    cache.a = a;
    cache.z = z;
    cache.W = W;
    cache.b = b;
    cache.N = N;
end

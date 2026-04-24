function [alpha, info] = line_search_wolfe(phi, dphi, phi0, dphi0, opts)
%LINE_SEARCH_WOLFE Strong Wolfe line search (Nocedal & Wright Alg. 3.5/3.6).
%
%   phi(alpha), dphi(alpha)     function handles, scalar in, scalar out
%   phi0 = phi(0), dphi0 = dphi(0)
%
%   opts fields (all optional):
%     c1        (default 1e-4)   Armijo constant
%     c2        (default 0.1)    curvature constant
%     alpha0    (default 1.0)    initial trial step
%     alpha_max (default 100)
%     max_iter  (default 25)
%
%   info.success  true if a valid alpha was found
%   info.nfev     number of phi/dphi evaluations

    if nargin < 5, opts = struct(); end
    c1 = getfielddef(opts, 'c1', 1e-4);
    c2 = getfielddef(opts, 'c2', 0.1);
    a0 = 0;
    a1 = getfielddef(opts, 'alpha0', 1.0);
    a_max = getfielddef(opts, 'alpha_max', 100);
    max_iter = getfielddef(opts, 'max_iter', 25);

    assert(dphi0 < 0, 'line_search_wolfe: p is not a descent direction');

    nfev = 0;
    phi_a0 = phi0;
    for i = 1:max_iter
        phi_a1 = phi(a1); nfev = nfev + 1;
        if phi_a1 > phi0 + c1*a1*dphi0 || (i > 1 && phi_a1 >= phi_a0)
            [alpha, nz] = zoom_(phi, dphi, a0, a1, phi_a0, phi_a1, phi0, dphi0, c1, c2, max_iter);
            nfev = nfev + nz;
            info.success = ~isnan(alpha);
            info.nfev = nfev;
            if ~info.success, alpha = backtrack(phi, phi0, dphi0, a1, c1); info.success = true; end
            return;
        end
        dphi_a1 = dphi(a1); nfev = nfev + 1;
        if abs(dphi_a1) <= -c2*dphi0
            alpha = a1;
            info.success = true;
            info.nfev = nfev;
            return;
        end
        if dphi_a1 >= 0
            [alpha, nz] = zoom_(phi, dphi, a1, a0, phi_a1, phi_a0, phi0, dphi0, c1, c2, max_iter);
            nfev = nfev + nz;
            info.success = ~isnan(alpha);
            info.nfev = nfev;
            if ~info.success, alpha = backtrack(phi, phi0, dphi0, a1, c1); info.success = true; end
            return;
        end
        a0 = a1;
        phi_a0 = phi_a1;
        a1 = min(2 * a1, a_max);
        if a1 == a_max
            alpha = backtrack(phi, phi0, dphi0, a_max, c1);
            info.success = true;
            info.nfev = nfev + 10;
            return;
        end
    end

    alpha = backtrack(phi, phi0, dphi0, 1.0, c1);
    info.success = true;
    info.nfev = nfev + 10;
end


function [alpha, nfev] = zoom_(phi, dphi, a_lo, a_hi, phi_lo, phi_hi, phi0, dphi0, c1, c2, max_iter)
    nfev = 0;
    dphi_lo = NaN;  % evaluated lazily if cubic interp needed
    for j = 1:max_iter
        aj = interp_alpha(a_lo, a_hi, phi_lo, phi_hi, dphi_lo);
        phi_j = phi(aj); nfev = nfev + 1;
        if phi_j > phi0 + c1*aj*dphi0 || phi_j >= phi_lo
            a_hi = aj;
            phi_hi = phi_j;
        else
            dphi_j = dphi(aj); nfev = nfev + 1;
            if abs(dphi_j) <= -c2*dphi0
                alpha = aj;
                return;
            end
            if dphi_j * (a_hi - a_lo) >= 0
                a_hi = a_lo;
                phi_hi = phi_lo;
            end
            a_lo = aj;
            phi_lo = phi_j;
            dphi_lo = dphi_j;
        end
        if abs(a_hi - a_lo) < 1e-12
            alpha = a_lo;
            return;
        end
    end
    alpha = NaN;  % failed to find
end


function aj = interp_alpha(a_lo, a_hi, phi_lo, phi_hi, dphi_lo)
    if isnan(dphi_lo)
        aj = 0.5 * (a_lo + a_hi);
        return;
    end
    d = a_hi - a_lo;
    if d == 0, aj = a_lo; return; end
    denom = 2*(phi_hi - phi_lo - dphi_lo*d);
    if denom == 0
        aj = 0.5 * (a_lo + a_hi);
        return;
    end
    aj = a_lo - dphi_lo * d^2 / denom;
    lo = min(a_lo, a_hi); hi = max(a_lo, a_hi);
    margin = 0.1 * (hi - lo);
    aj = min(max(aj, lo + margin), hi - margin);
end


function alpha = backtrack(phi, phi0, dphi0, a0, c1)
    alpha = a0;
    for i = 1:30
        if phi(alpha) <= phi0 + c1*alpha*dphi0
            return;
        end
        alpha = 0.5 * alpha;
    end
end


function v = getfielddef(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end

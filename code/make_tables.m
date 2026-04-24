function make_tables(tau_L)
%MAKE_TABLES Print LaTeX rows for the four Results tables.
%   Optional argument tau_L overrides the default loss threshold used for
%   tab:iters-to-tol and tab:time-to-tol. If omitted, uses a heuristic
%   (the largest value that at least one method reaches on >= 3 seeds).

    cfg = default_config();
    methods = {'lm', 'ncg', 'adam'};
    labels  = {'LM', 'NCG', 'Adam'};
    seeds   = cfg.seeds;
    runs = load_all(seeds, methods);

    summary = compute_summary(runs, methods);

    if nargin < 1 || isempty(tau_L)
        tau_L = pick_tau(summary);
    end
    fprintf('Using tau_L = %.4e for iters/time-to-tol tables.\n\n', tau_L);

    fprintf('%% tab:iters-to-tol\n');
    for m = 1:numel(methods)
        S = summary.(methods{m});
        [its, n_conv] = iters_to(S, tau_L);
        fprintf('%s  & %s & %d/%d \\\\\n', labels{m}, fmt_pm(its), n_conv, numel(seeds));
    end

    fprintf('\n%% tab:time-to-tol\n');
    for m = 1:numel(methods)
        S = summary.(methods{m});
        ts = time_to(S, tau_L);
        fprintf('%s  & %s \\\\\n', labels{m}, fmt_pm(ts));
    end

    fprintf('\n%% tab:test-rmse\n');
    for m = 1:numel(methods)
        S = summary.(methods{m});
        fprintf('%s & %s & %s & %s & %s \\\\\n', labels{m}, ...
            fmt_pm(S.train_rmse), fmt_pm(S.val_rmse), fmt_pm(S.test_rmse), ...
            fmt_pm(S.total_time));
    end

    fprintf('\n%% tab:seed-spread (test RMSE per seed)\n');
    for m = 1:numel(methods)
        S = summary.(methods{m});
        vals = S.test_rmse;
        row = sprintf('%.3f & ', vals);
        spread = max(vals) - min(vals);
        fprintf('%s & %s%.3f \\\\\n', labels{m}, row, spread);
    end
end


function runs = load_all(seeds, methods)
    runs = struct();
    for m = 1:numel(methods)
        for s = 1:numel(seeds)
            f = fullfile('results', sprintf('seed%d_%s.mat', seeds(s), methods{m}));
            if exist(f, 'file')
                runs.(methods{m}){s} = load(f);
            else
                warning('missing %s', f);
            end
        end
    end
end


function summary = compute_summary(runs, methods)
    summary = struct();
    for m = 1:numel(methods)
        R = runs.(methods{m});
        S.train_rmse = cellfun(@(r) r.meta.train_rmse, R);
        S.val_rmse   = cellfun(@(r) r.meta.val_rmse,   R);
        S.test_rmse  = cellfun(@(r) r.meta.test_rmse,  R);
        S.total_time = cellfun(@(r) r.hist.time(end),  R);
        S.histories  = cellfun(@(r) {r.hist},           R);
        summary.(methods{m}) = S;
    end
end


function tau = pick_tau(summary)
    final_vals = [];
    fns = fieldnames(summary);
    for i = 1:numel(fns)
        h = summary.(fns{i}).histories;
        for j = 1:numel(h)
            final_vals(end+1) = h{j}.loss(end); %#ok<AGROW>
        end
    end
    tau = 10^ceil(log10(median(final_vals) * 3));
end


function [iters, n_conv] = iters_to(S, tau)
    h = S.histories;
    its = nan(numel(h), 1);
    for i = 1:numel(h)
        idx = find(h{i}.loss <= tau, 1, 'first');
        if ~isempty(idx), its(i) = h{i}.iter(idx); end
    end
    n_conv = sum(~isnan(its));
    iters = its(~isnan(its));
end


function ts = time_to(S, tau)
    h = S.histories;
    ts = nan(numel(h), 1);
    for i = 1:numel(h)
        idx = find(h{i}.loss <= tau, 1, 'first');
        if ~isempty(idx), ts(i) = h{i}.time(idx); end
    end
    ts = ts(~isnan(ts));
end


function s = fmt_pm(v)
    if isempty(v), s = '---'; return; end
    s = sprintf('%.2f $\\pm$ %.2f', mean(v), std(v));
end

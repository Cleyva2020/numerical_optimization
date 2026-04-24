function make_plots()
%MAKE_PLOTS Produce the five figures for the Results section.
%   Reads results/seed*_*.mat and writes PDFs under ../report/figures/.

    out_dir = fullfile('..', 'report', 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    cfg = default_config();
    methods = {'lm', 'ncg', 'adam'};
    labels  = {'LM', 'NCG', 'Adam'};
    colors  = {[0.85 0.3 0.2], [0.2 0.5 0.8], [0.3 0.6 0.3]};

    seeds = cfg.seeds;
    runs = load_all(seeds, methods);

    plot_curve_vs_iter(runs, methods, labels, colors, 'loss', 'Training loss', out_dir, 'loss_vs_iter');
    plot_curve_vs_iter(runs, methods, labels, colors, 'grad_inf', '$\|\nabla L\|_\infty$', out_dir, 'grad_norm');
    plot_loss_vs_time(runs, methods, labels, colors, out_dir);
    plot_lm_lambda(runs, seeds, out_dir);
    plot_pred_vs_actual(runs, methods, labels, out_dir);
    fprintf('Figures written to %s.\n', out_dir);
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


function plot_curve_vs_iter(runs, methods, labels, colors, field, ylab, out_dir, fname)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 640 420]);
    hold on;
    for m = 1:numel(methods)
        R = runs.(methods{m});
        [xs, mu, lo, hi] = aggregate(R, field);
        if isempty(xs), continue; end
        fill([xs' fliplr(xs')], [lo' fliplr(hi')], colors{m}, ...
             'FaceAlpha', 0.18, 'EdgeColor', 'none');
        plot(xs, mu, '-', 'Color', colors{m}, 'LineWidth', 1.6, 'DisplayName', labels{m});
    end
    set(gca, 'YScale', 'log');
    xlabel('Iteration');
    ylabel(ylab, 'Interpreter', 'latex');
    legend(labels, 'Location', 'northeast');
    grid on; box on;
    exportgraphics(fig, fullfile(out_dir, [fname '.pdf']), 'ContentType', 'vector');
    close(fig);
end


function plot_loss_vs_time(runs, methods, labels, colors, out_dir)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 640 420]);
    hold on;
    for m = 1:numel(methods)
        R = runs.(methods{m});
        for s = 1:numel(R)
            if isempty(R{s}), continue; end
            hh = R{s}.hist;
            plot(hh.time, hh.loss, '-', 'Color', [colors{m} 0.3], 'LineWidth', 0.8, 'HandleVisibility', 'off');
        end
    end
    for m = 1:numel(methods)
        R = runs.(methods{m});
        if ~isempty(R{1})
            hh = R{1}.hist;
            plot(hh.time, hh.loss, '-', 'Color', colors{m}, 'LineWidth', 1.8, 'DisplayName', labels{m});
        end
    end
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Wall-clock time (s)');
    ylabel('Training loss');
    legend(labels, 'Location', 'northeast');
    grid on; box on;
    exportgraphics(fig, fullfile(out_dir, 'loss_vs_time.pdf'), 'ContentType', 'vector');
    close(fig);
end


function plot_lm_lambda(runs, seeds, out_dir)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 640 420]);
    hold on;
    R = runs.lm;
    for s = 1:numel(R)
        if isempty(R{s}), continue; end
        hh = R{s}.hist;
        plot(hh.iter, hh.lambda, '-', 'LineWidth', 1.2, 'DisplayName', sprintf('seed %d', seeds(s)));
    end
    set(gca, 'YScale', 'log');
    xlabel('Iteration');
    ylabel('\lambda_k');
    legend('show', 'Location', 'best');
    grid on; box on;
    exportgraphics(fig, fullfile(out_dir, 'lm_lambda.pdf'), 'ContentType', 'vector');
    close(fig);
end


function plot_pred_vs_actual(runs, methods, labels, out_dir)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 960 320]);
    cfg = default_config();
    for m = 1:numel(methods)
        if isempty(runs.(methods{m}){1}), continue; end
        R = runs.(methods{m}){1};
        problem = build_problem(cfg.seeds(1), cfg);
        yhat_std = nn('forward', R.theta, problem.X_test, problem.arch);
        yhat_mpa = yhat_std * problem.sigma_y + problem.mu_y;
        y_mpa    = problem.y_test * problem.sigma_y + problem.mu_y;

        subplot(1, 3, m);
        scatter(y_mpa, yhat_mpa, 16, 'filled', 'MarkerFaceAlpha', 0.5); hold on;
        lo = min([y_mpa; yhat_mpa]); hi = max([y_mpa; yhat_mpa]);
        plot([lo hi], [lo hi], 'k--', 'LineWidth', 1.0);
        xlabel('Actual strength (MPa)');
        ylabel('Predicted (MPa)');
        title(labels{m});
        axis square; grid on; box on;
    end
    exportgraphics(fig, fullfile(out_dir, 'pred_vs_actual.pdf'), 'ContentType', 'vector');
    close(fig);
end


function problem = build_problem(seed, cfg)
    data = data_loader(seed, cfg.split, cfg.csv_path);
    problem.arch = cfg.arch;
    problem.X_test = data.X_test; problem.y_test = data.y_test;
    problem.mu_y   = data.mu_y;   problem.sigma_y = data.sigma_y;
end


function [xs, mu, lo, hi] = aggregate(R, field)
    keep = cellfun(@(r) ~isempty(r), R);
    R = R(keep);
    if isempty(R), xs = []; mu = []; lo = []; hi = []; return; end
    lens = cellfun(@(r) numel(r.hist.(field)), R);
    K = max(lens);
    vals = nan(K, numel(R));
    for i = 1:numel(R)
        v = R{i}.hist.(field);
        vals(1:numel(v), i) = v;
        vals(numel(v)+1:end, i) = v(end);  % pad with final value
    end
    mu = nanmean(vals, 2);
    sd = nanstd(vals, 0, 2);
    lo = max(mu - sd, eps);
    hi = mu + sd;
    xs = (0:K-1)';
end

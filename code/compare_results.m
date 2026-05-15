function summary_table = compare_results()
%COMPARE_RESULTS Create a comparative table for LM, NCG, and Adam.
%
% This script loads the .mat files saved in the results/ folder and creates
% a summary table comparing:
%
%   - Train RMSE
%   - Validation RMSE
%   - Test RMSE
%   - Runtime
%   - Number of iterations/epochs
%   - Best hyperparameters
%   - Ranking based on test RMSE
%
% The main metric for deciding the best model is mean test RMSE because it
% measures generalization performance on unseen data.

    results_dir = 'results';

    methods = {'lm', 'ncg', 'adam'};
    method_names = {'LM', 'NCG', 'Adam'};

    if ~exist(results_dir, 'dir')
        error('The results/ folder does not exist. Run run_experiments first.');
    end

    all_rows = [];

    for m = 1:numel(methods)
        method = methods{m};
        files = dir(fullfile(results_dir, sprintf('seed*_%s.mat', method)));

        if isempty(files)
            warning('No result files found for method: %s', method);
            continue;
        end

        train_rmse = zeros(numel(files), 1);
        val_rmse   = zeros(numel(files), 1);
        test_rmse  = zeros(numel(files), 1);
        runtime    = zeros(numel(files), 1);
        iters      = zeros(numel(files), 1);
        seeds      = zeros(numel(files), 1);

        for i = 1:numel(files)
            fname = fullfile(results_dir, files(i).name);
            S = load(fname);

            meta = S.meta;
            hist = S.hist;

            train_rmse(i) = meta.train_rmse;
            val_rmse(i)   = meta.val_rmse;
            test_rmse(i)  = meta.test_rmse;
            seeds(i)      = meta.seed;

            % Runtime is taken as the last recorded time in the history.
            if isfield(hist, 'time') && ~isempty(hist.time)
                runtime(i) = hist.time(end);
            else
                runtime(i) = NaN;
            end

            % Iterations or epochs are taken from the history.iter field.
            % For Adam, this may represent epochs depending on the implementation.
            if isfield(hist, 'iter') && ~isempty(hist.iter)
                iters(i) = hist.iter(end);
            else
                iters(i) = NaN;
            end
        end

        row.Method = string(method_names{m});
        row.NumSeeds = numel(files);

        row.TrainRMSE_Mean = mean(train_rmse);
        row.TrainRMSE_Std  = std(train_rmse);

        row.ValRMSE_Mean = mean(val_rmse);
        row.ValRMSE_Std  = std(val_rmse);

        row.TestRMSE_Mean = mean(test_rmse);
        row.TestRMSE_Std  = std(test_rmse);

        row.Runtime_Mean = mean(runtime, 'omitnan');
        row.Runtime_Std  = std(runtime, 'omitnan');

        row.Iterations_Mean = mean(iters, 'omitnan');
        row.Iterations_Std  = std(iters, 'omitnan');

        row.BestHyperparameters = string(get_best_hyperparams(method, results_dir));

        all_rows = [all_rows; struct2table(row)];
    end

    if isempty(all_rows)
        error('No result files were found. Run run_experiments first.');
    end

    % Rank methods by mean test RMSE.
    % Lower test RMSE is better.
    [~, order] = sort(all_rows.TestRMSE_Mean, 'ascend');

    rank = zeros(height(all_rows), 1);
    rank(order) = 1:height(all_rows);

    all_rows.RankByTestRMSE = rank;

    % Sort final table from best to worst.
    summary_table = sortrows(all_rows, 'RankByTestRMSE');

    fprintf('\n============================================================\n');
    fprintf('Comparative Results Summary\n');
    fprintf('Main criterion: lowest mean test RMSE\n');
    fprintf('============================================================\n\n');

    disp(summary_table);

    best_method = summary_table.Method(1);
    best_test_rmse = summary_table.TestRMSE_Mean(1);
    best_test_std = summary_table.TestRMSE_Std(1);

    fprintf('\nBest method based on mean test RMSE: %s\n', best_method);
    fprintf('Mean test RMSE: %.3f ± %.3f MPa\n', best_test_rmse, best_test_std);

    fprintf('\nInterpretation:\n');
    fprintf('The best method is the one with the lowest average test RMSE,\n');
    fprintf('because test RMSE measures performance on unseen data.\n\n');

    % Save table as CSV.
    out_csv = fullfile(results_dir, 'comparative_summary.csv');
    writetable(summary_table, out_csv);
    fprintf('CSV summary saved to: %s\n', out_csv);

    % Save LaTeX-style table.
    out_tex = fullfile(results_dir, 'comparative_summary_latex.txt');
    write_latex_table(summary_table, out_tex);
    fprintf('LaTeX-style table saved to: %s\n', out_tex);
end


function txt = get_best_hyperparams(method, results_dir)
%GET_BEST_HYPERPARAMS Return selected hyperparameters from grid search files.

    txt = "N/A";

    switch lower(method)
        case 'lm'
            f = fullfile(results_dir, 'lm_grid.mat');

            if exist(f, 'file')
                S = load(f);

                if isfield(S, 'best_lm')
                    b = S.best_lm;
                    txt = sprintf('lambda0=%.0e, nu=%g, rho_acc=%.0e', ...
                        b.lambda0, b.nu, b.rho_acc);
                end
            end

        case 'ncg'
            f = fullfile(results_dir, 'ncg_grid.mat');

            if exist(f, 'file')
                S = load(f);

                if isfield(S, 'best_ncg')
                    b = S.best_ncg;
                    txt = sprintf('c1=%.0e, c2=%.2f', b.c1, b.c2);
                end
            end

        case 'adam'
            f = fullfile(results_dir, 'adam_grid.mat');

            if exist(f, 'file')
                S = load(f);

                if isfield(S, 'best_adam')
                    b = S.best_adam;
                    txt = sprintf('lr=%.0e, batch=%d', b.lr, b.batch);
                end
            end
    end
end


function write_latex_table(T, filename)
%WRITE_LATEX_TABLE Write a simple LaTeX-style table to a text file.

    fid = fopen(filename, 'w');

    if fid == -1
        warning('Could not open file for writing: %s', filename);
        return;
    end

    fprintf(fid, 'Method & Train RMSE & Val RMSE & Test RMSE & Runtime & Rank \\\\\n');
    fprintf(fid, '\\hline\n');

    for i = 1:height(T)
        fprintf(fid, '%s & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %.3f $\\pm$ %.3f & %d \\\\\n', ...
            T.Method(i), ...
            T.TrainRMSE_Mean(i), T.TrainRMSE_Std(i), ...
            T.ValRMSE_Mean(i), T.ValRMSE_Std(i), ...
            T.TestRMSE_Mean(i), T.TestRMSE_Std(i), ...
            T.Runtime_Mean(i), T.Runtime_Std(i), ...
            T.RankByTestRMSE(i));
    end

    fclose(fid);
end
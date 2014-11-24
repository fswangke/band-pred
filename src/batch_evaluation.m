clear all;
close all;
clc;

datasets = {
'../data/exp4_range50_rates8_pkts16';
'../data/exp5_range50_rates2_pkts64';
'../data/exp6_range50_rates3_pkts43';
'../data/exp7_range50_rates6_pkts21';
'../data/Nov9_range50_rates4_pkts32';
};

stream_length = [128, %exp4
128, % exp5
129, % exp6 ?
126, % exp7 ?
128 % Nov9
];

regressors = dir('bd_pred*py');
regressor_legend = cell(length(regressors), 1);
errors = zeros(length(regressors), 5);
stds = zeros(length(regressors), 5);

for r = 1 : length(regressors)
    [~, clf_filename, ~] = fileparts(regressors(r).name);
    regressor_name = strrep(clf_filename, 'bd_pred_', '');
    regressor_name = strrep(regressor_name, '_', ' ');
    regressor_legend{r} = regressor_name;
end

% assume training is done
for i = 1 : length(datasets)
    % convert data
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');
    if ~exist(feature_file, 'file')
        fprintf('Datasets %s is not processed.\n', datasets{i});
        continue;
    end

    data = load(feature_file);
    for r = 1 : length(regressors)
        [~, clf_filename, ~] = fileparts(regressors(r).name);
        regressor_name = strrep(clf_filename, 'bd_pred_', '');

        pred_raw = importdata(fullfile(datasets{i}, [regressor_name, '_raw.txt']));
        pred_raw_fft = importdata(fullfile(datasets{i}, [regressor_name, '_raw_fft.txt']));
        pred_smooth = importdata(fullfile(datasets{i}, [regressor_name, '_smooth.txt']));
        pred_smooth_fft = importdata(fullfile(datasets{i}, [regressor_name, '_smooth_fft.txt']));

        pred_raw_error = abs(pred_raw' - data.testY) ./ data.testY;
        pred_raw_fft_error = abs(pred_raw_fft' - data.testY) ./ data.testY;
        pred_smooth_error = abs(pred_smooth' - data.testY) ./ data.testY;
        pred_smooth_fft_error = abs(pred_smooth_fft' - data.testY) ./ data.testY;
        base_error = abs(data.baseY - data.testY) ./ data.testY;

        errors(r, 1) = mean(base_error);
        errors(r, 2) = mean(pred_raw_error);
        errors(r, 3) = mean(pred_raw_fft_error);
        errors(r, 4) = mean(pred_smooth_error);
        errors(r, 5) = mean(pred_smooth_fft_error);

        stds(r, 1) = std(base_error);
        stds(r, 2) = std(pred_raw_error);
        stds(r, 3) = std(pred_raw_fft_error);
        stds(r, 4) = std(pred_smooth_error);
        stds(r, 5) = std(pred_smooth_fft_error);
    end

    % render relative error figure
    [~, dataset_name, ~] = fileparts(datasets{i});
    title_name = ['Dataset: ', strrep(dataset_name, '_', ' ')];
    fig_h = figure(1);
    bar(errors);
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    legend({'Baseline', 'Raw', 'FFT on Raw', 'Smooth', 'FFT on Smooth'});
    xlabel('Algorithm');
    ylabel('Average relative error');
    ax = gca;
    ax.XTickLabel = regressor_legend;
    %title(title_name);
    pdf_name = sprintf('error_%s.pdf', dataset_name);
    print(fig_h, '-dpdf', pdf_name);
    png_name = sprintf('error_%s.png', dataset_name);
    print(fig_h, '-dpng', png_name);
    close(fig_h);

    % render standard deviation figure
    fig_h = figure(2);
    bar(stds);
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    legend({'Baseline', 'Raw', 'FFT on Raw', 'Smooth', 'FFT on Smooth'});
    xlabel('Algorithm');
    ylabel('Standard deviation of relative error');
    ax = gca;
    ax.XTickLabel = regressor_legend;
    %title(title_name);
    pdf_name = sprintf('std_%s.pdf', dataset_name);
    print(fig_h, '-dpdf', pdf_name);
    png_name = sprintf('std_%s.png', dataset_name);
    print(fig_h, '-dpng', png_name);
    close(fig_h);
end

%% Render histogram on one dataset
i = 1;
feature_file = fullfile(datasets{i}, 'data_numpy.mat');
data = load(feature_file);
for r = 1 : length(regressors)
    [~, clf_filename, ~] = fileparts(regressors(r).name);
    regressor_name = strrep(clf_filename, 'bd_pred_', '');

    pred_raw = importdata(fullfile(datasets{i}, [regressor_name, '_raw.txt']));
    pred_raw_fft = importdata(fullfile(datasets{i}, [regressor_name, '_raw_fft.txt']));
    pred_smooth = importdata(fullfile(datasets{i}, [regressor_name, '_smooth.txt']));
    pred_smooth_fft = importdata(fullfile(datasets{i}, [regressor_name, '_smooth_fft.txt']));

    pred_raw_error = abs(pred_raw' - data.testY) ./ data.testY;
    pred_raw_fft_error = abs(pred_raw_fft' - data.testY) ./ data.testY;
    pred_smooth_error = abs(pred_smooth' - data.testY) ./ data.testY;
    pred_smooth_fft_error = abs(pred_smooth_fft' - data.testY) ./ data.testY;
    base_error = abs(data.baseY - data.testY) ./ data.testY;

    [~, dataset_name, ~] = fileparts(datasets{i});
    fig_h = figure(1);
    h1 = histogram(pred_smooth_fft_error);
    hold on;
    h2 = histogram(base_error);
    h1.Normalization = 'count';
    h2.Normalization = 'count';
    h1.BinWidth = 0.01;
    h2.BinWidth = 0.01;
    xlabel('Relative error');
    legend(strrep(regressor_name, '_', ' '), 'Baseline')
    pdf_name = sprintf('histogram_%s.pdf', regressor_name);
    print(fig_h, '-dpdf', pdf_name);
    png_name = sprintf('histogram_%s.png', regressor_name);
    print(fig_h, '-dpng', png_name);
    close(fig_h);
end

%% Render error over regressors
errors = zeros(length(datasets), 5);
stds = zeros(length(datasets), 5);

for r = 1 : length(regressors)
    [~, clf_filename, ~] = fileparts(regressors(r).name);
    regressor_name = strrep(clf_filename, 'bd_pred_', '');
    for i = 1 : length(datasets)
        feature_file = fullfile(datasets{i}, 'data_numpy.mat');
        if ~exist(feature_file, 'file')
            fprintf('Datasets %s is not processed.\n', datasets{i});
            continue;
        end

        data = load(feature_file);

        pred_raw = importdata(fullfile(datasets{i}, [regressor_name, '_raw.txt']));
        pred_raw_fft = importdata(fullfile(datasets{i}, [regressor_name, '_raw_fft.txt']));
        pred_smooth = importdata(fullfile(datasets{i}, [regressor_name, '_smooth.txt']));
        pred_smooth_fft = importdata(fullfile(datasets{i}, [regressor_name, '_smooth_fft.txt']));

        pred_raw_error = abs(pred_raw' - data.testY) ./ data.testY;
        pred_raw_fft_error = abs(pred_raw_fft' - data.testY) ./ data.testY;
        pred_smooth_error = abs(pred_smooth' - data.testY) ./ data.testY;
        pred_smooth_fft_error = abs(pred_smooth_fft' - data.testY) ./ data.testY;
        base_error = abs(data.baseY - data.testY) ./ data.testY;

        errors(i, 1) = mean(base_error);
        errors(i, 2) = mean(pred_raw_error);
        errors(i, 3) = mean(pred_raw_fft_error);
        errors(i, 4) = mean(pred_smooth_error);
        errors(i, 5) = mean(pred_smooth_fft_error);

        stds(i, 1) = std(base_error);
        stds(i, 2) = std(pred_raw_error);
        stds(i, 3) = std(pred_raw_fft_error);
        stds(i, 4) = std(pred_smooth_error);
        stds(i, 5) = std(pred_smooth_fft_error);
    end

    % render relative error figure
    [~, dataset_name, ~] = fileparts(datasets{i});
    fig_h = figure(1);
    bar(errors);
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    legend({'Baseline', 'Raw', 'FFT on Raw', 'Smooth', 'FFT on Smooth'});
    xlabel('Number of Probing Rates');
    ylabel('Average relative error');
    ax = gca;
    ax.XTickLabel = {'8', '2', '3', '6', '4'};
    pdf_name = sprintf('error_%s.pdf', regressor_name);
    print(fig_h, '-dpdf', pdf_name);
    png_name = sprintf('error_%s.png', regressor_name);
    print(fig_h, '-dpng', png_name);
    close(fig_h);

    fig_h = figure(2);
    bar(stds);
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    legend({'Baseline', 'Raw', 'FFT on Raw', 'Smooth', 'FFT on Smooth'});
    xlabel('Number of Probing Rates');
    ylabel('Standard deviation of relative error');
    ax = gca;
    ax.XTickLabel = {'8', '2', '3', '6', '4'};
    pdf_name = sprintf('std_%s.pdf', regressor_name);
    print(fig_h, '-dpdf', pdf_name);
    png_name = sprintf('std_%s.png', regressor_name);
    print(fig_h, '-dpng', png_name);
    close(fig_h);
end

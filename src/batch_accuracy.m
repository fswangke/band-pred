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
errors = zeros(length(datasets), length(regressors), 5);
stds = zeros(length(datasets), length(regressors), 5);

for r = 1 : length(regressors)
    [~, clf_filename, ~] = fileparts(regressors(r).name);
    regressor_name = strrep(clf_filename, 'bd_pred_', '');
    regressor_name = strrep(regressor_name, '_', ' ');
    regressor_legend{r} = regressor_name;
end

% assume training is done
total_stream_number = zeros(length(datasets), 1);
valid_stream_number = zeros(length(datasets), 1);
train_stream_number = zeros(length(datasets), 1);
test_stream_number = zeros(length(datasets), 1);

for i = 1 : length(datasets)
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

        errors(i, r, 1) = mean(base_error);
        errors(i, r, 2) = mean(pred_raw_error);
        errors(i, r, 3) = mean(pred_raw_fft_error);
        errors(i, r, 4) = mean(pred_smooth_error);
        errors(i, r, 5) = mean(pred_smooth_fft_error);

        stds(i, r, 1) = std(base_error);
        stds(i, r, 2) = std(pred_raw_error);
        stds(i, r, 3) = std(pred_raw_fft_error);
        stds(i, r, 4) = std(pred_smooth_error);
        stds(i, r, 5) = std(pred_smooth_fft_error);
    end

    % count stream numbers
    raw_streams_filename = fullfile(datasets{i}, 'allrecv_streams_strip.dat');
    raw_stream_data = importdata(raw_streams_filename);
    raw_pid = unique(raw_stream_data(:, 1));
    total_stream_number(i) = length(raw_pid);
    valid_stream_number(i) = length(data.testY) + length(data.trainY);
    train_stream_number(i) = length(data.trainY);
    test_stream_number(i) = length(data.testY);
end

mean_error = squeeze(mean(errors, 1));
mean_std = squeeze(mean(stds, 1));
mean_prob = abs(mean_std - mean_error) ./ mean_error;

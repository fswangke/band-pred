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
errors = zeros(length(datasets), 5, length(regressors));
stds = zeros(length(datasets), 5, length(regressors));

parfor i = 1 : length(datasets)
    % convert data
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');
    if ~exist(feature_file, 'file')
        convert_smooth_data_to_numpy(datasets{i}, stream_length(i));
    end

    data = load(feature_file);
    for r = 1 : length(regressors)
        command = sprintf('python %s %s\n', regressors(r).name, datasets{i});
        system(command);
        disp(command);
    end
end

for i = 1 : length(datasets)
    % convert data
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');

    data = load(feature_file);
    for r = 1 : length(regressors)
        [~, regressors_name, ~] = fileparts(regressors(r).name);
        regressor_name = strrep(regressors_name, 'bd_pred_', '');

        pred_raw = importdata(fullfile(datasets{i}, [regressor_name, '_raw.txt']));
        pred_raw_fft = importdata(fullfile(datasets{i}, [regressor_name, '_raw_fft.txt']));
        pred_smooth = importdata(fullfile(datasets{i}, [regressor_name, '_smooth.txt']));
        pred_smooth_fft = importdata(fullfile(datasets{i}, [regressor_name, '_smooth_fft.txt']));

        pred_raw_error = abs(pred_raw' - data.testY) ./ data.testY;
        pred_raw_fft_error = abs(pred_raw_fft' - data.testY) ./ data.testY;
        pred_smooth_error = abs(pred_smooth' - data.testY) ./ data.testY;
        pred_smooth_fft_error = abs(pred_smooth_fft' - data.testY) ./ data.testY;
        base_error = abs(data.baseY - data.testY) ./ data.testY;

        errors(i, 1, r) = mean(base_error);
        errors(i, 2, r) = mean(pred_raw_error);
        errors(i, 3, r) = mean(pred_raw_fft_error);
        errors(i, 4, r) = mean(pred_smooth_error);
        errors(i, 5, r) = mean(pred_smooth_fft_error);

        stds(i, 1, r) = std(base_error);
        stds(i, 2, r) = std(pred_raw_error);
        stds(i, 3, r) = std(pred_raw_fft_error);
        stds(i, 4, r) = std(pred_smooth_error);
        stds(i, 5, r) = std(pred_smooth_fft_error);
    end
end

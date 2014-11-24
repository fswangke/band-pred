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
times = zeros(length(datasets), length(regressors), 6);

for i = 1 : length(datasets)
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');
    if ~exist(feature_file, 'file')
        fprintf('Datasets %s is not processed.\n', datasets{i});
        continue;
    end

    data = load(feature_file);
    [n_streams, n_dim] = size(data.trainX_raw);
    for r = 1 : length(regressors)
        [~, clf_filename, ~] = fileparts(regressors(r).name);
        regressor_name = strrep(clf_filename, 'bd_pred_', '');

        timing = importdata(fullfile(datasets{i}, [regressor_name, '_time.txt']));
        times(i, r, :) = timing ./ n_streams;
    end
end

avg_time = squeeze(mean(times, 1)) * 1e6;

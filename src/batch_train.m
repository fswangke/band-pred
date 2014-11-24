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

% training in parallel
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

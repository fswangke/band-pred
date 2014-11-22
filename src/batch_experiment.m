clear all;
close all;
clc;

datasets = {
'../data/exp1_range50_rates3_pkts32';
'../data/exp2_range50_rates4_pkts64';
'../data/exp3_range50_rates4_pkts32'
};

stream_length = [96, 256, 128];

regressors = dir('bd_pred*py');

for i = 1 : length(datasets)
    % convert data
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');
    if ~exist(feature_file, 'file')
        convert_data_to_numpy(datasets{i}, stream_length(i));
    end
    
    load(feature_file);
    for j = 1 : length(regressors)
        command = sprintf('python %s %s', regressors(j).name, datasets{i});
        %disp(command);
        % collect results
        [~, algorithm_name, ~] = fileparts(regressors(j).name);
        algorithm_name = strrep(algorithm_name, 'bd_pred_', '');
        prediction_filename = fullfile(datasets{i}, ...
            [algorithm_name, '_pred.txt']);
        if ~exist(prediction_filename, 'file')
            fprintf('Results of %s on dataset %s not found.\n', ...
                algorithm_name, datasets{i});
            continue;
        end
        pred = importdata(prediction_filename);
        err = abs(pred - testY) ./ testY;
        fprintf('Average error rate of %s on dataset %s is %f.\n', ...
            algorithm_name, datasets{i}, mean(err));
        % analyze results
    end
end

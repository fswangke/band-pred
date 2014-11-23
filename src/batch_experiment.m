clear all;
close all;
clc;

% We aim at process raw and smooth data, and compare them

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
128, % Nov9
];


lasso_errors = zeros(length(datasets), 5);
nnr_errors = zeros(length(datasets), 5);
lasso_err_std = zeros(length(datasets), 5);
nnr_err_std = zeros(length(datasets), 5);

for i = 1 : length(datasets)
    % convert data
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');
    if ~exist(feature_file, 'file')
        convert_smooth_data_to_numpy(datasets{i}, stream_length(i));
    end

    data = load(feature_file);
    % right now we only have lasso and nnr working :(
    command = sprintf('python ./bd_pred_lasso_smooth.py %s\n', datasets{i});
    system(command);
    % collect results
    lasso_raw = importdata(fullfile(datasets{i}, 'lasso_raw.txt'));
    lasso_raw_error = abs(lasso_raw' - data.testY) ./ data.testY;

    lasso_raw_fft = importdata(fullfile(datasets{i}, 'lasso_raw_fft.txt'));
    lasso_raw_fft_error = abs(lasso_raw_fft' - data.testY) ./ data.testY;

    lasso_smooth = importdata(fullfile(datasets{i}, 'lasso_smooth.txt'));
    lasso_smooth_error = abs(lasso_smooth' - data.testY) ./ data.testY;

    lasso_smooth_fft = importdata(fullfile(datasets{i}, 'lasso_smooth_fft.txt'));
    lasso_smooth_fft_error = abs(lasso_smooth_fft' - data.testY) ./ data.testY;

    base_error = abs(data.baseY - data.testY) ./ data.testY;

    lasso_errors(i, :) = [mean(base_error), mean(lasso_raw_error), mean(lasso_raw_fft_error), mean(lasso_smooth_error), mean(lasso_smooth_fft_error)];
    lasso_err_std(i, :) = [std(base_error), std(lasso_raw_error), std(lasso_raw_fft_error), std(lasso_smooth_error), std(lasso_smooth_fft_error)];
    
    figure;
    h1 = histogram(base_error);
    hold on;
    h2 = histogram(lasso_raw_error);
    h3 = histogram(lasso_raw_fft_error);
    h4 = histogram(lasso_smooth_error);
    h5 = histogram(lasso_smooth_fft_error);
    h1.BinWidth = 0.01;
    h2.BinWidth = 0.01;
    h3.BinWidth = 0.01;
    h4.BinWidth = 0.01;
    h5.BinWidth = 0.01;
    

    legend('Baseline', 'Lasso on raw', 'Lasso on Raw-FFT', 'Lasso on smooth', 'Lasso on Smooth-FFT');
    command = sprintf('python ./bd_pred_nnr.py %s\n', datasets{i});
    system(command);
    % collect results
    nnr_raw = importdata(fullfile(datasets{i}, 'nnr_raw.txt'));
    nnr_raw_error = abs(nnr_raw' - data.testY) ./ data.testY;

    nnr_raw_fft = importdata(fullfile(datasets{i}, 'nnr_raw_fft.txt'));
    nnr_raw_fft_error = abs(nnr_raw_fft' - data.testY) ./ data.testY;

    nnr_smooth = importdata(fullfile(datasets{i}, 'nnr_smooth.txt'));
    nnr_smooth_error = abs(nnr_smooth' - data.testY) ./ data.testY;

    nnr_smooth_fft = importdata(fullfile(datasets{i}, 'nnr_smooth_fft.txt'));
    nnr_smooth_fft_error = abs(nnr_smooth_fft' - data.testY) ./ data.testY;

    nnr_errors(i, :) = [mean(base_error), mean(nnr_raw_error), mean(nnr_raw_fft_error), mean(nnr_smooth_error), mean(nnr_smooth_fft_error)];
    nnr_err_std(i, :) = [std(base_error), std(nnr_raw_error), std(nnr_raw_fft_error), std(nnr_smooth_error), std(nnr_smooth_fft_error)];
end


figure;
bar(lasso_errors);
xlabel('Datasets');
ylabel('Average relative error');
legend('Baseline', 'Lasso on raw', 'Lasso on Raw-FFT', 'Lasso on smooth', 'Lasso on Smooth-FFT');
ax = gca;
ax.XTickLabel = {'rates8', 'rates2', 'rates3', 'rates6', 'rates4'};

figure;
bar(lasso_err_std);
xlabel('Datasets');
ylabel('Standard variation of relative error');
legend('Baseline', 'Lasso on raw', 'Lasso on Raw-FFT', 'Lasso on smooth', 'Lasso on Smooth-FFT');
ax = gca;
ax.XTickLabel = {'rates8', 'rates2', 'rates3', 'rates6', 'rates4'};

figure;
bar(nnr_errors);
xlabel('Datasets');
ylabel('Average relative error');
legend('Baseline', 'NNR on raw', 'NNR on Raw-FFT', 'NNR on smooth', 'NNR on Smooth-FFT');
ax = gca;
ax.XTickLabel = {'rates8', 'rates2', 'rates3', 'rates6', 'rates4'};

figure;
bar(nnr_err_std);
xlabel('Datasets');
ylabel('Standard variation of relative error');
legend('Baseline', 'NNR on raw', 'NNR on Raw-FFT', 'NNR on smooth', 'NNR on Smooth-FFT');
ax = gca;
ax.XTickLabel = {'rates8', 'rates2', 'rates3', 'rates6', 'rates4'};

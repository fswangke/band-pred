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
nbins = 15;
xbins = zeros(length(regressors) + 1, nbins);
hists = zeros(length(regressors)+1, nbins);

for r = 1 : length(regressors)
    [~, clf_filename, ~] = fileparts(regressors(r).name);
    regressor_name = strrep(clf_filename, 'bd_pred_', '');
    regressor_name = strrep(regressor_name, '_', ' ');
    regressor_legend{r} = regressor_name;
end

% assume training is done
for i = 1 : 1
    feature_file = fullfile(datasets{i}, 'data_numpy.mat');
    if ~exist(feature_file, 'file')
        fprintf('Datasets %s is not processed.\n', datasets{i});
        continue;
    end

    data = load(feature_file);
    for r = 1 : length(regressors)
        [~, clf_filename, ~] = fileparts(regressors(r).name);
        regressor_name = strrep(clf_filename, 'bd_pred_', '');

        pred_smooth_fft = importdata(fullfile(datasets{i}, [regressor_name, '_smooth_fft.txt']));

        pred_smooth_fft_error = abs(pred_smooth_fft' - data.testY) ./ data.testY;
        base_error = abs(data.baseY - data.testY) ./ data.testY;

        [hists(r, :), xbins(r, :)] = hist(pred_smooth_fft_error, nbins);
    end
end
[hists(end, :), xbins(end, :)] = hist(base_error, nbins);
fig_h = figure;
plot(xbins(1, :), hists(1, :), ...
xbins(2, :), hists(2, :), ...
xbins(3, :), hists(3, :), ...
xbins(4, :), hists(4, :), ...
xbins(5, :), hists(5, :), ...
xbins(6, :), hists(6, :), ...
xbins(7, :), hists(7, :));
regressor_legend{7} = 'Baseline';
legend(regressor_legend);
xlabel('Relative error');
print(fig_h, '-dpdf', 'hist_all.pdf');
print(fig_h, '-dpng', 'hist_all.png');
close(fig_h);
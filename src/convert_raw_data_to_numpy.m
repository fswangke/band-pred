function convert_raw_data_to_numpy(datapath, stream_length)

if nargin < 2
    stream_length = 96;
end

if ~exist(datapath, 'dir')
    error('Datapath %s not found.\n', datapath);
end

% convert allrecv_stremas.dat
raw_streams_filename = fullfile(datapath, 'allrecv_streams.dat');
striped_streams_filename = fullfile(datapath, 'allrecv_streams_strip.dat');

if ~exist(raw_streams_filename, 'file')
    error('Raw stream data file %s not found.\n', raw_streams_filename);
end

if ~exist(striped_streams_filename, 'file')
    strip_command = sprintf('python strip_raw_streams.py %s %s\n', ...
        raw_streams_filename, ...
        striped_streams_filename);
    fprintf('Stripping raw streams files.\n');
    system(strip_command);
end

% load raw data files
ground_truth_data = importdata(fullfile(datapath, 'actual_ab.dat'));
error_data = importdata(fullfile(datapath, 'error.dat'));
stream_data = importdata(striped_streams_filename);
fprintf('Loaded input data files.\n');

pid = unique(stream_data(:, 1));
sendgap = stream_data(:, 3);
recvgap = stream_data(:, 5);

data = [];
for i = 1 : length(pid)
    cur_data.pid = pid(i);
    cur_index = stream_data(:, 1) == pid(i);
    if sum(cur_index) ~= stream_length
        fprintf('Incomplete stream %d will be discarded.\n', pid(i));
        continue;
    end

    error_idx = find(error_data(:, 1) == pid(i));
    if isempty(error_idx)
        fprintf('Baseline prediction for stream %d not found.\n', pid(i));
        continue;
    else
        cur_data.base_bd = error_data(error_idx, 2);
    end
    cur_data.sendgap = sendgap(cur_index);
    cur_data.recvgap = recvgap(cur_index);
    cur_data.actual_bd = ground_truth_data(ground_truth_data(:, 1) == pid(i), 2);
    data = [data; cur_data];
end
fprintf('Assembled streams.\n');

% split training and test datasets
base_pred = extractfield(data, 'base_bd');
% TODO: why are we missing entries in the baseline predictions
train = data(isnan(base_pred));
test = data(~isnan(base_pred));

% extract fields
sendgap = extractfield(train, 'sendgap');
sendgap = reshape(sendgap, stream_length, numel(sendgap) / stream_length)';
recvgap = extractfield(train, 'recvgap');
recvgap = reshape(recvgap, stream_length, numel(recvgap) / stream_length)';
trainY = extractfield(train, 'actual_bd')';
trainX = [sendgap, recvgap];

sendgap = extractfield(test, 'sendgap');
sendgap = reshape(sendgap, stream_length, numel(sendgap) / stream_length)';
recvgap = extractfield(test, 'recvgap');
recvgap = reshape(recvgap, stream_length, numel(recvgap) / stream_length)';
testY = extractfield(test, 'actual_bd')';
baseY = extractfield(test, 'base_bd')';
testX = [sendgap, recvgap];

% feature normalization
features = normc([trainX; testX]);
trainXN = features(1:size(trainX, 1), :);
testXN = features(size(trainX, 1)+1:end, :);
fprintf('Feature normalized.\n');

numpy_filename = fullfile(datapath, 'data_numpy.mat');
save(numpy_filename, 'trainX', 'trainY', 'testX', 'testY', 'baseY', ...
'trainXN', 'testXN');
end

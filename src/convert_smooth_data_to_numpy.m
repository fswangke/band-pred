function convert_smooth_data_to_numpy(datapath, stream_length)

if nargin < 2
    stream_length = 96;
end

if ~exist(datapath, 'dir')
    error('Datapath %s not found.\n', datapath);
end

% convert allrecv_stremas.dat
smooth_streams_filename = fullfile(datapath, 'streams_pass1.dat');
striped_smooth_streams_filename = fullfile(datapath, 'streams_pass1_strip.dat');

raw_streams_filename = fullfile(datapath, 'allrecv_streams.dat');
striped_raw_streams_filename = fullfile(datapath, 'allrecv_streams_strip.dat');

if ~exist(raw_streams_filename, 'file')
    error('Raw data file %s not found.\n', raw_streams_filename);
end

if ~exist(striped_raw_streams_filename, 'file')
    strip_command = sprintf('python strip_raw_streams.py %s %s\n', ...
        raw_streams_filename, ...
        striped_raw_streams_filename);
    fprintf('Stripping raw streams files.\n');
    system(strip_command);
end

if ~exist(smooth_streams_filename, 'file')
    error('Smooth data file %s not found.\n', smooth_streams_filename);
end

if ~exist(striped_smooth_streams_filename, 'file')
    strip_command = sprintf('python strip_smooth_streams.py %s %s\n', ...
        smooth_streams_filename, ...
        striped_smooth_streams_filename);
    fprintf('Stripping smooth streams files.\n');
    system(strip_command);
end

% load data files
error_data = importdata(fullfile(datapath, 'error.dat'));
raw_stream_data = importdata(striped_raw_streams_filename);
smooth_stream_data = importdata(striped_smooth_streams_filename);
fprintf('Loaded input data files.\n');

valid_pid = unique(error_data(:, 1));

raw_pid = raw_stream_data(:, 1);
raw_sendgap = raw_stream_data(:, 3);
raw_recvgap = raw_stream_data(:, 5);

smooth_pid = smooth_stream_data(:, 1);
smooth_sendgap = smooth_stream_data(:, 3);
smooth_recvgap = smooth_stream_data(:, 4);

data = [];
for i = 1 : length(valid_pid)
    cur_data.pid = valid_pid(i);

    raw_index = raw_pid == valid_pid(i);
    if sum(raw_index) ~= stream_length
        fprintf('Raw stream %d missing packet. Discarded.\n', cur_data.pid);
        continue;
    end

    smooth_index = smooth_pid == valid_pid(i);
%     if sum(smooth_index) ~= stream_length
%         fprintf('Smooth stream %d missing %d packet.\n', cur_data.pid, ...
%             stream_length - sum(smooth_index));
%     end

    error_idx = error_data(:, 1) == valid_pid(i);
    cur_data.base_bd = error_data(error_idx, 2);
    cur_data.raw_sendgap = raw_sendgap(raw_index);
    cur_data.raw_recvgap = raw_recvgap(raw_index);
    cur_data.smooth_sendgap = smooth_sendgap(smooth_index);
    cur_data.smooth_recvgap = smooth_recvgap(smooth_index);
    if sum(smooth_index) < stream_length
        end_sendgap = smooth_sendgap(end);
        end_recvgap = smooth_recvgap(end);
        cur_data.smooth_sendgap(stream_length) = end_sendgap;
        cur_data.smooth_recvgap(stream_length) = end_recvgap;
    end
    cur_data.actual_bd = error_data(error_idx, 3);

    % append current valid stream to array
    data = [data; cur_data];
end
fprintf('Assembled streams.\n');

% split training and test datasets
index = randperm(length(data));
traing_set_size = round(length(data) * 0.7);
train_index = index(1: traing_set_size);
test_index = index(traing_set_size + 1 : end);

train = data(train_index);
test = data(test_index);

% extract field
raw_sendgap = extractfield(train, 'raw_sendgap');
raw_sendgap = reshape(raw_sendgap, stream_length, numel(raw_sendgap) / stream_length)';
raw_recvgap = extractfield(train, 'raw_recvgap');
raw_recvgap = reshape(raw_recvgap, stream_length, numel(raw_recvgap) / stream_length)';
smooth_sendgap = extractfield(train, 'smooth_sendgap');
smooth_sendgap = reshape(smooth_sendgap, stream_length, numel(smooth_sendgap) / stream_length)';
smooth_recvgap = extractfield(train, 'smooth_recvgap');
smooth_recvgap = reshape(smooth_recvgap, stream_length, numel(smooth_recvgap) / stream_length)';
trainY = extractfield(train, 'actual_bd');
trainX_raw = [raw_sendgap, raw_recvgap];
trainX_smooth = [smooth_sendgap, smooth_recvgap];

raw_sendgap = extractfield(test, 'raw_sendgap');
raw_sendgap = reshape(raw_sendgap, stream_length, numel(raw_sendgap) / stream_length)';
raw_recvgap = extractfield(test, 'raw_recvgap');
raw_recvgap = reshape(raw_recvgap, stream_length, numel(raw_recvgap) / stream_length)';
smooth_sendgap = extractfield(test, 'smooth_sendgap');
smooth_sendgap = reshape(smooth_sendgap, stream_length, numel(smooth_sendgap) / stream_length)';
smooth_recvgap = extractfield(test, 'smooth_recvgap');
smooth_recvgap = reshape(smooth_recvgap, stream_length, numel(smooth_recvgap) / stream_length)';
testY = extractfield(test, 'actual_bd');
testX_raw = [raw_sendgap, raw_recvgap];
testX_smooth = [smooth_sendgap, smooth_recvgap];
baseY = extractfield(test, 'base_bd');

numpy_filename = fullfile(datapath, 'data_numpy.mat');
save(numpy_filename, 'trainX_raw', 'trainX_smooth', 'trainY', 'testX_raw', ...
    'testX_smooth', 'testY', 'baseY');
end

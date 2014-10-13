function process_data()
if exist('data.mat', 'file')
    load data.mat;
else
    data = parse_data();
end

split_data(data);
end

function data = parse_data()% load raw data files
G = importdata('../data/actual_ab.dat');
E = importdata('../data/error.dat');
R = importdata('../data/conv.dat');

pid = unique(R(:, 1));
sendgap = R(:, 3);
recvgap = R(:, 5);

data = [];
h = waitbar(0, 'Parsing data');
for i = 1 : length(pid)
    cur_data.pid = pid(i);
    if sum(R(:, 1) == pid(i)) ~= 96
        fprintf('In complete stream %d will be discarded.\n', pid(i));
        continue;
    end
    cur_data.sendgap = sendgap(R(:, 1) == pid(i));
    cur_data.recvgap = recvgap(R(:, 1) == pid(i));
    cur_data.actual_bd = G(G(:, 1) == pid(i), 2);
    idx = find(E(:, 1) == pid(i));
    if isempty(idx)
        fprintf('Baseline prediction not found.\n');
        cur_data.base_bd = NaN;
    else
        cur_data.base_bd = E(idx, 2);
    end
    data = [data; cur_data];
    waitbar(i / length(pid));
end
close(h);
end

function split_data(data)
% split training set and test set
base_pred = extractfield(data, 'base_bd');
train = data(isnan(base_pred));
test = data(~isnan(base_pred));

% extract fields
sendgap = extractfield(train, 'sendgap');
sendgap = reshape(sendgap, 96, numel(sendgap) / 96)';
recvgap = extractfield(train, 'recvgap');
recvgap = reshape(recvgap, 96, numel(recvgap) / 96)';
trainY = extractfield(train, 'actual_bd')';
trainX = [sendgap, recvgap];

sendgap = extractfield(test, 'sendgap');
sendgap = reshape(sendgap, 96, numel(sendgap) / 96)';
recvgap = extractfield(test, 'recvgap');
recvgap = reshape(recvgap, 96, numel(recvgap) / 96)';
testY = extractfield(test, 'actual_bd')';
baseY = extractfield(test, 'base_bd')';
testX = [sendgap, recvgap];

% feature normalization
features = normc([trainX; testX]);
trainXN = features(1:size(trainX, 1), :);
testXN = features(size(trainX, 1):end, :);

save('data_numpy.mat', 'trainX', 'trainY', 'testX', 'testY', 'baseY', ...
    'trainXN', 'testXN');
end
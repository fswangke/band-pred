function process_data()
if exist('data.mat', 'file')
    load data.mat;
else
    data = parse_data();
	save('data.mat', 'data');
end

split_data(data);

% visualize conv.dat
% R = importdata('../data/conv.dat');
% plot(R)
% pause;
% visualize_clean_data(data);

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
        fprintf('Incomplete stream %d will be discarded.\n', pid(i));
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
testXN = features(size(trainX, 1)+1:end, :);

save('data_numpy.mat', 'trainX', 'trainY', 'testX', 'testY', 'baseY', ...
    'trainXN', 'testXN');
end

function visualize_clean_data(data)
%for i = 1:length(data)
fprintf('%d\n', length(data));
for i = 2:51
	d = data(i);
	h = figure(d.pid);
	sendtime = cumsum(d.sendgap);
	recvtime = cumsum(d.recvgap);
	fprintf('total length: %d\n', length(d.sendgap) + length(d.recvgap));
	idx = 1:length(sendtime);
	%plot(idx, sendtime ./ idx', idx, recvtime ./ idx')
	plot(idx, d.sendgap, idx, d.recvgap)
	%plot(idx, d.recvgap);
	set(gca, 'FontSize', 20);
	xlabel('Packet index', 'FontSize', 20);
	ylabel('Time', 'FontSize', 20);
	title('Received gap', 'FontSize', 20);
	h_legend = legend('Send time', 'Receive time');
	set(h_legend, 'FontSize', 20);

	pause;
end
end

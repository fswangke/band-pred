function visualize_data(pid, pos, sendgap, daggap, rcvtsgap, seqno)
N = length(pid);
assert(N == length(pid));
assert(N == length(pos));
assert(N == length(sendgap));
assert(N == length(daggap));
assert(N == length(rcvtsgap));
assert(N == size(seqno, 1));

probes = unique(pid);

for i = 1 : length(probes)
    idx = find(pid == probes(i));
    h = figure(probes(i));
    sendtime = cumsum(sendgap(idx));
    recvtime = cumsum(rcvtsgap(idx));
    dagtime = cumsum(daggap(idx));
    idx = 1 : length(sendtime);
    plot(idx, sendtime ./ idx', idx, recvtime./ idx', idx, dagtime ./ idx');
    %plot(idx, speed);
    xlabel('Packet index');
    ylabel('Time');
    legend('Send time', 'Receive time', 'Ground truth');
    print(h, sprintf('probe-%d.png', probes(i)), '-dpng');
    close(h);
end
end
function [pid, pos, sendgap, daggap, rcvtsgap, seqno] = parse_data(filename)

if ~exist(filename, 'file')
    error('file not found');
end
pid = [];
pos = [];
sendgap = [];
daggap = [];
rcvtsgap = [];
seqno = [];
fid = fopen(filename, 'r');
tline = fgets(fid);
while ischar(tline)
    data= sscanf(tline, 'pid= %d pos= %d sendgap= %d daggap= %d rcvtsgap= %d seqno= %f %f');
    pid = [pid; data(1)];
    pos = [pos; data(2)];
    sendgap = [sendgap; data(3)];
    daggap = [daggap; data(4)];
    rcvtsgap = [rcvtsgap; data(5)];
    seqno = [seqno; data(6:7)'];
    
    tline = fgets(fid);
end
fclose(fid);
end
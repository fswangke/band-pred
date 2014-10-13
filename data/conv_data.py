fid = open('./allrecv_streams.dat')
fout = open('./conv.dat', 'w')

for line in fid:
    line = str.replace(line, "pid=", " ")
    line = str.replace(line, 'pos=', ' ')
    line = str.replace(line, 'sendgap=', ' ')
    line = str.replace(line, 'daggap=', ' ')
    line = str.replace(line, 'rcvtsgap=', ' ')
    line = str.replace(line, 'seqno=', ' ')
    fout.write(line)

fid.close()
fout.close()

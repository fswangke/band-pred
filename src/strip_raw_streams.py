import sys


def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    fid = open(input_filename, 'r')
    fout = open(output_filename, 'w')

    for line in fid:
        line = str.replace(line, 'pid=', ' ')
        line = str.replace(line, 'pos=', ' ')
        line = str.replace(line, 'sendgap=', ' ')
        line = str.replace(line, 'daggap=', ' ')
        line = str.replace(line, 'rcvtsgap=', ' ')
        line = str.replace(line, 'seqno=', ' ')
        fout.write(line)

    fid.close()
    fout.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: strip_raw_streams.py input output")
        exit()
    else:
        main()

import sys


def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    fid = open(input_filename, 'r')
    fout = open(output_filename, 'w')

    for line in fid:
        line = str.replace(line, 'finallypass1', '')
        line = str.replace(line, 'pid=', ' ')
        line = str.replace(line, 'nospike_cnt=', ' ')
        line = str.replace(line, 'nospike_sendgap=', ' ')
        line = str.replace(line, 'nospike_recvgap=', ' ')
        line = str.replace(line, 'lowestrate_gap=', ' ')
        line = str.replace(line, 'highestrate_gap=', ' ')
        fout.write(line)

    fid.close()
    fout.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: strip_smooth_streams.py input output")
        exit()
    else:
        main()

import sys
import collections

def main(vocab_size, ofile, inputs):
    ctr = collections.Counter()
    for file in inputs:
        with open(file, 'rt', encoding='utf-8') as of:
            for line in of:
                parts = line.rstrip('\n').split(' ')
                for p in parts:
                    ctr[p] += 1
    with open(ofile, 'wt', encoding='utf-8') as of:
        of.write('name\tfreq\n')
        of.write('<pad>\t0\n')
        of.write('<u>\t0\n')
        of.write('<b>\t0\n')
        of.write('<e>\t0\n')
        for x, cnt in ctr.most_common(vocab_size - 4):
            of.write(f"{x}\t{cnt}\n")

if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3:])
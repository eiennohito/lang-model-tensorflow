import sys
import gzip
import argparse

def main(fld, files, out):
    space = False

    for file in files:
        xopen = open
        if file.endswith('.gz'):
            xopen = gzip.open
        with xopen(file, 'rt', encoding='utf-8', errors='ignore') as fl:
            for line in fl:
                if line.startswith('#') or line.startswith('+') or line.startswith('*'):
                    continue
                elif line.startswith("EOS"):
                    out.write('\n')
                    space = False
                else:
                    data = line.split(' ')
                    if space:
                        out.write(' ')
                    out.write(data[fld])
                    space = True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o")
    p.add_argument('field', type=int)
    p.add_argument('input', nargs='*')
    args = p.parse_args()

    if args.output is None:
        main(args.field, args.input, sys.stdout)
    else:
        with open(args.output, 'wt', encoding='utf-8') as of:
            main(args.field, args.input, of)

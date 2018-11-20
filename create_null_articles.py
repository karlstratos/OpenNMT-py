import sys

fname = sys.argv[1]
num_articles = int(sys.argv[2])

with open(fname, 'w') as outf:
    for _ in range(num_articles):
        outf.write(".\n")

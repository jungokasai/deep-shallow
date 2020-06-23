import re, sys

def dehyphenate(sent):
    return re.sub(r'(\S)-(\S)', r'\1 ##AT##-##AT## \2', sent).replace('##AT##', '@')
with open(sys.argv[1], 'rt') as fin:
    sents = fin.readlines()
with open(sys.argv[2], 'wt') as fout:
    for sent in sents:
        fout.write(dehyphenate(sent))


from fairseq import pybleu
import sys

def get_results(gold_file, pred_file):
    results = []
    with open(gold_file) as fin_g:
        with open(pred_file) as fin_p:
            for line_g, line_p in zip(fin_g, fin_p):
                results.append((line_g.strip(), line_p.strip()))
    return results
            

scorer = pybleu.PyBleuScorer()
results = get_results(sys.argv[1], sys.argv[2])
ref, out = zip(*results)
print('BLEU4 = {:2.2f}, '.format(scorer.score(ref, out)))

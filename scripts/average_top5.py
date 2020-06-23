import subprocess, os
import numpy as np
import sys
import glob, re

Model_Path = sys.argv[1]
k = 5
out_file = os.path.join(Model_Path, 'checkpoint_top{}_average.pt'.format(k))
if os.path.exists(out_file):
    print('Do not overwrite {}'.format(out_file))
    raise

valid_files = glob.glob(Model_Path + "/valid*[1-9]*.out")
scores = []
scores_files = []
for valid_file in valid_files:
    with open(valid_file) as fin:
        for line in fin:
            if 'BLEU4 =' in line:
                line = line.strip()
                bleu_idx = [(int(m.start(0)), int(m.end(0))) for m in re.finditer('BLEU4 = ', line)][0][1]
                score = float(line[bleu_idx:bleu_idx+5].replace(',', ''))
                scores.append(score)
                scores_files.append((score, valid_file))
scores = np.array(scores)
order = (-scores).argsort()
print('Top {} epochs'.format(k))
top_k_order = order[:k]
print(scores[top_k_order])
model_files = []
for i in list(top_k_order):
    epoch = int(scores_files[i][1].replace('.out', '').split('_')[-1])
    #epoch = int(valid_files[i].replace('.out', '').split('_')[-1])
    model_files.append(os.path.join(Model_Path, 'checkpoint{}.pt'.format(epoch)))

command = 'python scripts/average_checkpoints.py'
#command = 'python ~/projects/fairseq_master/fairseq-py/scripts/average_checkpoints.py'
command += ' --inputs {}'.format(' '.join(model_files))
command += ' --output {}'.format(os.path.join(Model_Path, 'checkpoint_top{}_average.pt'.format(k)))
print(command)
output = subprocess.check_output(command, shell=True, universal_newlines=True)

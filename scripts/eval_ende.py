import sys, os, subprocess

model_dir = sys.argv[1]
# inference
command = 'python generate.py ~/data-bin/master/wmt16.en-de.dist/ --path {}/checkpoint_top5_average.pt --task translation --max-sentences 10 --beam 5 --gen-subset test --remove-bpe  --lenpen 1.0 > {}/ende.out'.format(model_dir, model_dir)
#command = 'python generate.py ~/nmt_data/char_space/wmt15_de_en/ --path {}/checkpoint58.pt --task translation --max-sentences 10 --beam 5 --gen-subset test --remove-bpe char --lenpen 1.0 > {}/ende.out'.format(model_dir, model_dir)
#command = 'python generate.py ~/data-bin/master/wmt16.en-de/ --path {}/checkpoint_top5_average.pt --task translation --max-sentences 10 --beam 5 --gen-subset test --remove-bpe  --lenpen 1.0 > {}/ende.out'.format(model_dir, model_dir)
output = subprocess.check_output(command, shell=True, universal_newlines=True)
# separation
command = "cat {}/ende.out | grep -P '^T-' | cut -c3- | sort -n -k 1 |uniq | cut -f 2 > {}/ende.gold".format(model_dir, model_dir)
output = subprocess.check_output(command, shell=True, universal_newlines=True)
command = "cat {}/ende.out | grep -P '^H-' | cut -c3- | sort -n -k 1 |uniq | cut -f 3 > {}/ende.pred".format(model_dir, model_dir)
output = subprocess.check_output(command, shell=True, universal_newlines=True)
# dehyp
command = "python scripts/dehyphenate.py {}/ende.gold {}/ende.detok.gold".format(model_dir, model_dir)
output = subprocess.check_output(command, shell=True, universal_newlines=True)
command = "python scripts/dehyphenate.py {}/ende.pred {}/ende.detok.pred".format(model_dir, model_dir)
output = subprocess.check_output(command, shell=True, universal_newlines=True)
# scoring
command = 'python scripts/score_pybleu.py {}/ende.detok.gold {}/ende.detok.pred'.format(model_dir, model_dir)
output = subprocess.check_output(command, shell=True, universal_newlines=True)
print(output)

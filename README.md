# Deep Encoder, Shallow Decoder

### Download trained deep-shallow models
See our [paper](https://arxiv.org/abs/2006.10369) for more detail. `12-1` denotes a 12-layer encoder and 1-layer decoder.
Model | Data | Test BLEU
|---|---|---|
 [WMT16 EN-DE 12-1 w/ Distillation](https://drive.google.com/uc?id=1x_G2cjvM1nW5hjAB8-vWxRqtQTlmIaQU) | [WMT16/14 Distilled Data](https://drive.google.com/uc?id=1mNufoynJ9-Zy1kJh2TA_lHm2squji0i9) | 28.3
 [WMT16 EN-DE 6-1 w/ Distillation](https://drive.google.com/uc?id=1oA2aqZlVNj5FarxBlNXEHpBS4lRetTzU) | [WMT16/14 Distilled Data](https://drive.google.com/uc?id=1mNufoynJ9-Zy1kJh2TA_lHm2squji0i9) | 27.4
 [WMT EN-DE 12-1 w/o Distillation](https://drive.google.com/uc?id=1Wup2D318QYBFPW_NKI1mfP_hXOfmUI9r) | [WMT16/14 Raw Data](https://drive.google.com/uc?id=1iO7um-HWoNoRKDtw27YUSgyeubn9uXqj) | 26.9
Additional Models
|---|---|---|
 [WMT19 EN-DE Transformer Base](https://drive.google.com/uc?id=1yT7ZjqfvUYOBXvMjeY8uGRHQFWoSo8Q5) | [WMT19 EN-DE Transformer Big](https://drive.google.com/uc?id=15gAzHeRUCs-QV8vHeTReMPEh1j8excNE) | [WMT19 EN-DE moses tokenize + fastbpe](https://drive.google.com/uc?id=1j6z9fYdlUyOYsh7KJoumRlr1yHczxR5T) 


### Train
Here is the command to train a deep-shallow model with a 12-layer encoder and 1-layer decoder of base size. We tuned the dropout rate from `[0.1, 0.2, 0.3]` and chose 0.2 for WMT16 EN-DE translation. We used 16 GPUs. See our [paper](https://arxiv.org/abs/2006.10369) for more detail.

```bash
python train.py wmt16.en-de.deep-shallow.dist/data-bin/ --arch transformer --share-all-embeddings --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.98)' \
--max-tokens 4096 --dropout 0.2 --encoder-layers 12 --encoder-embed-dim 512 --decoder-layers 1 \
--decoder-embed-dim 512 --max-update 300000 \
--distributed-world-size 16 --distributed-port 54186 --fp16 --max-source-positions 10000 --max-target-positions 10000 \
--save-dir  checkpoint/trans_ende-dist_12-1_0.2/  --seed 1
```

### Evaluation

After downloading the tarballs from the table above:

```bash
tar -xvzf trans_ende-dist_12-1_0.2.tar.gz
tar -xvzf wmt16.en-de.deep-shallow.dist.tar.gz
python generate.py wmt16.en-de.deep-shallow.dist/data-bin/ --path trans_ende-dist_12-1_0.2/checkpoint_top5_average.pt \
--beam 5 --remove-bpe  --lenpen 1.0 --max-sentences 10
```

# Citation

Please cite as:

```bibtex
@article{Kasai2020DeepES,
  title={Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation},
  author={Jungo Kasai and Nikolaos Pappas and Hao Peng and J. Cross and Noah A. Smith},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.10369}
}
```

## Note
This code is based on the [fairseq](https://github.com/pytorch/fairseq/) library. Pretrained models should work with the most recent fairseq as well.

#!/bin/sh

#BSUB -q gpu
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=1]"

python -u ./train.py -d YAGO --batch_size 1024 --max_epochs 20

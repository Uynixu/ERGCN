#!/bin/sh

#BSUB -q gpu
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "span[ptile=1]"

python -u ./get_data_history.py -d YAGO

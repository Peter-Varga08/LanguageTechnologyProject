#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=3000


module restore LTP
echo "Modules Loaded"

python ./lstm_text_classification.py 

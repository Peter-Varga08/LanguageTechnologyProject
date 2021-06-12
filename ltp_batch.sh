#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=2000

module restore LTP
echo "Modules Loaded"

python ./lstm_text_classification.py 



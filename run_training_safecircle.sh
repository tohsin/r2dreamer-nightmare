#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"
conda activate nightmare-dreamer
python train.py env=sgym env.task=sgym_SafetyPointGoal1-v0 wandb=True

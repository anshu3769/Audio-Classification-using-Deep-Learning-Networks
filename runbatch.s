#!/bin/bash

#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=bdml_project_speech
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asd508@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1 -c1

module load anaconda3/5.3.1
source activate speech
python run.py --train_path <train_data_path> --valid_path <valid_data_path> --test_path <test_data_path>

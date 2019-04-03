# bdml
Creating Dataset command - 
python create_dataset.py \\<original-folder> --out_path <path to save the data for project>
  
  
We have run this project on NYU Prince server using Slurm batch script.

1. Create environment using requiremnets.yaml
conda env create -f requirements.yaml

2. Run the batch script 
sbatch runbatch.s

On local 
make sure all requirements are installed - (torch, librosa, numpy) and run command

python run.py --train_path <train_data_path> --valid_path <valid_data_path> --test_path <test_data_path>

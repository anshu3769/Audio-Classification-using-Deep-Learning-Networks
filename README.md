# bdml
1. mkdir bdml 
2. cd bdml 
3. git clone 
4. Download Speech data to this directory
5. gunzip speech_commands_v0.01.tar.gz
7. mkdir data 
8. cd data
9. tar xopf ..path_to/speech_commands_v0.01.tar 
10. cd ..
11. mkdir speechdata 
12. cd BDML
Creating Dataset command - 
python create_dataset.py ../data --out_path ../speechdata
  
  
We have run this project on NYU Prince server using Slurm batch script.

1. Create environment using requiremnets.yaml
conda env create -f requirements.yaml

2. Run the batch script 
sbatch runbatch.s

On local 
make sure all requirements are installed - (torch, librosa, numpy) and run command

python run.py --train_path <train_data_path> --valid_path <valid_data_path> --test_path <test_data_path>

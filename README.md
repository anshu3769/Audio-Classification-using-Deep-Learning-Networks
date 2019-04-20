# bdml
Steps to run on NYU Server 
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
13. module load anaconda3/5.3.1
14. conda env create -f requirements.yaml
15. source activate speech
16. python create_dataset.py ../data --out_path ../speechdata
17. python run.py --train_path ../speechdata/train/ --valid_path ../speechdata/valid --test_path ../speechdata/test


We have run this project on NYU Prince server using Slurm batch script.

1. To run using that the above steps have to be used once to create the environment 
2. After first use steps 13, 14, 15 and 17 can run using the batch script provided
3. Run the batch script 
sbatch runbatch.s

4. To change the configuration running changes can be made to batch script




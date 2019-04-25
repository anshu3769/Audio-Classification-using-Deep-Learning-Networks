# Speech to Text Classification (Keyword spotting)
The aim of the project is to analyse performance of various neural networks in identifying the word spoken by a person. The data used in this process is Google command dataset. It contains ~65k audio files each of which has a word spoken by a person  anf tag for that file which is the text for that audio file. There are 30 different words in the dataset spoken by different people. Thus, the task is to classify the audio files based on the word spoken. We ran the following neural networks to perform the task:
1. Lenet
2. VGG
3. ResNet
4. RNN (TO BE ADDED)

# Performance of the networks
TO BE ADDED

# Steps to train/test a model
Please follow the steps to train/test a model: 
## Setup the environment
  a. mkdir bdml 
  b. cd bdml 
  c. git clone 
  d. module load anaconda3/5.3.1
  e. conda env create -f requirements.yaml
  f. source activate bdml
  
## Load the dataset
  a. Download Speech data to this directory
  b. gunzip speech_commands_v0.01.tar.gz
  c. mkdir data 
  d. cd data
  e. tar xopf ..path_to/speech_commands_v0.01.tar 
  f. cd ..
  g. mkdir speechdata 
  h. cd BDML
  i. python create_dataset.py ../data --out_path ../speechdata

## Run the model
  a. python run.py --train_path ../speechdata/train/ --valid_path ../speechdata/valid --test_path ../speechdata/test
  Note: You can specify other arguments to the run.py script like batch_size, model e.t.c. You can find them all in the file itself.


We have run this project on NYU Prince server using Slurm batch script.
## To run the batch script on NYU server: 
a. sbatch runbatch.s
Note: You can change the arguments in the runbatch.s script to run with various network configuration.




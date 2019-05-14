# Audio Classification using Deep Learning Networks (Keyword spotting)
The aim of the project is to analyse performance of various neural networks in identifying the word spoken by a person. The data used in this process is Google command dataset. It contains ~65k audio files each of which has a word spoken by a person  anf tag for that file which is the text for that audio file. There are 30 different words in the dataset spoken by different people. Thus, the task is to classify the audio files based on the word spoken. We ran the following neural networks to perform the task:
1. Lenet
2. VGG
3. ResNet
4. CNNRNN
5. CNN-1D
6. Parallel Net ( combination of two netoworks trained in parallel)

# Performance of the networks
The following table shows the accuracies of the trained networks:


# Steps to train/test a model
Please follow the steps to train/test a model: 
## Setup the environment
  1. mkdir bdml 
  2. cd bdml 
  3. git clone 
  4. module load anaconda3/5.3.1
  4. conda env create -f requirements.yaml
  6. source activate bdml
  
## Load the dataset
  1. Download Speech data to this directory  <br />
     wget "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
  2. gunzip speech_commands_v0.01.tar.gz
  3. mkdir data; mv speech_commands_v0.01.tar data
  4. cd data
  5. tar xopf ..path_to/speech_commands_v0.01.tar 
  6. cd ..
  7. mkdir speechdata 
  8. cd BDML
  9. python create_dataset.py ../data --out_path ../speechdata

## Run the model
  1. python run.py --train_path ../speechdata/train/ --valid_path ../speechdata/valid --test_path ../speechdata/test --model CNN1D 
  
  Note: You can specify/change arguments to the run.py script like batch_size, model e.t.c. The information on other options is present in the run.py script.


We have run this project on NYU Prince server using Slurm batch script.
## To run the batch script on NYU server: 
  1. sbatch runbatch.s
  
  Note: You can change the arguments in the runbatch.s script to run with various network configuration.




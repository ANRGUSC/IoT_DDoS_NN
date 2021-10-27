# Neural Networks for DDoS Attack Detection using an Enhanced Urban IoT Dataset

This repository presents the source code for analyzing an enhanced urban IoT event-driven dataset. It helps with generating DDoS attacks on the datasets and training different neural network models for detecting DDoS attacks.


## Instructions for running the codes

The requirements.txt file contains the modules needed to run these scripts and can be installed by running the following command in the terminal:
* pip install -r requirements.txt

## Project config file

The project config file can be found in [/source_code](https://github.com/ANRGUSC/IoT_DDoS_NN/tree/main/source_code). The path to the output directory can be set in this file.

## Preprocess the dataset and generate training dataset

Before running any code, the original dataset need to be unzip in the [/dataset directory](https://github.com/ANRGUSC/IoT_DDoS_NN/tree/main/dataset). Two python scripts can be found in [/source_code/pre_process](https://github.com/ANRGUSC/IoT_DDoS_NN/tree/main/source_code/pre_process) folder for pre-processing the original dataset and also generating the training/testing dataset. 

### clean_dataset.py

This file genrates the bening_dataset which has N nodes and each node has time entries starting from the beginning to the end of original dataset with a step of time_step.

Input:
- Input Dataset
- Number of nodes
- Timestep
- Benign packet truncated Cauchy distribution parameters

Output:
- Benign dataset


### generate_attack_and_train_data.py

This script genrates the attacked dataset by considering the ratio of the nodes that are under attack, the attack duration, and also the attack start dates. Finally, it combines all different combinations of the attacked datasets to create train/test dataset.

Input:
- Bening dataset
- Benign packet truncated Cauchy distribution parameters
- Attack packet distribution parameter (k)
- Number of attack days
- Attack ratio
- Attack duration
- Attack start dates

Output:
- Attacked dataset
- Train/Test dataset



## Training neural network

Two python scripts can be found in /source_code/nn_training_"model" folders to train different neural network "model"s and generate results. "model" could be dense, cnn, lstm, and autoencoder.


### train_nn_"model".py

This script create a "model" neural network to train on the training dataset for detecting the attackers. The scrip save the final model and also the epochs logs and weights.

Input:
- Training/testing dataset
- Number of epochs

Output:
- Trained neural network model with epochs' logs and weights


### generate_results_"model".py

This script provides analysis like, accuracy, loss, confusion matrix, etc. based on the trained "model". Furthermore, it plots that true positive, false positive, and true attacks versus time.

Input:
- Training/testing dataset
- Trained neural network model

Output:
- General analysis on the training like accuracy, loss, confusion matrix, etc.
- Plots of true positive, false positive, and true attacks versus time for different attack ratios and durations


##  Compare neural network models

One python script can be found in [/source_code/nn_model_analysis](https://github.com/ANRGUSC/IoT_DDoS_NN/tree/main/source_code/nn_model_analysis) folder for comparing the performance of different neural network models.


### compare_models.py

This script plots the binary accuracy and recall values for all the neural network models versus the attack distribution parameter (k).

Input:
- Training/testing dataset
- Trained neural network models

Output:
- Graphs of the binary accuracy and recall values for all the neural network models versus the attack distribution parameter (k).


## Acknowledgement

   This material is based upon work supported in part by Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0160 for the Open, Programmable, Secure 5G (OPS-5G) program. Any views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. 




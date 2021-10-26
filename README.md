# Deidos Mirai

This repository presents the source code for analyzing an enhanced urban IoT event-driven dataset. It helps with generating DDoS attacks on the datasets and training different neural network models for detecting DDoS attacks.


## Instructions for running the codes

The requirements.txt file contains the modules needed to run these scripts and can be installed by running the following command in the terminal:
* pip install -r requirements.txt

## Project config file

The project config file can be found in [/source_code](https://github.com/ANRGUSC/deidos_mirai/tree/main/source_code). The path to the output directory can be set in this file.

## Preprocess the dataset and generate training dataset

Before running any code, the original dataset need to be unzip in the [/dataset directory](https://github.com/ANRGUSC/deidos_mirai/tree/main/dataset). One python scripts can be found in [/source_code/clean_dataset](https://github.com/ANRGUSC/deidos_mirai/tree/main/source_code/clean_dataset) folder for pre-processing the original dataset. 

### clean_dataset.py

This file genrates the bening_dataset which has N nodes and each node has time entries starting from the beginning to the end of original dataset with a step of time_step.

Input:
- Input Dataset
- Number of nodes
- Timestep

Output:
- Benign dataset

### generate_training_data.py

This script generates the training data by considering the different time windows for averaging the occupancies on the nodes.

Input:
- Attacked dataset
- Averaging time windows

Output:
- Training dataset


### generate_attack.py

This script genrates the attacked dataset by considering the ratio of the nodes that are under attack, the attack duration, and also the attack start dates.

Input:
- Bening dataset
- Number of attack days
- Attack ratio
- Attack duration
- Attack start dates

Output:
- Attacked dataset


## Training neural network

Two python scripts can be found in /source_code/nn_training_"model" folder to train a different neural network "model"s and generate results.


### train_nn_"model".py

This script create a "model" neural network to train on the training dataset for detecting the attackers. The scrip save the final model and also the epochs logs and weights.

Input:
- Training dataset
- Number of epochs

Output:
- Trained neural network model with epochs' logs and weights


### generate_results_"model".py

This script provides analysis like, accuracy, loss, confusion matrix, etc. based on the trained "model". Furthermore, it plots that true positive, false positive, and true attacks versus time.

Input:
- Training dataset
- Trained neural network model

Output:
- General analysis on the training like accuracy, loss, confusion matrix, etc.
- Plots of true positive, false positive, and true attacks versus time for different attack ratios and durations


## Acknowledgement

   This material is based upon work supported in part by Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0160 for the Open, Programmable, Secure 5G (OPS-5G) program. Any views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government. 




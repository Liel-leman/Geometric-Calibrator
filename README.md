# A Geometric Method for Improved Uncertainty Estimation in Real time

This repository provides an implementation of the paper "A Geometric Method for Real time Confidence Evaluation in Machine-Learning Models". 
All results presented in our work were produced with this code.

<p align="center">
  <img src="https://www.linkpicture.com/q/fitting_func.png" alt="latent_process" width="400"/>
</p>

## Datasets

All the datasets could be download from the provided links:\
-[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) \
-[MNIST](http://yann.lecun.com/exdb/mnist/) \
-[GTSRB](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) \
-[FashionMNIST](https://www.kaggle.com/zalando-research/fashionmnist) \
-[SignLanguageMNIST](https://www.kaggle.com/datamunge/sign-language-mnist) 

## Experiment
Note: in the code we call fast separation as stability. \
We represent dynamic string as {} for examples : 
{DatasetName} could be : "MNIST" or "CIFAR10" 

### Pre-procesing and configuration:
- Download datasets from the links above to a folder called /DatasetName/RealData 
- /{DatasetName}/{DatasetName}_divide-ALL.ipynb - divide the dataset to 10 different shuffles. 
- /{DatasetName}/{DatasetName}_paramTuning.ipynb - param tuning to exact dataset .
- config.py - configuration of the best params tuned.
- VARS.json - configuration of dataset batch size , epocs and #classes.  

### model training and computation of separation values: 
We use Slurm cluster system for this stage.
- /SLURM/computation.sh - bash script that opens computing node's for each dataset for each shuffle. \
the data is saved in a form of: \
├── {dataset}  \
│>      └── {shuffle_num} \
│>      │>      ├── model  \
│>      │>      │>       └── model_{dataset}_{model} - the main model as 'sav' format \
│>      │>      │>       ├── model.... \
│>      │>      │>       ├── m....  \
│>      │>      │>       └── ... \
│>      │>      ├── {model} \
│>      │>      │>   └── {model} \
│>      │>      │>       │>   └── y_pred_{val|test|train}.npy - predicted values on val|test|train. \
│>      │>      │>       │>   ├── {fast_separation|separation}_{val|test|train}.npy - computed metric \
│>      │>      │>       │>   ├── all_predictions_{val|test|train}.npy - the predict_proba on specific part of the dataset. \

### evaluation
- results.ipynb - main result on the computed data.



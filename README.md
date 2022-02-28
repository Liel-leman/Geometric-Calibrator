# A Geometric Method for Improved Uncertainty Estimation in Real time

This repository provides an implementation of the paper "A Geometric Method for Real time Confidence Evaluation in Machine-Learning Models". 
All results presented in our work were produced with this code.

<p align="center">
  <img src="https://www.linkpicture.com/q/fitting_func.png" alt="latent_process" width="400"/>
</p>

## Setup
Inside a python (>=3.6) virtual enviroment run:

    pip install -e .
    pip install -r requirements.txt


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
- Download datasets from the links above to a folder called /{DatasetName}/{RealData} 
- /{DatasetName}/{DatasetName}_divide-ALL.ipynb - divide the dataset to 10 different shuffles. 
- /{DatasetName}/{DatasetName}_paramTuning.ipynb - param tuning to exact dataset .
- config.py - configuration of the best params tuned.
- VARS.json - configuration of dataset batch size , epocs and #classes.  

### Model training and computation of separation values: 
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

### Evaluation
- results.ipynb - main result on the computed data.



|     Dataset    	| Model 	| Fast Separation 	|   Separation  	|         SKlearn        	|           SBC          	|           HB           	|
|:--------------:	|:-----:	|:---------------:	|:-------------:	|:----------------------:	|:----------------------:	|:----------------------:	|
|       MNIST    	| CNN   	| 0.0029±0.0008   	| 0.0031±0.0007 	| 0.0032±0.0004 (9.38%)  	| 0.0190±0.0018 (84.74%) 	| 0.0046±0.0005 (36.96%) 	|
|      MNIST     	| RF    	| 0.0044±0.0005   	| 0.0045±0.0006 	| 0.0098±0.0012 (55.1%)  	| 0.0331±0.0017 (86.71%) 	| 0.0080±0.0009 (45%)    	|
|      MNIST     	| GB    	| 0.0040±0.0007   	| 0.0040±0.0008 	| 0.0176±0.0014 (77.27%) 	| 0.0384±0.0025 (89.58%) 	| 0.0074±0.0012 (45.95%) 	|
|     GTSRB_RGB  	| CNN   	| 0.0014±0.0005   	| 0.0015±0.0003 	| 0.0018±0.0004 (22.22%) 	| 0.1955±0.0072 (99.28%) 	| 0.0021±0.0005 (33.33%) 	|
|    GTSRB_RGB   	| RF    	| 0.0042±0.0005   	| 0.0042±0.0007 	| 0.0262±0.0014 (83.97%) 	| 0.0788±0.0047 (94.67%) 	| 0.0065±0.0013 (35.38%) 	|
|    GTSRB_RGB   	| GB    	| 0.0071±0.0010   	| 0.0075±0.0013 	| 0.1002±0.0029 (92.91%) 	| 0.1733±0.0045 (95.9%)  	| 0.0175±0.0013 (59.43%) 	|
|   SignLanguage 	| CNN   	| 0.0001±0.0001   	| 0.0001±0.0001 	| 0.0004±0.0002 (75%)    	| 0.1287±0.0151 (99.92%) 	| 0.0008±0.0004 (87.5%)  	|
|   SignLanguage 	| RF    	| 0.0008±0.0002   	| 0.0009±0.0003 	| 0.0049±0.0006 (83.67%) 	| 0.0728±0.0079 (98.9%)  	| 0.0026±0.0008 (69.23%) 	|
|   SignLanguage 	| GB    	| 0.0008±0.0003   	| 0.0008±0.0002 	| 0.0406±0.0018 (98.03%) 	| 0.2564±0.0207 (99.69%) 	| 0.0065±0.0009 (87.69%) 	|
|      Fashion   	| CNN   	| 0.0090±0.0007   	| 0.0087±0.0010 	| 0.0091±0.0013 (1.1%)   	| 0.0266±0.0021 (66.17%) 	| 0.0141±0.0018 (36.17%) 	|
|      Fashion   	| RF    	| 0.0091±0.0014   	| 0.0101±0.0016 	| 0.0111±0.0011 (18.02%) 	| 0.0506±0.0022 (82.02%) 	| 0.0170±0.0019 (46.47%) 	|
|      Fashion   	| GB    	| 0.0092±0.0014   	| 0.0093±0.0006 	| 0.0384±0.0019 (76.04%) 	| 0.0667±0.0027 (86.21%) 	| 0.0160±0.0021 (42.5%)  	|
|     CIFAR_RGB  	| CNN   	| 0.0110±0.0026   	| 0.0102±0.0020 	| 0.0155±0.0028 (29.03%) 	| 0.0559±0.0017 (80.32%) 	| 0.0188±0.0021 (41.49%) 	|
|     CIFAR_RGB  	| RF    	| 0.0142±0.0020   	| 0.0153±0.0017 	| 0.0342±0.0025 (58.48%) 	| 0.1437±0.0020 (90.12%) 	| 0.0370±0.0029 (61.62%) 	|
|     CIFAR_RGB  	| GB    	| 0.0155±0.0023   	| 0.0153±0.0016 	| 0.0769±0.0020 (79.84%) 	| 0.2036±0.0028 (92.39%) 	| 0.0404±0.0015 (61.63%) 	|


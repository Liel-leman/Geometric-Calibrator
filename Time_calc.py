import numpy as np
import warnings
from utils import *
from calibrators import * 
import pickle
warnings.filterwarnings("ignore")
import torch
import sys
sys.path.append('../')
import time
import copy
from SLURM.sklearn_config import *
import time
import tensorflow as tf
from skimage import color
from skimage import io


#avg
all_time=[]
Time_dict={}
for pool_size in range(1,5):
    data_dict={}
    for dataset_name in ['MNIST','Fashion','SignLanguage','CIFAR_RGB','GTSRB_RGB']:       
        for model_name in ['RF','GB','pytorch']:
            shuffle_list=[]
            for shuffle in range(1):
                model_info= load_model(dataset_name, model_name, shuffle)
                num_labels=model_info.data.num_labels
                RGB='RGB' in dataset_name 
                
                stability,time_all,ex_in_time= stability_calc_pool(model_info.data.X_train,model_info.data.X_val,model_info.data.y_train,model_info.y_pred_val,num_labels,pool_type='Avg',pool_size=pool_size,RGB=RGB) 
                stability_test,_,_= stability_calc_pool(model_info.data.X_train,model_info.data.X_test,model_info.data.y_train,model_info.y_pred_test,num_labels,pool_type='Avg',pool_size=pool_size,RGB=RGB) 

                all_time.append((dataset_name,model_name,shuffle,time_all,ex_in_time))
                shuffle_list.append(ex_in_time)
            data_dict[f'{dataset_name}-{model_name}']= mean_confidence_interval2(shuffle_list)
    Time_dict[pool_size]=data_dict
    pd.DataFrame(Time_dict).to_csv("Full_time_df.csv")
    print(Time_dict)
print(all_time)
print(Time_dict)

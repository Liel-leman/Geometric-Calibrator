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



for dataset_name in ['MNIST','Fashion','SignLanguage','CIFAR_RGB','GTSRB_RGB']:
    for model_name in ['RF','GB','pytorch']:
        for shuffle in range(10):
            for pool_size in range(3,5):
                model_info= load_model(dataset_name, model_name, shuffle)
                num_labels=model_info.data.num_labels


                if 'RGB' in dataset_name:
                    stability,time_all,ex_in_time= stability_calc_pool(model_info.data.X_train,model_info.data.X_val,model_info.data.y_train,model_info.y_pred_val,num_labels,pool_type='Max',pool_size=pool_size,RGB=True) 
                    stability_test,_,_= stability_calc_pool(model_info.data.X_train,model_info.data.X_test,model_info.data.y_train,model_info.y_pred_test,num_labels,pool_type='Max',pool_size=pool_size,RGB=True) 

                else:

                    stability,time_all,ex_in_time= stability_calc_pool(model_info.data.X_train,model_info.data.X_val,model_info.data.y_train,model_info.y_pred_val,num_labels,pool_type='Max',pool_size=pool_size) 
                    stability_test,_,_= stability_calc_pool(model_info.data.X_train,model_info.data.X_test,model_info.data.y_train,model_info.y_pred_test,num_labels,pool_type='Max',pool_size=pool_size)
                np.save(f'./stab/{dataset_name}/stab_{pool_size}maxpool_{dataset_name}_{model_name}_{shuffle}.npy',stability)
                np.save(f'./stab/{dataset_name}/test_stab_{pool_size}maxpool_{dataset_name}_{model_name}_{shuffle}.npy',stability_test)

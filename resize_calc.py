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
from ModelInfo import *
import tensorflow as tf

#resize(2-4)
method_name='Resize'
all_time=[]
for red_param in [2,3,4]:
    for dataset_name in ['Fashion','MNIST','SignLanguage']:
        for model_name in ['RF','GB','pytorch']:
            for shuffle in range(10):
                if model_name=='pytorch':
                    model_info=load_model_pytorch(dataset_name, model_name, shuffle)
                else:
                    model_info= load_model(dataset_name, model_name, shuffle)
                data=model_info.data
                pixels=int(sqrt(data.X_train.shape[1]))
                size=pixels//red_param

                length= len(model_info.data.X_train)
                X_train=torch.tensor(model_info.data.X_train.reshape(length,pixels,pixels))
                X_train = X_train[ ..., tf.newaxis]
                model_info.data.X_train=tf.image.resize(X_train, [size,size]).numpy().reshape(length,-1)

                length= len(model_info.data.X_test)
                X_test=torch.tensor(model_info.data.X_test.reshape(length,pixels,pixels))
                X_test = X_test[ ..., tf.newaxis]
                model_info.data.X_test=tf.image.resize(X_test, [size,size]).numpy().reshape(length,-1)

                length= len(model_info.data.X_val)
                X_val=torch.tensor(model_info.data.X_val.reshape(length,pixels,pixels))
                X_val = X_val[ ..., tf.newaxis]
                model_info.data.X_val=tf.image.resize(X_val, [size,size]).numpy().reshape(length,-1)
            
                num_label=model_info.data.num_labels
                stability,time_all,ex_in_time=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_val),model_info.data.y_train,model_info.y_pred_val,num_label)
                stability_test,_,_=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_test),model_info.data.y_train,model_info.y_pred_test,num_label)

                all_time.append((model_name,time_all,ex_in_time))
                np.save(f'./stab/{dataset_name}/stab_{method_name}_{red_param}_{dataset_name}_{model_name}_{shuffle}.npy',stability)
                np.save(f'./stab/{dataset_name}/test_stab_{method_name}_{red_param}_{dataset_name}_{model_name}_{shuffle}.npy',stability_test)

all_time
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


#avg(2)
all_time=[]
for dataset_name in ['CIFAR_RGB','GTSRB_RGB','MNIST','Fashion','SignLanguage']:
    for model_name in ['RF','GB','pytorch']:
        shuffle=0
        polling = torch.nn.AvgPool2d(2)
        model_info= load_model(dataset_name, model_name, shuffle)
        data=model_info.data
        
        if 'RGB' in dataset_name:
            if 'CIFAR' in dataset_name:
                pixels=32
            elif 'GTSRB' in dataset_name:
                pixels=30
            train=model_info.data.X_train.reshape(len(data.X_train),pixels,pixels,3)
            imgGray_train = color.rgb2gray(train)
            pool_train=polling(torch.tensor(imgGray_train)).reshape(len(data.X_train),-1)

            test=model_info.data.X_test.reshape(len(data.X_test),pixels,pixels,3)
            imgGray_test = color.rgb2gray(test)
            pool_test=polling(torch.tensor(imgGray_test)).reshape(len(data.X_test),-1)

            val=model_info.data.X_val.reshape(len(data.X_val),pixels,pixels,3)
            imgGray_val = color.rgb2gray(val)
            pool_val=polling(torch.tensor(imgGray_val)).reshape(len(data.X_val),-1)


            num_label=model_info.data.num_labels
            stability,time_all,ex_in_time=new_stability_calc(np.array(pool_train),np.array(pool_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(pool_train),np.array(pool_test),model_info.data.y_train,model_info.y_pred_test,num_label)
        
            all_time.append((model_name,time_all,ex_in_time))
        
        else:
            pixels=int(sqrt(data.X_train.shape[1]))
            model_info.data.X_train=polling(torch.tensor(model_info.data.X_train.reshape((len(data.X_train),pixels,pixels)))).reshape(len(data.X_train),-1)
            model_info.data.X_test=polling(torch.tensor(model_info.data.X_test.reshape((len(data.X_test),pixels,pixels)))).reshape(len(data.X_test),-1)
            model_info.data.X_val=polling(torch.tensor(model_info.data.X_val.reshape((len(data.X_val),pixels,pixels)))).reshape(len(data.X_val),-1)
            num_label=model_info.data.num_labels
            stability,time_all,ex_in_time=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_test),model_info.data.y_train,model_info.y_pred_test,num_label)

            all_time.append((model_name,time_all,ex_in_time))
print("avg2",all_time)
np.save('all_time_2avg.npy',np.array(all_time))

#avg(3)
all_time=[]
for dataset_name in ['CIFAR_RGB','GTSRB_RGB','MNIST','Fashion','SignLanguage']:
    for model_name in ['RF','GB','pytorch']:
        shuffle=0
        polling = torch.nn.AvgPool2d(3)
        model_info= load_model(dataset_name, model_name, shuffle)
        data=model_info.data
        
        if 'RGB' in dataset_name:
            if 'CIFAR' in dataset_name:
                pixels=32
            elif 'GTSRB' in dataset_name:
                pixels=30
            train=model_info.data.X_train.reshape(len(data.X_train),pixels,pixels,3)
            imgGray_train = color.rgb2gray(train)
            pool_train=polling(torch.tensor(imgGray_train)).reshape(len(data.X_train),-1)

            test=model_info.data.X_test.reshape(len(data.X_test),pixels,pixels,3)
            imgGray_test = color.rgb2gray(test)
            pool_test=polling(torch.tensor(imgGray_test)).reshape(len(data.X_test),-1)

            val=model_info.data.X_val.reshape(len(data.X_val),pixels,pixels,3)
            imgGray_val = color.rgb2gray(val)
            pool_val=polling(torch.tensor(imgGray_val)).reshape(len(data.X_val),-1)


            num_label=model_info.data.num_labels
            stability,time_all,ex_in_time=new_stability_calc(np.array(pool_train),np.array(pool_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(pool_train),np.array(pool_test),model_info.data.y_train,model_info.y_pred_test,num_label)
        
            all_time.append((model_name,time_all,ex_in_time))
        
        else:
            pixels=int(sqrt(data.X_train.shape[1]))
            model_info.data.X_train=polling(torch.tensor(model_info.data.X_train.reshape((len(data.X_train),pixels,pixels)))).reshape(len(data.X_train),-1)
            model_info.data.X_test=polling(torch.tensor(model_info.data.X_test.reshape((len(data.X_test),pixels,pixels)))).reshape(len(data.X_test),-1)
            model_info.data.X_val=polling(torch.tensor(model_info.data.X_val.reshape((len(data.X_val),pixels,pixels)))).reshape(len(data.X_val),-1)
            num_label=model_info.data.num_labels
            stability,time_all,ex_in_time=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_test),model_info.data.y_train,model_info.y_pred_test,num_label)

            all_time.append((model_name,time_all,ex_in_time))
print("avg3",all_time)
np.save('all_time_3avg.npy',np.array(all_time))


#avg(4)
all_time=[]
for dataset_name in ['CIFAR_RGB','GTSRB_RGB','MNIST','Fashion','SignLanguage']:
    for model_name in ['RF','GB','pytorch']:
        shuffle=0
        polling = torch.nn.AvgPool2d(4)
        model_info= load_model(dataset_name, model_name, shuffle)
        data=model_info.data
        
        if 'RGB' in dataset_name:
            if 'CIFAR' in dataset_name:
                pixels=32
            elif 'GTSRB' in dataset_name:
                pixels=30
            train=model_info.data.X_train.reshape(len(data.X_train),pixels,pixels,3)
            imgGray_train = color.rgb2gray(train)
            pool_train=polling(torch.tensor(imgGray_train)).reshape(len(data.X_train),-1)

            test=model_info.data.X_test.reshape(len(data.X_test),pixels,pixels,3)
            imgGray_test = color.rgb2gray(test)
            pool_test=polling(torch.tensor(imgGray_test)).reshape(len(data.X_test),-1)

            val=model_info.data.X_val.reshape(len(data.X_val),pixels,pixels,3)
            imgGray_val = color.rgb2gray(val)
            pool_val=polling(torch.tensor(imgGray_val)).reshape(len(data.X_val),-1)


            num_label=model_info.data.num_labels
            stability,time_all,ex_in_time=new_stability_calc(np.array(pool_train),np.array(pool_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(pool_train),np.array(pool_test),model_info.data.y_train,model_info.y_pred_test,num_label)
        
            all_time.append((model_name,time_all,ex_in_time))
        
        else:
            pixels=int(sqrt(data.X_train.shape[1]))
            model_info.data.X_train=polling(torch.tensor(model_info.data.X_train.reshape((len(data.X_train),pixels,pixels)))).reshape(len(data.X_train),-1)
            model_info.data.X_test=polling(torch.tensor(model_info.data.X_test.reshape((len(data.X_test),pixels,pixels)))).reshape(len(data.X_test),-1)
            model_info.data.X_val=polling(torch.tensor(model_info.data.X_val.reshape((len(data.X_val),pixels,pixels)))).reshape(len(data.X_val),-1)
            num_label=model_info.data.num_labels
            stability,time_all,ex_in_time=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_test),model_info.data.y_train,model_info.y_pred_test,num_label)

            all_time.append((model_name,time_all,ex_in_time))
print("avg4",all_time)
np.save('all_time_4avg.npy',np.array(all_time))

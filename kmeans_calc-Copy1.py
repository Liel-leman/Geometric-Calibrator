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

from sklearn.cluster import KMeans

def create_reduced_kmeans(X_train,y_train,num_label,red_param):
    new_points=[]
    new_y=[]
    start_time = time.time()
    for i in range(num_label):
        idx_same=np.where(y_train==i)    
        size=len(idx_same[0])//(red_param**2)
        kmeans = KMeans(n_clusters=size)
        kmeans.fit(X_train[idx_same])
        new_points.append(kmeans.cluster_centers_)
        new_y.append([i]*size)    
    new_X=np.array([img for class_grp in new_points for img in class_grp])
    new_y=np.array([i for listt in new_y for i in listt])
    tot_time=time.time()-start_time
    return new_X, new_y,tot_time






#K-means(2-4)
method_name='kmeans'
all_time=[]
Time_dict={}
for red_param in [1,2,3,4]:
    data_dict={}
    for dataset_name in ['Fashion','MNIST','SignLanguage']:
        for model_name in ['RF','GB','pytorch']:
            shuffle_list=[]
            for shuffle in range(10):
                if model_name=='pytorch':
                    model_info=load_model_pytorch(dataset_name, model_name, shuffle)
                else:
                    model_info= load_model(dataset_name, model_name, shuffle)
                
                
                num_label=model_info.data.num_labels
                
                
                model_info.data.X_train,model_info.data.y_train,tot_time = create_reduced_kmeans(model_info.data.X_train,model_info.data.y_train,num_label,red_param)
                np.save(f'./stab/{dataset_name}/kmeans_data_X_{method_name}_{red_param}_{dataset_name}_{model_name}_{shuffle}.npy',model_info.data.X_train)
                np.save(f'./stab/{dataset_name}/kmeans_data_y_{method_name}_{red_param}_{dataset_name}_{model_name}_{shuffle}.npy',model_info.data.y_train)


                
                
                
                
                
                stability,time_all,ex_in_time=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_val),model_info.data.y_train,model_info.y_pred_val,num_label)
            stability_test,_,_=new_stability_calc(np.array(model_info.data.X_train),np.array(model_info.data.X_test),model_info.data.y_train,model_info.y_pred_test,num_label)

                all_time.append((dataset_name, model_name,time_all,ex_in_time))
                np.save(f'./stab/{dataset_name}/stab_{method_name}_{red_param}_{dataset_name}_{model_name}_{shuffle}.npy',stability)
                np.save(f'./stab/{dataset_name}/test_stab_{method_name}_{red_param}_{dataset_name}_{model_name}_{shuffle}.npy',stability_test)
                shuffle_list.append(ex_in_time)
            data_dict[f'{dataset_name}-{model_name}']= mean_confidence_interval2(shuffle_list)
    Time_dict[red_param]=data_dict
    pd.DataFrame(Time_dict).to_csv("Kmeans_time_df_RGB.csv")
print(all_time)
print(Time_dict)


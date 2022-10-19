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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def apply_reduction(trainX,train_y,reduced_type,red_param,train=True,model=None):
    
    pixels=int(sqrt(trainX.shape[1]))
    pca=None
    if reduced_type == 'Avgpool':
        polling = torch.nn.AvgPool2d(red_param)
        trainX=polling(torch.tensor(trainX.reshape((len(trainX),pixels,pixels)))).reshape(len(trainX),-1)
        
    elif reduced_type == 'Maxpool':
        polling = torch.nn.MaxPool2d(red_param) 
        trainX=polling(torch.tensor(trainX.reshape((len(trainX),pixels,pixels)))).reshape(len(trainX),-1)
        
    elif reduced_type == 'resize':
        size=pixels//red_param
        length= len(trainX)
        X_train=torch.tensor(trainX.reshape(length,pixels,pixels))
        X_train = X_train[ ..., tf.newaxis]
        trainX=tf.image.resize(X_train, [size,size]).numpy().reshape(length,-1)
                
    elif reduced_type == 'PCA':
        if train == True:
            size=pixels//red_param
            pca = PCA(n_components=size**2)  
            trainX=pca.fit_transform(trainX)
        if train == False:
            size=pixels//red_param
            trainX=model.transform(trainX.reshape(1,-1))
    elif reduced_type == 'randpix':
        size=(pixels//red_param)**2
        string=np.random.randint(0,255, size=size)        
        trainX=trainX[:,string]
        
    elif reduced_type == 'randset':
        if train == True:
            length= len(trainX)
            size=(length//red_param)
            string=np.random.randint(0,length, size=size)
            trainX=trainX[string,:]
            train_y=train_y[string]
        
    #elif reduced_type == 'kmeans':
      
        
    else:
        print("error reduce type")
        
    return trainX,train_y,pca

def stability_calc_reduced(trainX,testX,train_y,test_y_pred,num_labels,reduced_type='Avgpool',red_param=1,RGB=False):
    '''
    Calculates the stability of the test set.
            Parameters:
                    trainX (List)
                    testX (List) 
                    train_y (List)
                    test_y_pred (list)
                    num_labels (Int)
            Returns:
                    stability(List)
    '''  
    
        
    # if RGB==False:
    trainX,train_y,model = apply_reduction(trainX,train_y,reduced_type,red_param)

    same_nbrs=[]
    other_nbrs=[]
    for i in range(num_labels):
        idx_other=np.where(train_y!=i)
        other_nbrs.append(NearestNeighbors(n_neighbors=1).fit(trainX[idx_other]))
        idx_same=np.where(train_y==i)
        same_nbrs.append(NearestNeighbors(n_neighbors=1).fit(trainX[idx_same]))

    stability=np.array([-1.]*testX.shape[0])
    start = time.time()
    for i in range(testX.shape[0]):
        x=testX[i]
        y=test_y_pred[i]
        x,y,_ = apply_reduction(x.reshape(1, -1),y,reduced_type,red_param,train=False,model= model)
        pred_label=y

        dist1,idx1= same_nbrs[pred_label].kneighbors(x)
        dist2,idx2= other_nbrs[pred_label].kneighbors(x)

        stability[i]=(dist2-dist1)/2
    end = time.time()
    time_all=end-start
    ex_in_time=testX.shape[0]/time_all
    return stability,time_all,ex_in_time  

#     else:
#         pixels=int(sqrt(trainX.shape[1]/3))
#         train=trainX.reshape(len(trainX),pixels,pixels,3)
#         imgGray_train = color.rgb2gray(train)
#         trainX=polling(torch.tensor(imgGray_train)).reshape(len(trainX),-1)

#         same_nbrs=[]
#         other_nbrs=[]
#         for i in range(num_labels):
#             idx_other=np.where(train_y!=i)
#             other_nbrs.append(NearestNeighbors(n_neighbors=1).fit(trainX[idx_other]))
#             idx_same=np.where(train_y==i)
#             same_nbrs.append(NearestNeighbors(n_neighbors=1).fit(trainX[idx_same]))
             
#         stability=np.array([-1.]*testX.shape[0])
#         start = time.time()
#         for i in range(testX.shape[0]):
#             test=color.rgb2gray(testX[i].reshape(1,pixels,pixels,3) )
#             x=polling(torch.tensor(test)).reshape(1,-1)
#             pred_label=test_y_pred[i]
            
#             dist1,idx1= same_nbrs[pred_label].kneighbors(x)
#             dist2,idx2= other_nbrs[pred_label].kneighbors(x)

#             stability[i]=(dist2-dist1)/2
#         end = time.time()
#         time_all=end-start
#         ex_in_time=testX.shape[0]/time_all
#         return stability,time_all,ex_in_time
    
    
    
    
    

#avg
all_time=[]
Time_dict={}
for reduced_method in ['PCA','randpix','randset','resize','Avgpool','Maxpool']:#,'kmeans']:
    print(reduced_method)
    for red_param in range(1,5):
        data_dict={}
        for dataset_name in ['SignLanguage','Fashion','MNIST']:#,'CIFAR_RGB','GTSRB_RGB']:       
            for model_name in ['RF','GB','pytorch']:
                shuffle_list=[]
                for shuffle in range(1):
                    model_info= load_model(dataset_name, model_name, shuffle)
                    num_labels=model_info.data.num_labels
                    RGB='RGB' in dataset_name 

                    stability,time_all,ex_in_time= stability_calc_reduced(model_info.data.X_train,model_info.data.X_val,model_info.data.y_train,model_info.y_pred_val,num_labels,reduced_type=reduced_method,red_param=red_param,RGB=RGB) 
                    #stability_test,_,_= stability_calc_reduced(model_info.data.X_train,model_info.data.X_test,model_info.data.y_train,model_info.y_pred_test,num_labels,pool_type='Avg',pool_size=pool_size,RGB=RGB) 

                    all_time.append((dataset_name,model_name,shuffle,time_all,ex_in_time))
                    shuffle_list.append(ex_in_time)
                data_dict[f'{dataset_name}-{model_name}']= mean_confidence_interval2(shuffle_list)
        Time_dict[f'{reduced_method}-{red_param}']=data_dict
        
pd.DataFrame(Time_dict).to_csv("Full_time_df.csv")


print(all_time)
print(Time_dict)






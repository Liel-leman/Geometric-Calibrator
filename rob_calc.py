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
from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.estimators.classification import  SklearnClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def calc_rob(X,Y,classifier,eps_init=0.01):
    rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier)
    rob=[]
    for i in tqdm(range(len(X))):
        average_bound, verified_error = rt.verify(x=X[i:i+1], y=Y[i:i+1], eps_init=0.01, nb_search_steps=100, max_clique=10, 
                                          max_level=10)
        rob.append(average_bound)
    return rob
        
    
def all_rob_calc(dataset_name,model_name,shuffle):
    #rob
    model_info= load_model(dataset_name, model_name, shuffle)
    #data processing
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_y_test = enc.fit_transform(model_info.data.y_test.reshape(-1,1)).toarray()
    enc_y_val = enc.fit_transform(model_info.data.y_val.reshape(-1,1)).toarray()
    enc_y_train = enc.fit_transform(model_info.data.y_train.reshape(-1,1)).toarray()
    norm_X_test=model_info.data.X_test/255
    norm_X_val=model_info.data.X_val/255
    norm_X_train=model_info.data.X_train/255
    #model
    model_dir =  f'{dataset_name}/{shuffle}/model/model_{dataset_name}_{model_name}.sav'
    model = pickle.load(open(model_dir, 'rb'))
    classifier = SklearnClassifier(model=model)
    #roc calc
    rob_val=calc_rob(norm_X_val[:10],enc_y_val[:10],classifier)
    rob_test=calc_rob(norm_X_test[:10],enc_y_test[:10],classifier)
    np.save(f'./rob/{dataset_name}/rob_{dataset_name}_{model_name}_{shuffle}_val.npy',rob_val)
    np.save(f'./rob/{dataset_name}/rob_{dataset_name}_{model_name}_{shuffle}_test.npy',rob_test)



if __name__ == "__main__":
    all_rob_calc(sys.argv[1],sys.argv[2],sys.argv[3])




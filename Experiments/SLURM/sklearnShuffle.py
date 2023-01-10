import sys
import numpy
import tensorflow as tf
from math import sqrt
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import math 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten,Input,MaxPooling2D,Activation,BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from math import sqrt
from utils import ECE_calc, stability_calc, hot_padding, sep_calc_parallel, normalize_dataset
import numpy as np


import os
from sklearn_config import *
import json


class Data:
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, num_labels, isRGB=False):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.num_labels = num_labels
        self.isRGB = isRGB



def load_data(dataset_name,shuffle_num):
    '''
    loads the data of specific shuffle
    '''
    try:
        VARS = json.load(open('../VARS.json'))
    except:
        VARS = json.load(open('./VARS.json'))

    NUM_LABELS = VARS['NUM_LABELS']
    
    data_dir = f'./{dataset_name}/{shuffle_num}/data/'

    #load data
    X_train       = np.load(data_dir+'X_train.npy',mmap_mode='r')
    X_test        = np.load(data_dir+'X_test.npy',mmap_mode='r')
    X_val         = np.load(data_dir+'X_val.npy',mmap_mode='r')
    y_train       = np.load(data_dir+'y_train.npy',mmap_mode='r')
    y_test        = np.load(data_dir+'y_test.npy',mmap_mode='r')
    y_val         = np.load(data_dir+'y_val.npy',mmap_mode='r')


    isRGB = "RGB" in dataset_name
    data = Data(X_train, X_test, X_val, y_train, y_test, y_val, NUM_LABELS[dataset_name],isRGB)
    
    return data
    
  
  
def save_params(y_pred_val,y_pred_test,y_pred_train,all_predictions_val,all_predictions_test,all_predictions_train,calc_dir):
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    np.save(calc_dir + f'y_pred_val.npy', y_pred_val)
    np.save(calc_dir + f'y_pred_test.npy', y_pred_test)
    np.save(calc_dir + f'y_pred_train.npy', y_pred_train)
    np.save(calc_dir + f'all_predictions_val.npy', all_predictions_val)
    np.save(calc_dir + f'all_predictions_test.npy', all_predictions_test)
    np.save(calc_dir + f'all_predictions_train.npy', all_predictions_train)
    


def calc_predictions(data, dataset_name, model_name, shuffle_num):
    '''
    main purpuse is to calculate y_pred for test and val.
    if no model(cali or not) exists :
        its create it.
    '''

    try:
        VARS = json.load(open('../VARS.json'))
    except:
        VARS = json.load(open('./VARS.json'))

    NO_EPOCHS = VARS['NO_EPOCHS']
    BATCH_SIZE = VARS['BATCH_SIZE']
    
    
    isFirstTime = False # indicator if we run the function on the first time
    
    calc_dir = f'./{dataset_name}/{shuffle_num}/{model_name}/'
    model_dir = f'./{dataset_name}/{shuffle_num}/model/'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)        

    existModel = os.path.exists(model_dir+f'model_{dataset_name}_{model_name}.sav')  
    #existModel = False # remove after 
    if not existModel:
        print(f'Fitting - {model_name}')
        isFirstTime = True
        model = eval(f'create_{model_name}_{dataset_name}()')
        model = model.fit(data.X_train, data.y_train)
        pickle.dump(model, open(model_dir+f'model_{dataset_name}_{model_name}.sav', 'wb')) 

    if isFirstTime:
        print('saving model predictions')
        y_pred_val            = model.predict(data.X_val)
        y_pred_test           = model.predict(data.X_test)
        y_pred_train          = model.predict(data.X_train)
        all_predictions_val   = model.predict_proba(data.X_val)
        all_predictions_test  = model.predict_proba(data.X_test)
        all_predictions_train = model.predict_proba(data.X_train)

        save_params(y_pred_val,y_pred_test,y_pred_train,all_predictions_val,all_predictions_test,all_predictions_train,calc_dir)
        
    else:
        print("model exists : preloading calculations")
        y_pred_val = np.load(calc_dir + f'y_pred_val.npy')
        y_pred_test = np.load(calc_dir + f'y_pred_test.npy')

    return y_pred_val, y_pred_test



def run_shuffle_on_data_model(dataset_name, model_name, shuffle_num, metric='both' ,norm='L1'):
    '''
    compute stab\sep on one shuffle and saves the results
    compute models and save them
    all thet for specific shuffle

            Parameters:
                dataset_name(String)
                model_name(String)
                shuffle_range(Range for example range(0,3))
                metric = 'stab'| 'sep' | 'both' | 'nothing'(when we wanna only models)

            Returns:
                None

    '''
    data_dir = f'./{dataset_name}/{shuffle_num}/data/'
    calc_dir = f'./{dataset_name}/{shuffle_num}/{model_name}/'

    
    data = load_data(dataset_name,shuffle_num)

    # calc predictions based on model
    y_pred_val, y_pred_test = calc_predictions(data, dataset_name, model_name, shuffle_num)

    # save separation file
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
        
    L_to_string = {'L1':'manhattan', 'Linf':'chebyshev', 'L2':'euclidean'}
    PATH_data = f'./{dataset_name}/{shuffle_num}/data/'
    
    
    if metric == 'stab' or metric == 'both':
        print('calculating stab')
        stability_val = stability_calc(data.X_train, data.X_val, data.y_train, y_pred_val, data.num_labels, L_to_string[norm])
        stability_test = stability_calc(data.X_train, data.X_test, data.y_train, y_pred_test, data.num_labels, L_to_string[norm])
        np.save(calc_dir + f'stability_val_{norm}.npy', stability_val)
        np.save(calc_dir + f'stability_test_{norm}.npy', stability_test)
        print('finish stab')

    if metric == 'sep' or metric == 'both':
        print('calculating Sep')
        sep_val = sep_calc_parallel(data.X_val, y_pred_val, PATH_data, norm)
        sep_test = sep_calc_parallel(data.X_test, y_pred_test, PATH_data, norm)
        np.save(calc_dir + f'sep_val_{norm}.npy', sep_val)
        np.save(calc_dir + f'sep_test_{norm}.npy', sep_test)
        print('finish sep')
        
    else:
        print('nothing picked')


if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    run_shuffle_on_data_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5])
    #                         dataset_name, model_name , shuffle_num, metric=None,norm=L1


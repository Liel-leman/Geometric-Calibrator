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



import os
from sep_funcs import *
from config import *
import json


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
    
    data = Data(X_train, X_test, X_val, y_train, y_test, y_val, NUM_LABELS[dataset_name],shuffle_num,dataset_name)
    
    return data
    
    
def save_params(y_pred_val,y_pred_test,y_pred_train,all_predictions_val,all_predictions_test,all_predictions_train,calc_dir,calibrated):
    adder = ""
    if calibrated:
        adder = '_calibrated'
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    np.save(calc_dir + f'y_pred_val{adder}.npy', y_pred_val)
    np.save(calc_dir + f'y_pred_test{adder}.npy', y_pred_test)
    np.save(calc_dir + f'y_pred_train{adder}.npy', y_pred_train)
    np.save(calc_dir + f'all_predictions_val{adder}.npy', all_predictions_val)
    np.save(calc_dir + f'all_predictions_test{adder}.npy', all_predictions_test)
    np.save(calc_dir + f'all_predictions_train{adder}.npy', all_predictions_train)
    


def calc_predictions(data, dataset_name, model_name, shuffle_num, calibrated):
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
    adder = ""
    if calibrated:
        adder = '_calibrated'
    
    
    calc_dir = f'./{dataset_name}/{shuffle_num}/{model_name}/'

    model_cnn_dir = f'./{dataset_name}/{shuffle_num}/model/model_{dataset_name}_CNN_calibrated/'
    model_dir = f'./{dataset_name}/{shuffle_num}/model/'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)        

    # run separation function
    if model_name == "CNN":
        # load model
        pixels = data.get_pixels()
        channels = data.get_channels()
        
        # Reshaping to format which CNN expects (batch, height, width, channels)
        trainX_cnn = data.X_train.reshape(data.X_train.shape[0], pixels, pixels, channels).astype('float32')
        valX_cnn = data.X_val.reshape(data.X_val.shape[0], pixels, pixels, channels).astype('float32')
        testX_cnn = data.X_test.reshape(data.X_test.shape[0], pixels, pixels, channels).astype('float32')

        # Normalize images from 0-255 to 0-1
        trainX_cnn /= 255
        valX_cnn /= 255
        testX_cnn /= 255

        train_y_cnn = utils.to_categorical(data.y_train, data.num_labels)
        val_y_cnn = utils.to_categorical(data.y_val, data.num_labels)
        test_y_cnn = utils.to_categorical(data.y_test, data.num_labels)

            
        if not calibrated: # CNN not cali
            print('at CNN not calibrated')
            if not os.path.exists(model_dir + f'model_{dataset_name}_{model_name}'):
                print('calculating CNN not calibrated')
                isFirstTime = True
                model = eval(f'create_{model_name}_{dataset_name}()')
                model.fit(trainX_cnn, train_y_cnn, validation_split=0.2, epochs=NO_EPOCHS[dataset_name],
                          batch_size=BATCH_SIZE[dataset_name])
                model.save(model_dir + f'model_{dataset_name}_{model_name}')
                
                # predictions
                y_pred_val = np.argmax(model.predict(valX_cnn), axis=1)
                y_pred_test = np.argmax(model.predict(testX_cnn), axis=1)
                y_pred_train = np.argmax(model.predict(trainX_cnn),axis=1)
                
                all_predictions_val   = model.predict(valX_cnn)
                all_predictions_test  = model.predict(testX_cnn)
                all_predictions_train = model.predict(trainX_cnn)


        else: 
            print('at CNN calibrated')
            if not os.path.exists(model_cnn_dir):
                print('calculating CNN calibrated')
                isFirstTime = True
                os.makedirs(model_cnn_dir)
                model = MyKerasClassifier(build_fn=eval(f'create_{model_name}_{dataset_name}'),
                                          epochs=NO_EPOCHS[dataset_name], batch_size=BATCH_SIZE[dataset_name])
                model.fit(trainX_cnn, train_y_cnn)
                model = CalibratedClassifierCV(base_estimator=model, cv='prefit', method='isotonic')
                model.fit(valX_cnn, val_y_cnn)

                y_pred_val = model.predict(valX_cnn)
                y_pred_test = model.predict(testX_cnn)
                y_pred_train = model.predict(trainX_cnn)
                
                all_predictions_test = model.predict_proba(testX_cnn)
                all_predictions_val = model.predict_proba(valX_cnn)
                all_predictions_train = model.predict_proba(trainX_cnn)
        


    else: # all other models (sklearn models)
        existModel = os.path.exists(model_dir+f'model_{dataset_name}_{model_name}.sav')
        if not calibrated:     
            print('at sklearn not calibrated')
            if not existModel:
                print(f'calculating {model_name} not calibrated')
                isFirstTime = True
                model = eval(f'create_{model_name}_{dataset_name}()')
                model = model.fit(data.X_train, data.y_train)
                pickle.dump(model, open(model_dir+f'model_{dataset_name}_{model_name}.sav', 'wb')) 

        else:
            print('at sklearn calibrated')
            existsModelCali=os.path.exists(model_dir+f'model_{dataset_name}_{model_name}_calibrated.sav')
            if not existsModelCali:
                print(f'calculating {model_name} not cali')
                isFirstTime = True
                if existModel:
                    model = pickle.load(open(model_dir+f'model_{dataset_name}_{model_name}.sav',"rb"))
                else:
                    model = eval(f'create_{model_name}_{dataset_name}()')
                    model = model.fit(data.X_train, data.y_train)
                    
                model = CalibratedClassifierCV(base_estimator=model, cv="prefit", method='isotonic')
                model.fit(data.X_val, data.y_val)
                
                pickle.dump(model, open(model_dir+f'model_{dataset_name}_{model_name}_calibrated.sav', 'wb'))

        if isFirstTime:
            y_pred_val            = model.predict(data.X_val)
            y_pred_test           = model.predict(data.X_test)
            y_pred_train          = model.predict(data.X_train)
            all_predictions_val   = model.predict_proba(data.X_val)
            all_predictions_test  = model.predict_proba(data.X_test)
            all_predictions_train = model.predict_proba(data.X_train)
        
        
    if isFirstTime:
       save_params(y_pred_val,y_pred_test,y_pred_train,all_predictions_val,all_predictions_test,all_predictions_train,calc_dir,calibrated)
    else:
       print("preloading calculations")
       y_pred_val = np.load(calc_dir + f'y_pred_val{adder}.npy')
       y_pred_test = np.load(calc_dir + f'y_pred_test{adder}.npy')

    return y_pred_val, y_pred_test



def run_shuffle_on_data_model(dataset_name, model_name, shuffle_num, metric='both', calibrated=False):
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
    y_pred_val, y_pred_test = calc_predictions(data, dataset_name, model_name, shuffle_num, calibrated)

    # save separation file
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)

    adder = ''
    if calibrated:
        adder = '_calibrated'

    ##stab with model
    if metric == 'stab' or metric == 'both':
        stability_val = data.compute_stab('val', y_pred_val)
        stability_test = data.compute_stab('test', y_pred_test)
        np.save(calc_dir + f'stability_test{adder}.npy', stability_test)
        np.save(calc_dir + f'stability_val{adder}.npy', stability_val)
        print('finish stab')

    if metric == 'sep' or metric == 'both':
        wholeSep_val = data.compute_wholeSep('val', y_pred_val)
        wholeSep_test = data.compute_wholeSep('test', y_pred_test)
        np.save(calc_dir + f'wholeSep_test{adder}.npy', wholeSep_test)
        np.save(calc_dir + f'wholeSep_val{adder}.npy', wholeSep_val)
        print('finish sep')
        
    else:
        print('nothing picked')


if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    run_shuffle_on_data_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    #                        dataset_name, model_name , shuffle_num, metric=None, calibrated=False


from sklearn.neighbors import NearestNeighbors

from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt
import os
import math
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


from tensorflow.keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
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
import concurrent.futures
from itertools import repeat

class Data:
    def __init__(self,X_train,X_test,X_val,y_train,y_test,y_val,num_labels,shuffle_num,dataset_name):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.num_labels = num_labels
        self.shuffle_num = shuffle_num
        self.dataset_name = dataset_name
        self.isRGB = "RGB" in dataset_name
        
    
    
    def compute_stab(self,whom,y_pred):
        '''
        input:
                - seperation for whom ? : 
                                        -'test'
                                        -'val'
                - y_pred_val : predicted labels of test\val
                
        return: list of seperations 
        '''
        if whom == 'test':
            return new_stability_calc(self.X_train, self.X_test, self.y_train, y_pred, self.num_labels)
        elif whom == 'val':
            return new_stability_calc(self.X_train, self.X_val, self.y_train, y_pred, self.num_labels)
        else :
            print('error')
            return
        
        
        

        
        
    
    def compute_wholeSep(self,whom,y_pred):
        '''
        input:
                - whom ? : 
                                        -'test'
                                        -'val'
                - y_pred_val : predicted labels of test\val
        return: list of Whole seperations 
        '''
        
        
        data_dir = f'./{self.dataset_name}/{self.shuffle_num}/data/'
        if whom == 'test': 
            return all_test_sep_calc(self.X_test,y_pred,data_dir)
        elif whom == 'val':
            return all_test_sep_calc(self.X_val,y_pred,data_dir)
        else :
            print('error')
            
    def get_channels(self):
        if self.isRGB:
            return 3
        return 1
    
    def get_pixels(self):
        '''
        return the number of pixels real photo has
        '''
        channels = self.get_channels()
        num_of_flatten_pixels = self.X_train.shape[1] / channels
        pixels = int(sqrt(num_of_flatten_pixels))
        return pixels


def two_point_sep_calc(x, x1, x2):
    '''
    given 3 points- the point x and 2 point near it one with the true class and the other with another class
    calculate the sep parameter
    x - is a test example
    o, s are tuple of the distance with x and index of the examples in the training set

            Parameters:
                    x (matrix) : x instance of test/val that we want to calculate the seperation on it.
                    x1 (tuple): instance of train set that is the candidate of same clasification
                    x2 (tuple) : instance of train set that is the candidate of other clasification
                    trainX (list): X instances of train set.
            Returns:
                    return the speration
    '''
    a = np.linalg.norm(x1 - x, 2)
    b = np.linalg.norm(x2 - x, 2)
    c = np.linalg.norm(x1 - x2, 2)

    sep = ((b ** 2 - a ** 2) / (2 * c))
    return sep


def sep_calc_point(x, trainX, train_y, y):
    '''
    given a point x and its label y_pred calculate the separation
    based on the formula:
    min_on_all_other ( max_on_all_same (sep (x,same,other)).
            Parameters:
                    x (matrix) : x instance of test/val that we want to calculate the seperation on it.
                    trainX (list): X instances of train set.
                    train_y (list) : y classes of train set.
                    y (int): predicted class.
            Returns:
                    return the seperation found.
    '''

    # finding points in my classification ('same') , and different clathifications ('others')
    same = [(np.linalg.norm(x - train, 2), index) for index, train in enumerate(trainX) if train_y[index] == y]
    others = [(np.linalg.norm(x - train, 2), index) for index, train in enumerate(trainX) if train_y[index] != y]

    same.sort(key=lambda x: x[0])
    others.sort(key=lambda x: x[0])

    # threshold min_r
    min_r = same[0][0] + 2 * others[0][0]
    sep_other = min_r
    for o in others:
        sep_same = np.NINF
        if o[0] > min_r:
            break
        for s in same:
            if s[0] > min(min_r, o[0]) and o[0] > same[0][0]:
                break
            x_s = trainX[s[1]]
            x_o = trainX[o[1]]
            op = two_point_sep_calc(x, x_s, x_o)
            sep_same = max(op, sep_same)
        sep_other = min(sep_same, sep_other)
        min_r = same[0][0] + 2 * max(0, sep_other)
    return sep_other

def all_test_sep_calc(testX,pred_y,data_dir):
    '''
    calculate the separation of all the examples of (test/val).
            Parameters:
                    trainX (list) : X instances of train set.
                    testX (list): X instances of (test/val) set.
                    train_y (list) : y classes of train set.
                    pred_y (list): y predictionns of (test/val).

            Returns:
                    separation (list) : list of seperations per X instance of (test/train) set.
    '''
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        separation = list(executor.map(sep_parallel,testX,pred_y,repeat(data_dir)))
    return separation



def sep_parallel(x, pred,data_dir):
    # load data
    X_train = np.load(data_dir + 'X_train.npy',mmap_mode='r')
    y_train = np.load(data_dir + 'y_train.npy',mmap_mode='r')
    return sep_calc_point(x, X_train, y_train, pred)

def new_stability_calc(trainX, testX, train_y, test_y_pred, num_labels):
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
    # time_lst = []
    same_nbrs = []
    other_nbrs = []
    for i in range(num_labels):
        idx_other = np.where(train_y != i)
        other_nbrs.append(NearestNeighbors(n_neighbors=1).fit(trainX[idx_other]))
        idx_same = np.where(train_y == i)
        same_nbrs.append(NearestNeighbors(n_neighbors=1).fit(trainX[idx_same]))

    stability = np.array([-1.] * testX.shape[0])

    for i in range(testX.shape[0]):
        # start = time.time()
        x = testX[i]
        pred_label = test_y_pred[i]

        dist1, idx1 = same_nbrs[pred_label].kneighbors([x])
        dist2, idx2 = other_nbrs[pred_label].kneighbors([x])

        stability[i] = (dist2 - dist1) / 2
        # end = time.time()
        # time_lst.append(end-start)
    return stability  # ,time_lst
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import scipy.stats
from tqdm import tqdm
import torchvision.transforms as transforms

from Data import *
from ModelInfo import *

import concurrent.futures
from itertools import repeat




def sigmoid_func(x, x0, k):
    return 1. / (1. + np.exp(-k * (x - x0)))


def split_and_save_range(train_X_original, test_X_original, train_y_original, test_y_original, split_range):
    '''
    Splits the data to range of chunks
            Parameters:
                train_X_original(List)
                test_X_original(List)
                train_y(List)
                test_y(List)
                pixels(Int) - (train_X.shape)[1] , the amount of pixels in flatern array.
                split_range(Range)
            Returns:
                None.

    '''
    if len(train_X_original[0].shape) != 1:
        # length of row * col * channels (when RGB)
        pixels = 1
        for dim in train_X_original[0].shape:
            pixels *= dim  # multiply all dims
    else:  # its already flatten
        pixels = train_X_original[0].shape[0]

    for i in split_range:
        trainX = train_X_original.reshape(len(train_X_original), pixels).astype(np.float64)
        testX = test_X_original.reshape(len(test_X_original), pixels).astype(np.float64)
        data = np.concatenate((trainX, testX), axis=0)
        y = np.concatenate((train_y_original, test_y_original), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=i)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=i)
        print(f'shuffle :{i}')
        print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
        directory = f'./{i}/data/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + '/X_train.npy', X_train)
        np.save(directory + '/X_test.npy', X_test)
        np.save(directory + '/X_val.npy', X_val)
        np.save(directory + '/y_train.npy', y_train)
        np.save(directory + '/y_test.npy', y_test)
        np.save(directory + '/y_val.npy', y_val)


def fitting_function(stability, y_true, plot=False):
    '''
    return all the popts
    params :
             - stability - metric to calcultation of val or test
             - y_true    - binary (true/false) for calsification index

    return : best_fiting_curve(String)
             popt_lst(List of [z_stab,popt_stab_exp,popt_stab_square,popt_stab_log,popt_stab_inverse_x,popt_stab_inverse_poly])
             min_idx(where the index that return the minimum error interpulation)

    '''

    # compute s_acc_stab = dict { stability : accuracy(1/0)
    s_acc_stab, _, _ = calc_acc(stability, y_true)

    xdata_stab = np.array([x for x in s_acc_stab.keys()])  # stability
    ydata_stab = np.array([x for x in s_acc_stab.values()])  # accuracy

    # function
    isotonic_regression = IsotonicRegression(out_of_bounds="clip").fit(xdata_stab[:, None], ydata_stab)
    p0 = [max(ydata_stab), min(ydata_stab)]
    popt_stab_sigmoid, _ = curve_fit(sigmoid_func, xdata_stab, ydata_stab, p0, maxfev=1000000)

    popt_lst = [isotonic_regression, popt_stab_sigmoid]
    return popt_lst


# def fitting_function(stability, y_real, y_pred, plot=False):
#     '''
#     return all the popts
#     params :
#              - stability - metric to calcultation of val or test
#              - y_real    - real classification of val or test
#              - y_pred    - predicted classification by some model of val or test

#     return : best_fiting_curve(String)
#              popt_lst(List of [z_stab,popt_stab_exp,popt_stab_square,popt_stab_log,popt_stab_inverse_x,popt_stab_inverse_poly])
#              min_idx(where the index that return the minimum error interpulation)

#     '''

#     # compute s_acc_stab = dict { stability : accuracy(1/0)
#     s_acc_stab, _, _ = calc_acc(stability, y_real, y_pred)

#     xdata_stab = np.array([x for x in s_acc_stab.keys()])  # stability
#     ydata_stab = np.array([x for x in s_acc_stab.values()])  # accuracy

#     # function
#     isotonic_regression = IsotonicRegression(out_of_bounds="clip").fit(xdata_stab[:, None], ydata_stab)
#     p0 = [max(ydata_stab), min(ydata_stab)]
#     popt_stab_sigmoid, _ = curve_fit(sigmoid_func, xdata_stab, ydata_stab, p0, maxfev=1000000)

#     popt_lst = [isotonic_regression, popt_stab_sigmoid]
#     return popt_lst


def calc_acc(stability, y_true):
    '''
        returns the dicts of description of stability \ seperation (accuracy,#num of True samples, #num of samples)

                Parameters:
                        stability (list of floats ) : stability list of calculations
                        y_true (list): binary (true/false) for specific index 

                Returns:
                        s_acc (dict): {key = normalize unique stability , value = accuracy for that stability}
                        s_true (dict): {key = normalize unique stability , value = amount of true samples per this stability}
                        s_all (dict): {key = normalize unique stability , value = amount of instances exist for that stability}
    '''
    test_size = y_true.shape[0]

    # normalization and geting the uniques values
    stab_values, reps = np.unique(stability, return_counts=True)

    # creating dictionaries
    s_true = dict(zip(stab_values, [0] * (stab_values.shape[0])))
    s_all = dict(zip(stab_values, reps))
    s_acc = s_true.copy()
    # counting number of vtrue classifications
    for i in range(test_size):
        stab = stability[i]
        if y_true[i]:
            s_true[stab] += 1
    for stab in s_all.keys():
        s_acc[stab] = s_true[stab] / s_all[stab]

    return s_acc, s_true, s_all


# def calc_acc(stability, test_y, y_pred):
#     '''
#         returns the dicts of description of stability \ seperation (accuracy,#num of True samples, #num of samples)

#                 Parameters:
#                         stability (list of floats ) : stability list of calculations
#                         test_y (list): predictions on test set
#                         y_pred (list) : list of predictions of the model

#                 Returns:
#                         s_acc (dict): {key = normalize unique stability , value = accuracy for that stability}
#                         s_true (dict): {key = normalize unique stability , value = amount of true samples per this stability}
#                         s_all (dict): {key = normalize unique stability , value = amount of instances exist for that stability}
#     '''
#     test_size = test_y.shape[0]

#     # normalization and geting the uniques values
#     stab_values, reps = np.unique(stability, return_counts=True)

#     # creating dictionaries
#     s_true = dict(zip(stab_values, [0] * (stab_values.shape[0])))
#     s_all = dict(zip(stab_values, reps))
#     s_acc = s_true.copy()
#     # counting number of vtrue classifications
#     for i in range(test_size):
#         stab = stability[i]
#         if y_pred[i] == test_y[i]:
#             s_true[stab] += 1
#     for stab in s_all.keys():
#         s_acc[stab] = s_true[stab] / s_all[stab]

#     return s_acc, s_true, s_all


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h




### data handling ###


def load_data(dataset_name, shuffle_num):
    '''
    loads the data of specific shuffle
    '''
    try:
        VARS = json.load(open('../SLURM/VARS.json'))
    except:
        VARS = json.load(open('./SLURM/VARS.json'))

    NUM_LABELS = VARS['NUM_LABELS']

    data_dir = f'./{dataset_name}/{shuffle_num}/data/'

    # load data
    X_train = np.load(data_dir + 'X_train.npy')
    X_test = np.load(data_dir + 'X_test.npy')
    X_val = np.load(data_dir + 'X_val.npy')
    y_train = np.load(data_dir + 'y_train.npy')
    y_test = np.load(data_dir + 'y_test.npy')
    y_val = np.load(data_dir + 'y_val.npy')

    isRGB = "RGB" in dataset_name
    data = Data(X_train, X_test, X_val, y_train, y_test, y_val, NUM_LABELS[dataset_name], isRGB)

    return data


def load_model(dataset_name, model_name, shuffle_num, isCalibrate=False):
    '''
    loads the model of specific shuffle
    '''
    data = load_data(dataset_name, shuffle_num)

    calc_dir = f'{dataset_name}/{shuffle_num}/{model_name}/'

    adder = "_calibrated" if isCalibrate else ""

    all_predictions_val = np.load(calc_dir + f'all_predictions_val{adder}.npy', allow_pickle=True)
    all_predictions_test = np.load(calc_dir + f'all_predictions_test{adder}.npy', allow_pickle=True)
    all_predictions_train = np.load(calc_dir + f'all_predictions_train{adder}.npy', allow_pickle=True)
    y_pred_test = np.load(calc_dir + f'y_pred_test{adder}.npy', allow_pickle=True)
    y_pred_val = np.load(calc_dir + f'y_pred_val{adder}.npy', allow_pickle=True)
    y_pred_train = np.load(calc_dir + f'y_pred_train{adder}.npy', allow_pickle=True)


    return ModelInfo(data, y_pred_val, all_predictions_val, y_pred_test, all_predictions_test, y_pred_train,
                     all_predictions_train, dataset_name, model_name, shuffle_num, isCalibrate)


def load_shuffle(dataset_name, model_name, shuffle_num, isCalibrate=False, print_acc=False):
    """
    loads the [data,prediction's,probe's] of specific shuffle
    """
    calc_dir = f'./{dataset_name}/{shuffle_num}/{model_name}/'

    adder = "_calibrated" if isCalibrate else ""

    model_info = load_model(dataset_name, model_name, shuffle_num, isCalibrate)

    try:
        stability_test = np.load(calc_dir + f'/stability_test{adder}.npy')
        stability_val = np.load(calc_dir + f'/stability_val{adder}.npy')
    except:
        stability_test = None
        stability_val = None
    try:
        sep_test = np.load(calc_dir + f'/sep_test{adder}.npy')
        sep_val = np.load(calc_dir + f'/sep_val{adder}.npy')
    except:
        sep_test = None
        sep_val = None

    model_info.stability_test = stability_test
    model_info.stability_val = stability_val
    model_info.sep_test = sep_test
    model_info.sep_val = sep_val

    if print_acc:
        print(f'shuffle : {shuffle_num}')
        print('val accuracy:', accuracy_score(model_info.data.y_val, model_info.y_pred_val))
        print('test accuracy:', accuracy_score(model_info.data.y_test, model_info.y_pred_test))

    return model_info


########## stability vs seperation ####################
def stability_calc(trainX, testX, train_y, test_y_pred, num_labels,metric='minkowski'):
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
        other_nbrs.append(NearestNeighbors(n_neighbors=1,metric=metric).fit(trainX[idx_other]))
        idx_same = np.where(train_y == i)
        same_nbrs.append(NearestNeighbors(n_neighbors=1,metric=metric).fit(trainX[idx_same]))

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


def stab_calc_vector(X_train, X_test, y_train, y_pred_test, num_labels):
    '''
    Calculates the stability of the test set.
            Parameters:
                    X_train (List)
                    X_test (List)
                    y_train (List)
                    y_pred_test (list)
                    num_labels (Int)
            Returns:
                    stabs(List) - vector of stabilitys with None in predicted place
    '''
    stabs = [[] for i in range(len(X_test))]
    same_nbrs = []

    # create 1NN tree for separate classes
    for i in range(num_labels):
        idx_same = np.where(y_train == i)
        same_nbrs.append(NearestNeighbors(n_neighbors=1).fit(X_train[idx_same]))

    for i, x in tqdm(enumerate(X_test)):
        stab = np.zeros(num_labels)  # vectorized stab [0,0,None,...,0]
        pred = y_pred_test[i]
        dist1, idx1 = same_nbrs[pred].kneighbors([x])  # same label closed dist

        # compute for different labels
        for label in range(num_labels):
            if label == y_pred_test[i]:
                stab[label] = None  # if its the predicted label so it should be None
                continue
            dist2, _ = same_nbrs[label].kneighbors([x])
            stab[label] = (dist2 - dist1) / 2

        stabs[i] = stab
    return np.array(stabs)


def sep_calc_parallel(testX, pred_y, data_dir, norm='L2'):
    """
    calculate the separation of all the examples of (test/val).
            Parameters:
                    trainX (list) : X instances of train set.
                    testX (list): X instances of (test/val) set.
                    train_y (list) : y classes of train set.
                    pred_y (list): y predictionns of (test/val).

            Returns:
                    separation (list) : list of seperations per X instance of (test/train) set.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        separation = list(executor.map(sep_parallel, testX, pred_y, repeat(data_dir), repeat(norm)))
    return separation 


def sep_parallel(x, pred, data_dir, norm='L1'):
    # load data
    X_train = np.load(data_dir + 'X_train.npy', mmap_mode='r')
    y_train = np.load(data_dir + 'y_train.npy', mmap_mode='r')
    return sep_calc_point(x, X_train, y_train, pred, norm)


def sep_calc(trainX, testX, train_y, pred_y, norm):
    """
    calculate the separation of all the examples of (test/val).
            Parameters:
                    trainX (list) : X instances of train set.
                    testX (list): X instances of (test/val) set.
                    train_y (list) : y classes of train set.
                    pred_y (list): y predictionns of (test/val).
                    num_label (int) : amount of labels dataset has.

            Returns:
                    separation (list) : list of seperations per X instance of (test/train) set.
    """
    separation = [sep_calc_point(x, trainX, train_y, pred_y[i],norm) for i, x in enumerate(testX)]
    return separation


def sep_calc_point(x, trainX, train_y, y,norm = 'L1'):
    """
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
    """
    if norm == 'L1':
        norm = 1
    elif norm == 'L2':
        norm = 2
    elif norm == 'Linf':
        norm = 'inf'
    else:
        raise ValueError(f'Not supported Norm: {norm}') 
    
    # finding points in my classification ('same') , and different clathifications ('others')
    same = [(np.linalg.norm(x - train, norm), index) for index, train in enumerate(trainX) if train_y[index] == y]
    others = [(np.linalg.norm(x - train, norm), index) for index, train in enumerate(trainX) if train_y[index] != y]

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
            op = two_point_sep_calc(x, x_s, x_o,norm)
            sep_same = max(op, sep_same)
        sep_other = min(sep_same, sep_other)
        min_r = same[0][0] + 2 * max(0, sep_other)
    return sep_other


def two_point_sep_calc(x, x1, x2, norm=1):
    """
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
    """
    
    a = np.linalg.norm(x1 - x, norm)
    b = np.linalg.norm(x2 - x, norm)
    c = np.linalg.norm(x1 - x2, norm)

    sep = ((b ** 2 - a ** 2) / (2 * c))
    return sep


################################## main results functions ##################################

def ECE_calc(probs, y_pred, y_real, bins=15):
    """
    params :
                probs - vector of the toplabel probabilities
                y_pred - predicted y by model
                y_real - real label
                bins - bins calculated on
    return :
                ECE - expected calibration error

    """

    def gap_calc(lst):
        # lst[1:] - the prob values in bucket
        # lst[0]  - number of instances collected that was true
        if lst == [0]:
            return 0
        s_lst = sum(lst[1:])
        l_lst = len(lst[1:])
        avg = s_lst / l_lst
        accuracy = lst[0] / l_lst
        return abs(avg - accuracy) * l_lst

    # if we send the 'Predeict_proba' as it is we need to take the maximum of its values:
    if isinstance(probs, np.ndarray):
        if len(probs.shape) == 2:
            probs = [max(i) for i in probs]

    # create bins
    # we use bin size+1 because the last been is not counted in our implementatiton
    # thats because bin of [1,) is not interesting and instead we fil 1 values in previous bin
    lin_space = np.linspace(0, 1, bins + 1)

    ziped = list(zip(probs, y_pred == y_real))
    ziped.sort(key=lambda x: x[0])

    # bucket divider
    b = [[0] for i in range(len(lin_space))]
    b_num = 0
    for x in ziped:
        p = x[0]
        inserted = False
        while not inserted:
            if p == 1:  # cannot be higher than one
                b[-2].append(p)  # last bucket
                inserted = True
            elif p < lin_space[b_num + 1]:
                b[b_num].append(p)
                inserted = True
            else:
                b_num += 1  # go for higher bucket ( its sorted)

        ## inc the counter if we correct
        if x[1]:
            if p == 1:
                b[-2][0] += 1
            else:
                b[b_num][0] += 1

    # calc the ECE error
    ECE_sum = 0
    for idx, data in enumerate(b):
        # data[0] how many times accured
        ECE_sum += gap_calc(data)
    ECE = ECE_sum / len(y_pred)

    return ECE





def plot_fitting_function(model_info, n_bins, save=False):
    stab_latex = r'$\underline{\mathcal{S}}^{\mathcal{M}}$' 
    correct = model_info.y_pred_val == model_info.data.y_val
    popt = fitting_function(model_info.stability_val,correct)  # [isotonic_regression , popt_stab_sigmoid]
    ylabels = model_info.y_pred_test == model_info.data.y_test
    xlabels = model_info.stability_test

    length = (max(xlabels) - min(xlabels)) / n_bins
    bins_data = [0 for i in range(n_bins + 1)]
    bins_data_num = [0 for i in range(n_bins + 1)]
    for i in range(len(xlabels)):
        bins_data[int((xlabels[i] - min(xlabels)) / length)] += ylabels[i]
        bins_data_num[int((xlabels[i] - min(xlabels)) / length)] += 1
    ydata = [[], []]
    xdata = [[], []]
    plot_x = []
    y_data_return = []
    all_col = []
    colors = ["r", "b"]
    markers = ['o',"D"]
    for i in range(n_bins + 1):
        if bins_data_num[i] == 0:
            continue
        if bins_data_num[i] < 100:
            idx = 0
        else:
            idx = 1
        ydata[idx].append(bins_data[i] / bins_data_num[i])
        xdata[idx].append(length * i + min(xlabels))
        plot_x.append(length * i + min(xlabels))
        y_data_return.append(bins_data[i] / bins_data_num[i])
    plt.xlabel('Fast Separation Score ' + stab_latex)
    plt.ylabel("Accuracy on Validation Set")
    for i in range(len(colors)):
        plt.scatter(xdata[i], ydata[i], c=colors[i], marker = markers[i])
    xdata = np.array(plot_x) 
    # sigmoid
    plt.plot(xdata, sigmoid_func(xdata, *popt[1]), color='k')
    # isotonic
    plt.plot(xdata, popt[0].predict(xdata.reshape(-1, 1)), color='g')
    #     plt.title(f'{model_info.data.dataset_name}-{model_info.data.model_name}-{model_info.data.shuffle_num}')
    plt.legend(["Less than 100 samples", "More than 100 samples","Sigmoid fitting", "Isotonic regression"])

    if save:
        plt.savefig('plot.pdf')
    plt.show()


# Normalize for R, G, B with img = img - mean / std
def normalize_dataset(data):
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize


def hot_padding(oneDim, positions, num_labels):
    hot_padding_probs = np.zeros((len(oneDim), num_labels))
    for i, pos in enumerate(positions):
        hot_padding_probs[i][pos] = oneDim[i]

    return hot_padding_probs


def calculate_avarege_acc(model_name, dataset_name, range_input=range(10)):
    print(f'Computing accuracy of {model_name}-{dataset_name}..')

    acc_lst = []
    for shuffle_num in tqdm(range_input):
        # load data
        if model_name == 'CNN':
            PATH = f'./{dataset_name}/{shuffle_num}/pytorch/acc_test.npy'
            acc_lst.append(np.load(PATH))
        else:
            model_info = load_shuffle(dataset_name, model_name, shuffle_num, isCalibrate=False)
            acc_lst.append(accuracy_score(model_info.data.y_test, model_info.y_pred_test))

    avg_acc = sum(acc_lst) / len(acc_lst)
    return pd.Series([avg_acc], index=[f'{model_name}-{dataset_name}'])


def get_bin(s, bins_ranges):
    import bisect
    return bisect.bisect(bins_ranges, s) - 1

def histogramBinning(probs,corrects,num_bins):
    #split to bins
    bins_nums,bins_ranges = np.histogram(probs,bins=num_bins)
    
    binned_values = [[] for _ in range(num_bins)]
    
    for prob, value in list(zip(probs, corrects)):
        bin_idx = get_bin(prob, bins_ranges) 
        if bin_idx>num_bins-1:
            binned_values[bin_idx-1].append(float(value))
        else:
            binned_values[bin_idx].append(float(value))  
            
    #check if we split the values to each bin in the right order as in np.hist
    assert ([len(b) for b in binned_values] == bins_nums).all()
    
    #get the mean of each bin
    bin_means = [np.mean(values) for values in binned_values]
    
    #empty bins clip
    for idx,val in enumerate(bin_means):
        if np.isnan(val):
            bin_means[idx] = bin_means[idx-1]
            
            
    #the new_ranges would be the mean of each inteval(Due to an extra value in list)       
    new_ranges = []
    for i in range(1,len(bins_ranges)):
        new_ranges.append( (bins_ranges[i-1] + bins_ranges[i]) / 2 )
        
    return bin_means,bins_ranges,new_ranges



######################################  Dataframe formating ######################################
def color_max(s):
    numbers = []
    for i in s:
        if isinstance(i,str) and len(i)>1:
            numbers.append(i)
        else:
            numbers.append(np.inf)
    numbers = np.array(numbers)
    is_max = numbers == min(numbers)
    return ['background-color: darkgreen' if v else '' for v in is_max]

def percentage_format(x):
    if isinstance(x,str) and len(x)>1:
        a,b = x.split('±')
        a=float(a)*100
        b=float(b)*100
        return f'{format(a, ".2f")}±{format(b, ".2f")}'
    return x

def mean_confidence_interval_str(data, confidence=0.95):
    if isinstance(data, list):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return f'{format(m, ".6f")}±{format(h, ".6f")}'
    else:
        return '-'
    
def non_zero_format(x):
    if x!='-':
        a,b = x.split('±')
        if a[:2]=='0.':
            a=a[1:]
        if b[:2]=='0.':
            b=b[1:]
        return f'{a}±{b}'
    return x

def order_by(indexes,order,second_order):
    ans = []
    first_sort = []
    for p in order:
        first_sort.append([item for item in indexes if item.startswith(p)])
    for lst in first_sort:
        for p in second_order:
            ans.extend([item for item in lst if item.endswith(p)])
    return ans
    

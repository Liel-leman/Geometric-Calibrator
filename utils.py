import numpy as np
from sklearn.neighbors import NearestNeighbors
from Data import *
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import json
import scipy.stats
from ModelInfo import *
import pandas as pd

def sigmoid_func(x,x0, k):
    return 1 / (1 + np.exp(-k*(x-x0)))

def fitting_function(stability, y_real, y_pred, plot=False):
    '''
    return all the popts
    params :
             - stability - metric to calcultation of val or test
             - y_real    - real classification of val or test
             - y_pred    - predicted classification by some model of val or test

    return : best_fiting_curve(String)
             popt_lst(List of [z_stab,popt_stab_exp,popt_stab_square,popt_stab_log,popt_stab_inverse_x,popt_stab_inverse_poly])
             min_idx(where the index that return the minimum error interpulation)

    '''

    # compute s_acc_stab = dict { stability : accuracy(1/0)
    s_acc_stab, _, _ = calc_acc(stability, y_real, y_pred)

    xdata_stab = np.array([x for x in s_acc_stab.keys()])  # stability
    ydata_stab = np.array([x for x in s_acc_stab.values()])  # accuracy

    # function
    isotonic_regression = IsotonicRegression(out_of_bounds="clip").fit(xdata_stab[:, None], ydata_stab)
    p0 = [max(ydata_stab), min(ydata_stab)]
    popt_stab_sigmoid, _ = curve_fit(sigmoid_func, xdata_stab, ydata_stab, p0, maxfev=1000000)

    popt_lst = [isotonic_regression, popt_stab_sigmoid]
    return popt_lst


def calc_acc(stability, test_y, y_pred):
    '''
        returns the dicts of description of stability \ seperation (accuracy,#num of True samples, #num of samples)

                Parameters:
                        stability (list of floats ) : stability list of calculations
                        test_y (list): predictions on test set
                        y_pred (list) : list of predictions of the model

                Returns:
                        s_acc (dict): {key = normalize unique stability , value = accuracy for that stability}
                        s_true (dict): {key = normalize unique stability , value = amount of true samples per this stability}
                        s_all (dict): {key = normalize unique stability , value = amount of instances exist for that stability}
    '''
    test_size = test_y.shape[0]

    # normalization and geting the uniques values
    stab_values, reps = np.unique(stability, return_counts=True)

    # creating dictionaries
    s_true = dict(zip(stab_values, [0] * (stab_values.shape[0])))
    s_all = dict(zip(stab_values, reps))
    s_acc = s_true.copy()
    # counting number of vtrue classifications
    for i in range(test_size):
        stab = stability[i]
        if y_pred[i] == test_y[i]:
            s_true[stab] += 1
    for stab in s_all.keys():
        s_acc[stab] = s_true[stab] / s_all[stab]

    return s_acc, s_true, s_all




def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



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
    y_pred_test = np.load(calc_dir + f'y_pred_test{adder}.npy', allow_pickle=True)
    y_pred_val = np.load(calc_dir + f'y_pred_val{adder}.npy', allow_pickle=True)
    y_pred_train = np.load(calc_dir + f'y_pred_train{adder}.npy', allow_pickle=True)
    all_predictions_train = np.load(calc_dir + f'all_predictions_train{adder}.npy', allow_pickle=True)

    return ModelInfo(data, y_pred_val, all_predictions_val, y_pred_test, all_predictions_test, y_pred_train,
                     all_predictions_train, isCalibrate)

def load_shuffle(dataset_name, model_name, shuffle_num, isCalibrate=True, print_acc=False):
    '''
    loads the [data,prediction's,proba's] of specific shuffle
    '''
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
        sep_test = np.load(calc_dir + f'/WholeSep_test{adder}.npy')
        sep_val = np.load(calc_dir + f'/WholeSep_val{adder}.npy')
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






########### stability vs seperation ####################
def stability_calc(trainX, testX, train_y, test_y_pred, num_labels):
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


def sep_calc(trainX, testX, train_y, pred_y):
    '''
    calculate the separation of all the examples of (test/val).
            Parameters:
                    trainX (list) : X instances of train set.
                    testX (list): X instances of (test/val) set.
                    train_y (list) : y classes of train set.
                    pred_y (list): y predictionns of (test/val).
                    num_label (int) : amount of labels dataset has.

            Returns:
                    separation (list) : list of seperations per X instance of (test/train) set.
    '''
    separation = [ sep_calc_point( x, trainX, train_y, pred_y[i] ) for i,x in enumerate(testX) ]
    return separation


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


################################## main results functions ##################################

def ECE_calc(probs, y_pred, y_real, bins=15):
    '''
    params :
                probs - vector of the highest probabilities of proba
                y_pred - predicted y by model
                y_real - real label
                bins - bins calculated on
    return :
                ECE - expected calibration error

    '''

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
        if len(probs.shape) > 1:
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





def compare_calib_table(model_name, dataset_name, range_input=range(10), fitting='isotonic_regression', err_metric='ECE'):
    '''
    *** compare with calibrated ***
    The function load all the computed data and display it in dataframe (stability and sep)

            Parameters:
                model_name(String)
                dataset_name(String)
                range_input(range)
                fitting = 'isotonic_regression''sigmoid'

            Returns:
                None.

    '''
    methods = ['stability', 'sep', 'sklearn', 'Base', 'SBC', 'HB']
    errors = {method: [] for method in methods}
    for shuffle_num in range_input:

        # load data
        model_info_cali = load_shuffle(dataset_name, model_name, shuffle_num, isCalibrate=True, print_acc=False)
        model_info = load_shuffle(dataset_name, model_name, shuffle_num, isCalibrate=False)

        #         prep_to_plot(model_info) # gabi func.

        # compute on not calibrated models
        for calibration_method in methods:
            # github.com/p-lambda/verified_calibration - article imp
            # github.com/aigen/df-posthoc-calibration - article imp
            if calibration_method == 'sklearn':
                err = model_info_cali.compute_error_metric(calibration_method, err_Func = ECE_calc)  # calibrated
            else:
                err = model_info.compute_error_metric(calibration_method,err_Func = ECE_calc)
            errors[calibration_method].append(err)

    err_metrics_names = [err_metric + "-" + mothods for mothods in errors.keys()]
    ans = []

    print(f'{model_name}-{dataset_name}:')
    # conf interval calc
    toexcel = []

    # key = cali method , val = error of some metric
    for k, v in errors.items():
        m, m_minus_h, _ = mean_confidence_interval(v)  # m, m-h, m+h
        h = m - m_minus_h
        ans.append(f'{format(m, ".4f")}+-{format(h, ".4f")}')
        print(f'{dataset_name}-{model_name}-{k}:  {m}+-{h}')
        toexcel.append(f'{format(m, ".4f")}+-{format(h, ".4f")}')

    # writeToExcel(dataset_name, model_name, err_metrics_names, toexcel)

    def color_max(s):
        numbers = []
        for i in s:
            numbers.append(float(i[:6]) + (0.001 * float(i[9:])))
        numbers = np.array(numbers)
        is_max = numbers == numbers.min()
        return ['background-color: lightgreen' if v else '' for v in is_max]

    indx = f'{model_name}-{dataset_name}'
    df = pd.DataFrame([ans], columns=err_metrics_names, index=[indx])
    display(df.style.apply(color_max, axis=1))
    return df

def color_max(s):
    numbers = []
    for i in s:
        numbers.append(float(i[:6])+(float(i[9:])/1_000))
    numbers = np.array(numbers)
    is_max = numbers == numbers.min()
    return ['background-color: lightgreen' if v else '' for v in is_max]
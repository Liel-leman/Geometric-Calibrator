from math import sqrt
import tensorflow as tf
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


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
    corrects = y_pred == y_real
    ziped = list(zip(probs, corrects))
    ziped.sort(key=lambda item: item[0])

    # bucket divider
    b = [[0] for _ in range(len(lin_space))]
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

        # inc the counter if we correct
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


def stability_calc(trainX, testX, train_y, test_y_pred, metric='l2'):
    """
    Calculates the stability of the test set.
            Parameters:
                    :param trainX: Train space.
                    :param testX: the set that we calculate teh stability metric on.
                    :param train_y: Train labels.
                    :param test_y_pred: model predicted labels on the set that we want to calculate its stability.
                    :param metric: metric of distance to compute the stability value.
            Returns:
                    stability(List)

    """
    trainX = np.array(trainX)
    testX = np.array(testX)
    num_labels = len(set(train_y))
    same_nbrs = []
    other_nbrs = []
    for i in range(num_labels):
        idx_other = np.where(train_y != i)
        other_nbrs.append(NearestNeighbors(n_neighbors=1, metric=metric).fit(trainX[idx_other]))
        idx_same = np.where(train_y == i)
        same_nbrs.append(NearestNeighbors(n_neighbors=1, metric=metric).fit(trainX[idx_same]))

    stability = np.array([-1.] * testX.shape[0])

    for i in tqdm(range(testX.shape[0])):
        x = testX[i]
        pred_label = test_y_pred[i]

        dist1, idx1 = same_nbrs[pred_label].kneighbors([x])
        dist2, idx2 = other_nbrs[pred_label].kneighbors([x])

        stability[i] = (dist2 - dist1) / 2
    return stability


def sep_calc(trainX, testX, train_y, pred_y, norm="l2"):
    """
    calculate the separation of all the examples of (test/val) set.
            Parameters:
                    trainX (list) : X instances of train set.
                    testX (list): X instances of (test/val) set.
                    train_y (list) : y classes of train set.
                    pred_y (list): y predictionns of (test/val).
                    norm (str) : type of norm.

            Returns:
                    separation (list) : list of separations for testX set.
    """
    separation = [sep_calc_point(x, trainX, train_y, pred_y[i], norm) for i, x in tqdm(enumerate(testX))]
    return separation


def sep_calc_point(x, trainX, train_y, y, norm='l2'):
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
    if norm == 'l1':
        norm = 1
    elif norm == 'l2':
        norm = 2
    elif norm == 'linf':
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
            op = two_point_sep_calc(x, x_s, x_o, norm)
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


def apply_compression(trainX, train_y, compression_type, red_param, train=True, pca=None):
    pixels = int(sqrt(trainX.shape[1]))
    pca = None
    if compression_type == 'Avgpool':
        polling = torch.nn.AvgPool2d(red_param)
        trainX = polling(torch.tensor(trainX.reshape((len(trainX), pixels, pixels)))).reshape(len(trainX), -1)

    elif compression_type == 'Maxpool':
        polling = torch.nn.MaxPool2d(red_param)
        trainX = polling(torch.tensor(trainX.reshape((len(trainX), pixels, pixels)))).reshape(len(trainX), -1)

    elif compression_type == 'resize':
        size = pixels // red_param
        length = len(trainX)
        X_train = torch.tensor(trainX.reshape(length, pixels, pixels))
        X_train = X_train[..., tf.newaxis]
        trainX = tf.image.resize(X_train, [size, size]).numpy().reshape(length, -1)

    elif compression_type == 'PCA':
        if train:
            size = pixels // red_param
            pca = PCA(n_components=size ** 2)
            trainX = pca.fit_transform(trainX)
        if not train:
            size = pixels // red_param
            trainX = model.transform(trainX.reshape(1, -1))
    elif compression_type == 'randpix':
        size = (pixels // red_param) ** 2
        string = np.random.randint(0, 255, size=size)
        trainX = trainX[:, string]

    elif compression_type == 'randset':
        if train == True:
            length = len(trainX)
            size = (length // (red_param ** 2))
            string = np.random.randint(0, length, size=size)
            trainX = trainX[string, :]
            train_y = train_y[string]

    else:
        print("error reduce type")

    return trainX, train_y, pca


# calibrators

class BaseCalibrator:
    """ 
    Abstract calibrator class
    """

    def __init__(self):
        self.num_labels = None

    def fit(self):
        raise NotImplementedError

    def calibrate(self):
        raise NotImplementedError


class GeometricCalibrator(BaseCalibrator):
    def __init__(self, model, X_train, y_train, method='Approx Seperation', comprasion_mode='Maxpool',
                 comprassion_param=2):
        super().__init__()
        self.Iso = None
        self._fitted = False
        self.model = model
        self.num_labels = len(set(y_train))

        # method
        if method == 'Approx Seperation':
            self.geo_func = stability_calc
        elif method == 'Seperation':
            self.geo_func = sep_calc
        else:
            raise ValueError("Wrong method variable, need to be 'Approx Seperation' or 'Seperation'")

        # Comprassion
        self.comprasion_mode = comprasion_mode
        self.comprassion_param = comprassion_param
        self.X_train_compressed, self.y_train_compressed, self.pca = apply_compression(X_train, y_train,
                                                                                       compression_type=comprasion_mode,
                                                                                       red_param=comprassion_param,
                                                                                       train=True, pca=None)

    def fit(self, X_val, y_val):
        # compression
        X_val_compressed, _, _ = apply_compression(X_val, y_val, compression_type=self.comprasion_mode,
                                                   red_param=self.comprassion_param,
                                                   train=False, pca=self.pca)
        y_pred_val = self.model.predict(X_val)
        correct = y_val == y_pred_val
        stability_val = self.geo_func(self.X_train_compressed, X_val_compressed, self.y_train_compressed, y_pred_val)
        self.Iso = IsotonicRegression(out_of_bounds="clip").fit(stability_val, correct)
        self._fitted = True

    def calibrate(self, X_test, y_test):

        # compression
        X_test_compressed, _, _ = apply_compression(X_test, y_test, compression_type=self.comprasion_mode,
                                                    red_param=self.comprassion_param,
                                                    train=False, pca=self.pca)

        if not self._fitted:
            raise ValueError('You must fit you calibrator first')
        y_pred_test = self.model.predict(X_test)
        stability_test = self.geo_func(self.X_train_compressed, X_test_compressed, self.y_train_compressed, y_pred_test)
        calibrated_probs = self.Iso.predict(stability_test)
        return calibrated_probs


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,  # row number
        n_features=900,  # n_features should represent image thus the number need to be from y=x^2
        n_informative=6,  # The number of informative features
        n_classes=2,  # The number of classes
        random_state=42  # random seed
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    model = RandomForestClassifier().fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_test_probs = model.predict_proba(X_test)
    print(f'accuracy:{accuracy_score(y_test, y_pred_test)}')

    # approx separation calibration
    GeoCalibratorAS = GeometricCalibrator(model, X_train, y_train, method="Approx Seperation",
                                          comprasion_mode='Maxpool', comprassion_param=2)
    GeoCalibratorAS.fit(X_val, y_val)
    calibrated_prob_AS = GeoCalibratorAS.calibrate(X_test, y_test)

    #  separation calibration
    GeoCalibratorS = GeometricCalibrator(model, X_train, y_train, method="Seperation",
                                         comprasion_mode='Maxpool', comprassion_param=2)
    GeoCalibratorS.fit(X_val, y_val)
    calibrated_prob_S = GeoCalibratorS.calibrate(X_test, y_test)

    # After Calibration
    print(f'Geometric Calibration approx seperation ECE: \t{(ECE_calc(calibrated_prob_AS, y_pred_test, y_test)):.4f}')
    print(f'Geometric Calibration seperation ECE : \t{(ECE_calc(calibrated_prob_S, y_pred_test, y_test)):.4f}')
    print(f'No Calibration ECE: \t{(ECE_calc(y_test_probs, y_pred_test, y_test)):.4f}')

from math import sqrt
import tensorflow as tf
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm


# Evaluation
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


class Compression():
    """
    Class used as compression of data sample space.
    The different supported compression methods are:
    -Avgpool
    -Maxpool
    -resize
    -PCA
    -randpix
    -randset
    """
    def __init__(self, compression_type, comprassion_param):
        """
        @param compression_type: string of (Avgpool or Maxpool or ...)
        @param compression_param: hyper param, the higher-> the higher the compression
        """

        self.compression_type = compression_type
        self.red_param = comprassion_param
        self.pca_model = None  # Would get value only if compression_type='PCA'

    def __call__(self, X_train, y_train, train=True):
        """
        On call the function if it train set it compressing X and y appropriately,
        otherwise it compresses only X.
        @param X_train: ndarray of flattened images.
        @param y_train: ndarray of labels.
        @param train: bool value to know if we are inside train set.
        @return: X,y that are compressed by desired method.
        """

        n = X_train.shape[1]
        if not sqrt(n).is_integer():
            print('Running without compression, the shape of X need to be square')
            return X_train, y_train

        pixels = int(sqrt(X_train.shape[1]))
        if self.compression_type == 'Avgpool':
            polling = torch.nn.AvgPool2d(self.red_param)
            X_train = polling(torch.tensor(X_train.reshape((len(X_train), pixels, pixels)))).reshape(len(X_train), -1)

        elif self.compression_type == 'Maxpool':
            polling = torch.nn.MaxPool2d(self.red_param)
            X_train = polling(torch.tensor(X_train.reshape((len(X_train), pixels, pixels)))).reshape(len(X_train), -1)

        elif self.compression_type == 'resize':
            size = pixels // self.red_param
            length = len(X_train)
            X_train = torch.tensor(X_train.reshape(length, pixels, pixels))
            X_train = X_train[..., tf.newaxis]
            X_train = tf.image.resize(X_train, [size, size]).numpy().reshape(length, -1)

        elif self.compression_type == 'PCA':
            size = pixels // self.red_param
            if train:
                self.pca_model = PCA(n_components=size ** 2)
                X_train = self.pca_model.fit_transform(X_train)
            else:
                X_train = self.pca_model.transform(X_train.reshape(1, -1))
        elif self.compression_type == 'randpix':
            size = (pixels // self.red_param) ** 2
            string = np.random.randint(0, 255, size=size)
            X_train = X_train[:, string]

        elif self.compression_type == 'randset':
            if train:
                length = len(X_train)
                size = (length // (self.red_param ** 2))
                string = np.random.randint(0, length, size=size)
                X_train = X_train[string, :]
                y_train = y_train[string]
        else:
            print(f'Theres no {self.compression_type} compression method')

        return X_train, y_train


# Geometric calculations
class Stability_space():
    """
    Class purpose is to compute the geometric values for the input X.
    """

    def __init__(self, X_train, y_train, compression=None, metric='l2'):
        """
        Computes the NN for the different permutations of subset labels, and this will be our search space.
        @param X_train: ndarray of flattened images.
        @param y_train: ndarray of labels
        @param compression: Compression class
        @param metric: the narm of distance metric
        """
        self.metric = metric
        self.num_labels = len(set(y_train))
        self.same_nbrs = []
        self.other_nbrs = []

        self.compression = compression

        if self.compression:
            X_train, y_train = compression(X_train, y_train)

        for i in range(self.num_labels):
            idx_other = np.where(y_train != i)
            self.other_nbrs.append(NearestNeighbors(n_neighbors=1, metric=self.metric).fit(X_train[idx_other]))
            idx_same = np.where(y_train == i)
            self.same_nbrs.append(NearestNeighbors(n_neighbors=1, metric=self.metric).fit(X_train[idx_same]))

    def calc_stab(self, X_test, y_test_pred):
        """
        Calculating geometric values for the test/train set.
        @param X_test: ndarray of flattened images.
        @param y_test_pred: ndarray of predicted test labels.
        @return: vector of geometric values.
        """

        if self.compression:
            X_test, _ = self.compression(X_test, None, train=False)

        stability = np.array([-1.] * X_test.shape[0])

        for i in tqdm(range(X_test.shape[0])):
            x = np.array(X_test[i])
            pred_label = y_test_pred[i]

            dist1, idx1 = self.same_nbrs[pred_label].kneighbors([x])
            dist2, idx2 = self.other_nbrs[pred_label].kneighbors([x])

            stability[i] = (dist2 - dist1) / 2
        return stability


# calibrators
class GeometricCalibrator():
    """
    Class popose to be a wrapper of our whole geometric calibration method
    """
    def __init__(self, model, X_train, y_train, comprasion_mode=None, compression_param=None, metric='l2'):
        """
        prepare the search space plus the compression mode.
        @param model: model that uses api of 'predict_proba' / 'predict' / 'fit' (basicly such sklearn).
        @param X_train: ndarray of flattened images.
        @param y_train: ndarray of labels.
        @param comprasion_mode: string of comprasion_mode .
        @param compression_param: int of comprassion (the higher the number the higher the compression).
        @param metric: string of the norm of distance.
        """
        self.Iso = None
        self._fitted = False
        self.model = model
        self.num_labels = len(set(y_train))
        compression = None
        # Compression
        if comprasion_mode and compression_param:
            compression = Compression(compression_type=comprasion_mode, comprassion_param=compression_param)

        # Prepare the stability space (Fast separation)
        self.stab_space = Stability_space(X_train=X_train, y_train=y_train, compression=compression, metric=metric)

    def fit(self, X_val, y_val):
        """
        fit the calibrator
        @param X_val: ndarray of flattened images.
        @param y_val: ndarray of labels.
        @return: None
        """
        # compression
        y_pred_val = self.model.predict(X_val)
        correct = y_val == y_pred_val
        stability_val = self.stab_space.calc_stab(X_val, y_pred_val)
        self.Iso = IsotonicRegression(out_of_bounds="clip").fit(stability_val, correct)
        self._fitted = True

    def calibrate(self, X_test):
        """
        Calibrate the desired input.
        @param X_test: ndarray of flatten images.
        @return: calibrated probability vector per image.
        """
        if self._fitted:
            y_test_pred = self.model.predict(X_test)
            stability_test = self.stab_space.calc_stab(X_test, y_test_pred)
            calibrated_probs = self.Iso.predict(stability_test)
            return calibrated_probs
        else:
            raise ValueError('You must fit you calibrator first')


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

    # Fast separation calibration
    GeoCalibrator = GeometricCalibrator(model, X_train, y_train)
    GeoCalibrator.fit(X_val, y_val)
    calibrated_prob_Geo = GeoCalibrator.calibrate(X_test)

    # Fast separation calibration -compressed
    GeoCalibrator_compressed = GeometricCalibrator(model, X_train, y_train, comprasion_mode='Maxpool',
                                                   compression_param=2)
    GeoCalibrator_compressed.fit(X_val, y_val)
    calibrated_prob_GeoCompressed = GeoCalibrator_compressed.calibrate(X_test)

    # After Calibration
    print(f'Geometric Calibration Fast separation ECE: \t{(ECE_calc(calibrated_prob_Geo, y_pred_test, y_test)):.4f}')
    print(f'Geometric Calibration Fast separation ECE: \t{(ECE_calc(calibrated_prob_GeoCompressed, y_pred_test, y_test)):.4f}')
    print(f'No Calibration ECE: \t{(ECE_calc(y_test_probs, y_pred_test, y_test)):.4f}')

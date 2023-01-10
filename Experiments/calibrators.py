import warnings
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn, optim
from other_calibrations.calibration_HB import HB_toplabel
import calibration as SBC
from scipy.optimize import curve_fit
from netcal.scaling import BetaCalibration
from netcal.binning import BBQ


# This file implements various calibration methods.

def sigmoid_func(x, x0, k):
    return 1. / (1. + np.exp(-k * (x - x0)))


class BaseCalibrator:
    """ Abstract calibrator class
    """

    def __init__(self):
        self.n_classes = None

    def fit(self, logits, y):
        raise NotImplementedError

    def calibrate(self, probs):
        raise NotImplementedError

    def ECE(self, probs_precalibrated, y_pred, y_real, bins=15):
        '''
        params :
                    probs - vector of the toplabel probabilities
                    y_pred - predicted y by model
                    y_real - real label
                    bins - bins calculated on
        return :
                    ECE - expected calibration error

        '''
        probs = self.calibrate(probs_precalibrated)

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


class IdentityCalibrator(BaseCalibrator):
    """ A class that implements no recalibration.
    """

    def fit(self, probs_val, y_true):
        return

    def calibrate(self, probs):
        return probs


class IsotonicCalibrator(BaseCalibrator):
    def calibrate(self, probs):
        if len(probs.shape) == 2:
            probs = np.max(probs, axis=1)
        output = self.Isotonic.predict(probs)
        return output

    def fit(self, y_prob_val, y_true):
        if len(y_prob_val.shape) == 2:
            y_prob_val = np.max(y_prob_val, axis=1)
        self.Isotonic = IsotonicRegression(out_of_bounds='clip',
                                           y_min=0,
                                           y_max=1)
        self.Isotonic.fit(y_prob_val, y_true)


class PlattCalibrator(BaseCalibrator):
    def calibrate(self, probs):
        if len(probs.shape) == 2:
            probs = np.max(probs, axis=1)
        output = sigmoid_func(probs, *self.popt_sigmoid)
        # output = self.logistic.predict(probs.reshape(-1, 1))
        return output

    def fit(self, y_prob_val, y_true):

        if len(y_prob_val.shape) == 2:
            y_prob_val = np.max(y_prob_val, axis=1)
        # the class expects 2d ndarray as input features
        p0 = [max(y_prob_val), min(y_prob_val)]
        self.popt_sigmoid, _ = curve_fit(sigmoid_func, y_prob_val, y_true, p0, maxfev=1000000)


#         self.logistic = LogisticRegression(C=1e10)
#         self.logistic.fit(y_prob_val.reshape(-1, 1), y_true)


class TSCalibrator(BaseCalibrator):
    """ Maximum likelihood temperature scaling (Guo et al., 2017)
    implemented by https://github.com/GavinKerrigan/conf_matrix_and_calibration/tree/80ad706280264b891611d5bcb76476f288ef89cc
    """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature

        self.loss_trace = None

    def fit(self, logits, y):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        self.n_classes = logits.shape[1]
        _model_logits = torch.from_numpy(logits)
        _y = torch.from_numpy(y)
        _temperature = torch.tensor(self.temperature, requires_grad=True)

        # Optimization parameters
        nll = nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 7500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            loss = nll(_model_logits / _temperature, _y)
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())

            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)

            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        self.temperature = _temperature.item()

    def calibrate(self, probs):
        calibrated_probs = probs ** (1. / self.temperature)  # Temper
        calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)  # Normalize
        return calibrated_probs


class StabilityCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.popt = None

    def calibrate(self, stability_test):
        calibrated_probs = self.popt[0].predict(stability_test).reshape(-1, 1)
        return calibrated_probs

    def fit(self, stab_valildation, y_true_valildation):
        from utils import fitting_function
        self.popt = fitting_function(stab_valildation, y_true_valildation)


class SeparationCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.popt = None

    def calibrate(self, sep_test):
        calibrated_probs = self.popt[0].predict(sep_test).reshape(-1, 1)
        return calibrated_probs

    def fit(self, sep_valildationl, y_true_valildation):
        from utils import fitting_function
        self.popt = fitting_function(sep_valildationl, y_true_valildation)


class EnsembleTSCalibrator(BaseCalibrator):
    """ Ensemble Temperature Scaling (Zhang et al., 2020)
    implemented by https://github.com/GavinKerrigan/conf_matrix_and_calibration/tree/80ad706280264b891611d5bcb76476f288ef89cc
    """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.weights = None

    def calibrate(self, probs):
        p1 = probs
        tempered_probs = probs ** (1. / self.temperature)  # Temper
        tempered_probs /= np.sum(tempered_probs, axis=1, keepdims=True)  # Normalize
        p0 = tempered_probs
        p2 = np.ones_like(p0) / self.n_classes

        calibrated_probs = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2

        return calibrated_probs

    def fit(self, logits, y):
        from other_calibrations.ensemble_ts import ets_calibrate
        self.n_classes = logits.shape[1]

        # labels need to be one-hot for ETS
        _y = np.eye(self.n_classes)[y]

        t, w = ets_calibrate(logits, _y, self.n_classes, loss='mse')  # loss = 'ce'
        self.temperature = t
        self.weights = w


class SBCCalibrator(BaseCalibrator):
    '''
    implemented by:
    https://github.com/p-lambda/verified_calibration
    '''

    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins

    def calibrate(self, probs):
        SBC_probs_test = self.calibrator.calibrate(probs)
        return SBC_probs_test

    def fit(self, val_proba, y_val):
        self.calibrator = SBC.PlattBinnerMarginalCalibrator(len(val_proba), num_bins=self.bins)
        self.calibrator.train_calibration(val_proba, y_val)


class SBC_TOP_Calibrator(BaseCalibrator):
    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins

    def calibrate(self, probs):
        top_probs = self._platt(SBC.utils.get_top_probs(probs))
        return self._discrete_calibrator(top_probs)

    def fit(self, top_probs, correct):
        if len(top_probs.shape) == 2:
            top_probs = np.squeeze(top_probs)
        self._num_calibration = len(top_probs)
        self._platt = SBC.utils.get_platt_scaler(top_probs, correct)
        platt_probs = self._platt(top_probs)
        bins = SBC.utils.get_equal_bins(platt_probs, num_bins=self.bins)
        self._discrete_calibrator = SBC.utils.get_discrete_calibrator(platt_probs, bins)


class stab_SBC_Calibrator(BaseCalibrator):
    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins
        self.stab_cali = StabilityCalibrator()
        self.SBCTOP_cali = SBC_TOP_Calibrator()

    def calibrate(self, stab_test):
        first_cali_probs = self.stab_cali.calibrate(stab_test)
        second_cali_probs = self.SBCTOP_cali.calibrate(first_cali_probs)
        return second_cali_probs

    def fit(self, stab_val, val_probs, corrects):
        top_probs = np.max(val_probs, axis=1)
        self.stab_cali.fit(stab_val, corrects)
        calibrated_val_probs = self.stab_cali.calibrate(stab_val)
        self.SBCTOP_cali.fit(calibrated_val_probs, corrects)


class HBCalibrator(BaseCalibrator):
    '''
    implemented by:
    https://github.com/aigen/df-posthoc-calibration
    '''

    def __init__(self, bins=50):
        super().__init__()
        self.bins = bins

    def calibrate(self, probs):
        HB_probs_test = self.calibrator.predict_proba(probs)
        return HB_probs_test

    def fit(self, val_proba, y_val):
        self.calibrator = HB_toplabel(points_per_bin=self.bins)
        self.calibrator.fit(val_proba, y_val)


class BBQCalibrator(BaseCalibrator):
    '''
    implemented by :
    https://github.com/EFS-OpenSource/calibration-framework
    '''

    def __init__(self, bins=50):
        super().__init__()
        self.bins = bins

    def calibrate(self, probs):
        cali_probs = self.calibrator.transform(probs)
        return cali_probs

    def fit(self, val_proba, y_val):
        self.calibrator = BBQ()
        self.calibrator.fit(val_proba, y_val)


class BetaCalibrator(BaseCalibrator):
    '''
    implemented by https://github.com/EFS-OpenSource/calibration-framework
    '''

    def __init__(self, bins=50):
        super().__init__()
        self.bins = bins

    def calibrate(self, probs):
        cali_probs = self.calibrator.transform(probs)
        return cali_probs

    def fit(self, val_proba, y_val):
        self.calibrator = BetaCalibration()
        self.calibrator.fit(val_proba, y_val)


class StabilityHistogramBinningCalibrator(BaseCalibrator):
    '''
    Composition of stability with HB.
    '''
    def __init__(self, num_bins=50):
        super().__init__()
        self.popt = None
        self.num_bins = num_bins

    def calibrate(self, stability_test):
        from utils import get_bin
        test_probs = self.popt[0].predict(stability_test).reshape(-1, 1)

        calibrated_probs = []
        for prob in test_probs:
            bin_idx = get_bin(prob, self.bins_ranges)
            if bin_idx > self.num_bins - 1:
                calibrated_probs.append(self.bin_means[bin_idx - 1])
            else:
                calibrated_probs.append(self.bin_means[bin_idx])

        return calibrated_probs

    def fit(self, stab_valildation, y_true_valildation):
        from utils import fitting_function, histogramBinning
        # fit an regression model
        self.popt = fitting_function(stab_valildation, y_true_valildation)

        # predictions of the regression on the same problem
        probs = self.popt[0].predict(stab_valildation).reshape(-1, 1)

        # histogram binning
        self.bin_means, self.bins_ranges, self.new_ranges = histogramBinning(probs, y_true_valildation, self.num_bins)


class SeparationHistogramBinningCalibrator(StabilityHistogramBinningCalibrator):
    pass


class SKlearn_calibrator():
    def __init__(self, data, method, orig_model):
        from Data import Data
        if method in ['sigmoid', 'isotonic'] and isinstance(data, Data):
            self.data = data
            self.method = method
            self.orig_model = orig_model
            self.calibrated_model = None
        else:
            raise ValueError("wrong initialization")

    def fit(self):
        from sklearn.calibration import CalibratedClassifierCV
        self.calibrated_model = CalibratedClassifierCV(base_estimator=self.orig_model, cv="prefit", method=self.method)
        self.calibrated_model.fit(self.data.X_val, self.data.y_val)
        return self

    def ECE(self):
        "ECE on the test set"
        from utils import ECE_calc
        pred_y_test_cal = self.calibrated_model.predict(self.data.X_test)
        probs_cal = self.calibrated_model.predict_proba(self.data.X_test)
        return ECE_calc(probs_cal, pred_y_test_cal, self.data.y_test)

    def __repr__(self):
        return f'{self.__name__}_{self.method}'






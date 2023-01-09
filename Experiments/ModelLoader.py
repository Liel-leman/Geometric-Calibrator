from Data import *
from calibrators import *
import pickle
from utils import hot_padding


class ModelLoader():
    def __init__(self, dataset_name, shuffle_num, model_name, isCalibrate=False, Norm='L1'):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.shuffle_num = shuffle_num

        self.data = load_data(dataset_name, shuffle_num)

        calc_dir = f'{dataset_name}/{shuffle_num}/{model_name}/'

        adder = "_calibrated" if isCalibrate else ""

        self.all_predictions_val = np.load(calc_dir + f'all_predictions_val{adder}.npy', allow_pickle=True)
        self.all_predictions_test = np.load(calc_dir + f'all_predictions_test{adder}.npy', allow_pickle=True)
        # self.all_predictions_train = np.load(calc_dir + f'all_predictions_train{adder}.npy', allow_pickle=True)
        self.y_pred_test = np.load(calc_dir + f'y_pred_test{adder}.npy', allow_pickle=True)
        self.y_pred_val = np.load(calc_dir + f'y_pred_val{adder}.npy', allow_pickle=True)
        # self.y_pred_train = np.load(calc_dir + f'y_pred_train{adder}.npy', allow_pickle=True)

        self.isCalibrate = isCalibrate

        # adder of norm:
        if Norm in ['L1', 'Linf', 'L2']:
            Norm = '_' + Norm
        else:
            raise ValueError(f'{Norm} not supported')

        try:
            # self.stability_test = np.load(calc_dir + f'/stability_test{adder}.npy', allow_pickle=True)
            # self.stability_val = np.load(calc_dir + f'/stability_val{adder}.npy', allow_pickle=True)
            self.stability_test = np.load(calc_dir + f'stability_test{Norm}.npy', allow_pickle=True)
            self.stability_val = np.load(calc_dir + f'stability_val{Norm}.npy', allow_pickle=True)
        except:
            print('couldnt load stability calculations')
            self.stability_test = None
            self.stability_val = None

        try:
            self.sep_test = np.load(calc_dir + f'/sep_test{Norm}.npy', allow_pickle=True)
            self.sep_val = np.load(calc_dir + f'/sep_val{Norm}.npy', allow_pickle=True)
        except:
            print('couldnt load saperation calculations')
            self.sep_test = None
            self.sep_val = None

        # for pytorch load of logits
        try:
            self.logits_test = np.load(calc_dir + f'/logits_test.npy', allow_pickle=True)
            self.logits_val = np.load(calc_dir + f'/logits_val.npy', allow_pickle=True)
            # self.logits_train = np.load(calc_dir + f'/logits_train.npy', allow_pickle=True)
        except:
            self.logits_test = None
            self.logits_val = None
            self.logits_train = None

    def compute_error_metric(self, method, err_Func, bins=15):
        if self.model_name in ['RF', 'GB']:
            model_dir = f'{self.dataset_name}/{self.shuffle_num}/model/model_{self.dataset_name}_{self.model_name}.sav'
            model = pickle.load(open(model_dir, 'rb'))
            allowed = ['Base', 'StabilityCalibrator', 'SeparationCalibrator', 'HBCalibrator', 'SBCCalibrator',
                       'SKlearn_calibrator_platt', 'SKlearn_calibrator_isotonic', 'IsotonicCalibrator',
                       'PlattCalibrator']

        elif self.model_name == 'pytorch':
            allowed = ['Base', 'StabilityCalibrator', 'SeparationCalibrator', 'HBCalibrator', 'SBCCalibrator',
                       'IsotonicCalibrator', 'PlattCalibrator', 'TSCalibrator',
                       'EnsembleTSCalibrator']

        allowed.extend(
            ['stab->SBC', 'stab->HB', 'StabilityHistogramBinningCalibrator', 'SeparationHistogramBinningCalibrator'])

        if method in allowed:
            if method == 'Base':
                # Base
                probs_calibrated = self.all_predictions_test
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'SKlearn_calibrator_isotonic':
                calibrator = SKlearn_calibrator(self.data, 'isotonic', model).fit()
                probs_calibrated = calibrator.calibrated_model.predict_proba(self.data.X_test)
                pred_y_test_calibrated = calibrator.calibrated_model.predict(self.data.X_test)

            elif method in ['IsotonicCalibrator', 'PlattCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.all_predictions_val, self.y_pred_val == self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'SKlearn_calibrator_platt':
                calibrator = SKlearn_calibrator(self.data, 'sigmoid', model).fit()
                probs_calibrated = calibrator.calibrated_model.predict_proba(self.data.X_test)
                pred_y_test_calibrated = calibrator.calibrated_model.predict(self.data.X_test)

            elif method in ['StabilityCalibrator', 'StabilityHistogramBinningCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.stability_val, self.y_pred_val == self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.stability_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method in ['SeparationCalibrator', 'SeparationHistogramBinningCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.sep_val, self.y_pred_val == self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.sep_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'HBCalibrator':
                calibrator = HBCalibrator()
                calibrator.fit(self.all_predictions_val, self.data.y_val + 1)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'SBCCalibrator':
                calibrator = SBCCalibrator()
                calibrator.fit(self.all_predictions_val, self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)

            elif method in ['EnsembleTSCalibrator', 'TSCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.logits_val, self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)

            elif method == 'stab->SBC':

                calibrator = StabilityCalibrator()
                calibrator.fit(self.stability_val, self.y_pred_val == self.data.y_val)

                stab_val_probs = calibrator.calibrate(self.stability_val)
                stab_test_probs = calibrator.calibrate(self.stability_test)
                stab_val_probs = hot_padding(stab_val_probs, self.y_pred_val, self.data.num_labels)
                stab_test_probs = hot_padding(stab_test_probs, self.y_pred_test, self.data.num_labels)

                calibrator = SBCCalibrator()
                calibrator.fit(stab_val_probs, self.data.y_val)

                probs_calibrated = calibrator.calibrate(stab_test_probs)
                pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)

            elif method == 'stab->HB':

                calibrator = StabilityCalibrator()
                calibrator.fit(self.stability_val, self.y_pred_val == self.data.y_val)

                stab_val_probs = calibrator.calibrate(self.stability_val)
                stab_test_probs = calibrator.calibrate(self.stability_test)
                stab_val_probs = hot_padding(stab_val_probs, self.y_pred_val, self.data.num_labels)
                stab_test_probs = hot_padding(stab_test_probs, self.y_pred_test, self.data.num_labels)

                calibrator = HBCalibrator()
                calibrator.fit(stab_val_probs, self.data.y_val + 1)

                probs_calibrated = calibrator.calibrate(stab_test_probs)
                pred_y_test_calibrated = self.y_pred_test

        else:
            raise ValueError(f'{method} Method do not exist')

        return err_Func(probs_calibrated, pred_y_test_calibrated, self.data.y_test, bins)

    def __repr__(self):
        return '(' + self.dataset_name + '-' + self.model_name + '-' + str(self.shuffle_num) + ')'


class Tabular_loader():
    def __init__(self, dataset_name, model_name, shuffle_num, all_predictions_val, all_predictions_test, y_pred_test,
                 y_pred_val, stability_test, stability_val, data, model):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.shuffle_num = shuffle_num
        self.all_predictions_val = all_predictions_val
        self.all_predictions_test = all_predictions_test
        self.y_pred_test = y_pred_test
        self.y_pred_val = y_pred_val
        self.stability_test = stability_test
        self.stability_val = stability_val
        self.data = data
        self.model = model

    def compute_error_metric(self, method, err_Func, bins=15):
        if method in ['Base', 'StabilityCalibrator', 'SeparationCalibrator', 'HBCalibrator', 'SBCCalibrator',
                      'SKlearn_calibrator_platt', 'SKlearn_calibrator_isotonic', 'IsotonicCalibrator',
                      'PlattCalibrator', 'StabilityHistogramBinningCalibrator', 'SeparationHistogramBinningCalibrator']:
            if method == 'Base':
                # Base
                probs_calibrated = self.all_predictions_test
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'SKlearn_calibrator_isotonic':
                calibrator = SKlearn_calibrator(self.data, 'isotonic', self.model).fit()

                probs_calibrated = calibrator.calibrated_model.predict_proba(self.data.X_test)
                pred_y_test_calibrated = calibrator.calibrated_model.predict(self.data.X_test)

            elif method in ['IsotonicCalibrator', 'PlattCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.all_predictions_val, self.y_pred_val == self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'SKlearn_calibrator_platt':
                calibrator = SKlearn_calibrator(self.data, 'sigmoid', self.model).fit()

                probs_calibrated = calibrator.calibrated_model.predict_proba(self.data.X_test)
                pred_y_test_calibrated = calibrator.calibrated_model.predict(self.data.X_test)

            elif method in ['StabilityCalibrator', 'StabilityHistogramBinningCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.stability_val, self.y_pred_val == self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.stability_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method in ['SeparationCalibrator', 'SeparationHistogramBinningCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.sep_val, self.y_pred_val == self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.sep_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'HBCalibrator':
                calibrator = HBCalibrator()
                calibrator.fit(self.all_predictions_val, self.data.y_val + 1)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = self.y_pred_test

            elif method == 'SBCCalibrator':
                calibrator = SBCCalibrator()
                calibrator.fit(self.all_predictions_val, self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)

            elif method in ['EnsembleTSCalibrator', 'TSCalibrator']:
                calibrator = eval(method)()
                calibrator.fit(self.logits_val, self.data.y_val)

                probs_calibrated = calibrator.calibrate(self.all_predictions_test)
                pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)

            elif method == 'stab->SBC':

                calibrator = StabilityCalibrator()
                calibrator.fit(self.stability_val, self.y_pred_val == self.data.y_val)

                stab_val_probs = calibrator.calibrate(self.stability_val)
                stab_test_probs = calibrator.calibrate(self.stability_test)
                stab_val_probs = hot_padding(stab_val_probs, self.y_pred_val, self.data.num_labels)
                stab_test_probs = hot_padding(stab_test_probs, self.y_pred_test, self.data.num_labels)

                calibrator = SBCCalibrator()
                calibrator.fit(stab_val_probs, self.data.y_val)

                probs_calibrated = calibrator.calibrate(stab_test_probs)
                pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)

            elif method == 'stab->HB':

                calibrator = StabilityCalibrator()
                calibrator.fit(self.stability_val, self.y_pred_val == self.data.y_val)

                stab_val_probs = calibrator.calibrate(self.stability_val)
                stab_test_probs = calibrator.calibrate(self.stability_test)
                stab_val_probs = hot_padding(stab_val_probs, self.y_pred_val, self.data.num_labels)
                stab_test_probs = hot_padding(stab_test_probs, self.y_pred_test, self.data.num_labels)

                calibrator = HBCalibrator()
                calibrator.fit(stab_val_probs, self.data.y_val + 1)

                probs_calibrated = calibrator.calibrate(stab_test_probs)
                pred_y_test_calibrated = self.y_pred_test

        else:
            raise ValueError(f'{method} Method do not exist')

        return err_Func(probs_calibrated, pred_y_test_calibrated, self.data.y_test, bins)

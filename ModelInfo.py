from other_calibrations.calibration_HB import *
# pip3 install uncertainty-calibration for HBC calibration (calibration package)
import calibration as SBC



class ModelInfo():
    def __init__(self, data, y_pred_val=None, all_predictions_val=None, y_pred_test=None, all_predictions_test=None,
                 y_pred_train=None, all_predictions_train=None, isCalibrate=None):
        self.data = data
        self.y_pred_val = y_pred_val
        self.all_predictions_val = all_predictions_val
        self.y_pred_test = y_pred_test
        self.all_predictions_test = all_predictions_test
        self.y_pred_train = y_pred_train
        self.all_predictions_train = all_predictions_train
        self.isCalibrate = isCalibrate
        self.stability_test = None
        self.stability_val = None
        self.sep_test = None
        self.sep_val = None

    def compute_error_metric(self, method, err_Func , bins=30):
        '''
        :param method: ['sep', 'stability', 'SBC', 'HB', 'sklearn', 'Base']
        :param err_Func: ECE-function
        :param bins: number of bins in the error metric, our default is '30'
        :return: err
        '''

        from utils import fitting_function

        allowed = ['sep', 'stability', 'SBC', 'HB', 'sklearn', 'Base']

        if method not in allowed:
            raise ValueError(f'Wrong method , need to be one of {allowed}')

        if method in ['sklearn', 'Base']:
            prob = np.max(self.all_predictions_test, axis=1)

        elif method in ['sep', 'stability']:
            # fitting function on validation:
            popt = fitting_function(eval(f'self.{method}_val'), self.data.y_val, self.y_pred_val)  # [isotonic_regression, popt_stab_sigmoid]

            # get probs of sigmoid and isotonic regression:
            iso_probs = popt[0].predict(eval(f'self.{method}_test').reshape(-1, 1))

            prob = iso_probs  # new with iso with 'clip'


        elif method == 'HB':
            hb = HB_toplabel(points_per_bin=bins)
            hb.fit(self.all_predictions_val, self.data.y_val + 1)
            prob = hb.predict_proba(self.all_predictions_test)

        elif method == 'SBC':
            appended_y = np.append(self.data.y_train, self.data.y_val)
            appended_predictions = np.append(self.all_predictions_train, self.all_predictions_val, axis=0)

            calibrator = SBC.PlattBinnerMarginalCalibrator(len(appended_predictions), num_bins=bins)
            calibrator.train_calibration(appended_predictions, appended_y)

            SBC_probs_test = calibrator.calibrate(self.all_predictions_test)
            SBC_probs_y = [np.argmax(i) for i in SBC_probs_test]

            err = err_Func(SBC_probs_test, SBC_probs_y, self.data.y_test)
            #             print(f'{error_metric}:{err} |||| calibration method:{method}')
            #             plot_calibration_stairs(np.max(SBC_probs_test, axis=1),SBC_probs_y,self.data.y_test,dataset_name,model_name,method)

            return err

        err = err_Func(prob, self.y_pred_test, self.data.y_test, bins)
        #         print(f'{error_metric}:{err} |||| calibration method:{method}')
        #         plot_calibration_stairs(prob,self.y_pred_test,self.data.y_test,dataset_name,model_name,method)

        return err


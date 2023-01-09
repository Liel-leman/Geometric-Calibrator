
from sklearn.model_selection import train_test_split

import pandas as pd
from calibrators import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from utils import ECE_calc
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from utils import stability_calc,sep_calc
import sys
import pickle

def transform_gender(x):
    if x == 'Female':
        return 1
    elif x == 'Male':
        return 0
    else:
        return -1


def transform_customer_type(x):
    if x == 'Loyal Customer':
        return 1
    elif x == 'disloyal Customer':
        return 0
    else:
        return -1


def transform_travel_type(x):
    if x == 'Business travel':
        return 1
    elif x == 'Personal Travel':
        return 0
    else:
        return -1


def transform_class(x):
    if x == 'Business':
        return 2
    elif x == 'Eco Plus':
        return 1
    elif x == 'Eco':
        return 0
    else:
        return -1


def transform_satisfaction(x):
    if x == 'satisfied':
        return 1
    elif x == 'neutral or dissatisfied':
        return 0
    else:
        return -1


def process_data(df):
    df = df.drop(['Unnamed: 0', 'id'], axis=1)
    df['Gender'] = df['Gender'].apply(transform_gender)
    df['Customer Type'] = df['Customer Type'].apply(transform_customer_type)
    df['Type of Travel'] = df['Type of Travel'].apply(transform_travel_type)
    df['Class'] = df['Class'].apply(transform_class)
    df['satisfaction'] = df['satisfaction'].apply(transform_satisfaction)
    df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)

    return df

def ece_shuffle(shuffle_num,norm='l2'):
    dataset_name = 'AIRLINE'
    print(f'Computing {dataset_name}-{shuffle_num} ECE')
    models = [RandomForestClassifier(), GradientBoostingClassifier(),
              xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)]
    model_names = ['RF', 'GB', 'XGB']
    methods = ['Stab', 'SKlearn_platt', 'SKlearn_iso', 'HB', 'SBC', 'accuracy']

    d = {f'{dataset_name}-{model}': {method: [] for method in methods} for model in model_names}


    train = pd.read_csv('./AIRLINE/train.csv')
    test = pd.read_csv('./AIRLINE/test.csv')
    data = pd.concat([train, test])
    data = process_data(data)
    X = data.drop(['satisfaction'], axis=1)
    y = data['satisfaction'].to_numpy()

    X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, test_size=0.2, random_state=shuffle_num)
    X_train, X_val, y_train, y_val = train_test_split(X_train_v, y_train_v, test_size=0.25, random_state=shuffle_num)
    scaler = RobustScaler()
    x_train = scaler.fit_transform(X_train)
    x_val = scaler.transform(X_val)
    x_test = scaler.transform(X_test)
    num_labels = len(np.unique(y_train))

    for model, model_name in zip(models, model_names):
        clf = model
        clf.fit(x_train, y_train)

        probs_test = clf.predict_proba(x_test)
        y_pred_test = clf.predict(x_test)
        probs_val = clf.predict_proba(x_val)
        y_pred_val = clf.predict(x_val)

        acc = accuracy_score(y_test, y_pred_test)

        corrects = y_pred_val == y_val

        stab_test = stability_calc(x_train, x_test, y_train, y_pred_test, num_labels, metric=norm)
        stab_val = stability_calc(x_train, x_val, y_train, y_pred_val, num_labels, metric=norm)

        # sep_val = sep_calc(x_train, x_val, y_train, y_pred_val, norm = 'L1')
        # sep_test = sep_calc(x_train, x_test, y_train, y_pred_test, norm = 'L1')
        d[f'{dataset_name}-{model_name}']['accuracy'].append(acc)
        # calibration Stage:
        ########################################################################################################
        # Stability Calibration
        stabCal = StabilityCalibrator()
        stabCal.fit(stab_val, corrects)
        ECE_Stab = stabCal.ECE(stab_test, y_pred_test, y_test)
        d[f'{dataset_name}-{model_name}']['Stab'].append(ECE_Stab)

        # #Separation Calibration
        # sepCal = SeparationCalibrator()
        # sepCal.fit(sep_val, corrects)
        # ECE_Sep = sepCal.ECE(sep_test, y_pred_test, y_test)
        # d[f'WINE-{model_name}']['Sep'].append(ECE_Sep)
        ########################################################################################################

        # SKlearn-isotonic
        sklearn_platt = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method='isotonic')
        sklearn_platt.fit(x_val, y_val)
        calibrated_probs_test = sklearn_platt.predict_proba(x_test)
        calibrated_y_pred_test = sklearn_platt.predict(x_test)
        ECE_iso = ECE_calc(calibrated_probs_test, calibrated_y_pred_test, y_test)
        d[f'{dataset_name}-{model_name}']['SKlearn_iso'].append(ECE_iso)

        # SKlearn-platt
        sklearn_platt = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method='sigmoid')
        sklearn_platt.fit(x_val, y_val)
        calibrated_probs_test = sklearn_platt.predict_proba(x_test)
        calibrated_y_pred_test = sklearn_platt.predict(x_test)
        ECE_platt = ECE_calc(calibrated_probs_test, calibrated_y_pred_test, y_test)
        d[f'{dataset_name}-{model_name}']['SKlearn_platt'].append(ECE_platt)

        ########################################################################################################

        # HB
        HBcalibrator = HBCalibrator()
        HBcalibrator.fit(probs_val, y_val + 1)
        HB_test_calibrated = HBcalibrator.calibrate(probs_test)
        ECE_HB = ECE_calc(HB_test_calibrated, y_pred_test, y_test)
        d[f'{dataset_name}-{model_name}']['HB'].append(ECE_HB)

        # #stab+HB(our implementation)
        # stabHB = StabilityHistogramBinningCalibrator()
        # stabHB.fit(stab_val,corrects)
        # stabHB_test_calibrated =stabHB.calibrate(stab_test)
        # ECE_stabHB = ECE_calc(stabHB_test_calibrated, y_pred_test, y_test)
        # d[f'WINE-{model_name}']['Stab+HB'].append(ECE_stabHB)

        # Sep+HB
        # sepHB = SeparationHistogramBinningCalibrator()
        # sepHB.fit(sep_val,corrects)
        # sepHB_test_calibrated =sepHB.calibrate(sep_test)
        # ECE_sepHB = ECE_calc(sepHB_test_calibrated, y_pred_test, y_test)
        # d[f'WINE-{model_name}']['Sep+HB'].append(ECE_sepHB)

        ########################################################################################################

        # SBC - making problems when we dont have all labels of Dataset in y_val
        try:
            # SBC
            SBCcalibrator = SBCCalibrator()
            SBCcalibrator.fit(probs_val, y_val)
            probs_calibrated = SBCcalibrator.calibrate(probs_test)
            pred_y_test_calibrated = np.argmax(probs_calibrated, axis=1)
            ECE_SBC = ECE_calc(probs_calibrated, pred_y_test_calibrated, y_test)
            d[f'{dataset_name}-{model_name}']['SBC'].append(ECE_SBC)

            # stab+SBC
        #             stab_SBCtop_calibrator = stab_SBC_Calibrator()
        #             stab_SBCtop_calibrator.fit(stab_val ,probs_val, corrects)
        #             calibratedTOP_test = stab_SBCtop_calibrator.calibrate(stab_test)
        #             ECE_stabSBC = ECE_calc(calibratedTOP_test, y_pred_test, y_test)
        #             d[f'WINE-{model_name}']['Stab+SBC'].append(ECE_stabSBC)

        #             #Sep+SBC
        #             stab_SBCtop_calibrator = stab_SBC_Calibrator()
        #             stab_SBCtop_calibrator.fit(sep_val ,probs_val, corrects)
        #             calibratedTOP_test = stab_SBCtop_calibrator.calibrate(sep_test)
        #             ECE_sepSBC = ECE_calc(calibratedTOP_test, y_pred_test, y_test)
        #             d[f'WINE-{model_name}']['Sep+SBC'].append(ECE_sepSBC)

        except BaseException as error:
            print('An exception occurred: {}'.format(error))


    print(d)
    with open(f'./saved_calculations/AIRLINE/ece-{dataset_name}-{shuffle_num}.pkl', 'wb') as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2],)
    ece_shuffle(int(sys.argv[1]),  sys.argv[2])
    #                shuffle_num, norm
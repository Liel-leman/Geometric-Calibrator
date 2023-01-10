from utils import *
from ModelLoader import *
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def ece_shuffle(dataset_name, model_name, norm = 'L2'):
    ece = {}
    ece[f'{dataset_name}-{model_name}'] = {}
    
    for shuffle_num in tqdm(range(1)):
        methods = ['StabilityCalibrator', 'SeparationCalibrator', 'HBCalibrator', 'SBCCalibrator', 'BetaCalibrator', 'BBQCalibrator']
        if model_name != 'pytorch':
            methods.extend(['SKlearn_calibrator_isotonic', 'SKlearn_calibrator_platt'])
        else:
            methods.extend(['TSCalibrator', 'EnsembleTSCalibrator', 'IsotonicCalibrator', 'PlattCalibrator'])

        model_loader = ModelLoader(dataset_name, shuffle_num, model_name, norm)

        for method in methods:
            print(f'{dataset_name}-{model_name}-{shuffle_num}-{method}')
            method_ece = model_loader.compute_error_metric(method, ECE_calc)

            if method in ece[f'{dataset_name}-{model_name}']:
                ece[f'{dataset_name}-{model_name}'][method].append(method_ece)
            else:
                ece[f'{dataset_name}-{model_name}'][method] = [method_ece]
                
    with open(f'./saved_calculations/ece-{dataset_name}-{model_name}.pkl', 'wb') as f:
        pickle.dump(ece, f)

    print(ece)

        
        

if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2], sys.argv[3])
    ece_shuffle(sys.argv[1], sys.argv[2], sys.argv[3])
    #                         dataset_name, model_name , norm


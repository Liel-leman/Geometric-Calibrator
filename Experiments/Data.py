from utils import *
from math import sqrt


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


class Data:
    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, num_labels, isRGB=False):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.num_labels = num_labels
        self.isRGB = isRGB

    def compute_stab(self, whom, y_pred, metric='minkowski'):
        '''
        input:
                - seperation for whom ? :
                                        -'test'
                                        -'val'
                                        -'metric': norm
                - y_pred_val : predicted labels of test\val

        return: list of seperations
        '''
        if whom == 'test':
            return stability_calc(self.X_train, self.X_test, self.y_train, y_pred, self.num_labels, metric='minkowski')
        elif whom == 'val':
            return stability_calc(self.X_train, self.X_val, self.y_train, y_pred, self.num_labels, metric='minkowski')
        else:
            print('error')
            return

    def compute_stab_vectored(self, whom, y_pred):
        '''
        input:
                - seperation for whom ? :
                                        -'test'
                                        -'val'
                - y_pred_val : predicted labels of test\val

        return: list of seperations
        '''
        if whom == 'test':
            return stab_calc_vector(self.X_train, self.X_test, self.y_train, y_pred, self.num_labels)
        elif whom == 'val':
            return stab_calc_vector(self.X_train, self.X_val, self.y_train, y_pred, self.num_labels)
        else:
            print('error')
            return

    def compute_sep(self, whom, y_pred):
        '''
        input:
                - whom ? :
                                        -'test'
                                        -'val'
                - y_pred_val : predicted labels of test\val
        return: list of Whole seperations
        '''
        if whom == 'test':
            return sep_calc(self.X_train, self.X_test, self.y_train, y_pred)
        elif whom == 'val':
            return sep_calc(self.X_train, self.X_val, self.y_train, y_pred)
        else:
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

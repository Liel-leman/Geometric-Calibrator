# Import Libraries
import sys
from calibrators import *
from Data import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from scipy.special import softmax
import random
import os
from utils import stability_calc, sep_calc_parallel, normalize_dataset
from SLURM.pytorch_config import *


def preprosses_CIFAR(dataset_name, shuffle_num):
    PATH = f'./{dataset_name}/{shuffle_num}/data/'

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    X_train = np.load(PATH + 'X_train.npy')
    X_test = np.load(PATH + 'X_test.npy')
    X_val = np.load(PATH + 'X_val.npy')
    y_train = np.load(PATH + 'y_train.npy')
    y_test = np.load(PATH + 'y_test.npy')
    y_val = np.load(PATH + 'y_val.npy')

    isRGB = "RGB" in dataset_name
    data = Data(X_train, X_test, X_val, y_train, y_test, y_val, len(set(y_train)), isRGB)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_dataset(X_train.reshape(-1, 32, 32, 3))])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_dataset(X_train.reshape(-1, 32, 32, 3))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_dataset(X_test.reshape(-1, 32, 32, 3))
    ])

    trainset = CIFAR_from_array(data=X_train, label=y_train, transform=train_transform)
    valset = CIFAR_from_array(data=X_val, label=y_val, transform=val_transform)
    testset = CIFAR_from_array(data=X_test, label=y_test, transform=test_transform)

    batch_size = 64

    train_loader = DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=False)

    val_loader = DataLoader(dataset=valset,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, data

def main(dataset_name, shuffle_num):
    train_loader, val_loader, test_loader, data = preprosses_CIFAR(dataset_name, shuffle_num)
    net = CIFAR_net()
    num_epochs = 30
    learning_rate = 0.001

    if torch.cuda.is_available():
        net = net.cuda()
    
    # CrossEntropyLoss - Loss function
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    PATH_model = f'./{dataset_name}/{shuffle_num}/pytorch/'
    # training and validating
    if not os.path.exists(PATH_model + 'model.pth'):
        hist = model_training(num_epochs, net, criterion, train_loader, optimizer, val_loader, verbose=True)
        # model params saving
        if not os.path.exists(PATH_model):
            os.makedirs(PATH_model)
        torch.save(net.state_dict(), PATH_model + 'model.pth')
        np.save(PATH_model + f'hist.npy', hist)
    else:
        print('preload state_dict')
        net.load_state_dict(torch.load(PATH_model + 'model.pth'))

    # calc predictions
    Y_pred_test = model_testing(net, criterion, test_loader, 'test')
    Y_pred_val = model_testing(net, criterion, val_loader, 'val')

    acc_test = accuracy_score(test_loader.dataset.label, Y_pred_test)
    np.save(PATH_model + f'acc_test.npy', acc_test)

    # calc logits
    logits_val = compute_logits(net, val_loader)
    logits_test = compute_logits(net, test_loader)

    # predict proba (after softmax)
    val_proba = softmax(logits_val, axis=1)
    test_proba = softmax(logits_test, axis=1)

    stability_val = stability_calc(data.X_train, data.X_val, data.y_train, Y_pred_val, data.num_labels)
    stability_test = stability_calc(data.X_train, data.X_test, data.y_train, Y_pred_test, data.num_labels)
    np.save(PATH_model + f'stability_test.npy', stability_test)
    np.save(PATH_model + f'stability_val.npy', stability_val)

    print('stability-done')

    PATH_data = f'./{dataset_name}/{shuffle_num}/data/'

    sep_val = sep_calc_parallel(data.X_val, Y_pred_val, PATH_data)
    sep_test = sep_calc_parallel(data.X_test, Y_pred_test, PATH_data)
    np.save(PATH_model + f'sep_test.npy', sep_test)
    np.save(PATH_model + f'sep_val.npy', sep_val)

    print('sep-done')

    ECEs =compute_all_ece(test_proba, Y_pred_test, Y_pred_val, logits_val, val_proba, data, test_loader, val_loader,
                       stability_val, stability_test, sep_test, sep_val)

    np.save(PATH_model + f'ECEs.npy', ECEs)


if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2])
    main(sys.argv[1], sys.argv[2])
    #    dataset_name, model_name

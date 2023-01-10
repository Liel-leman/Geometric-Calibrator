# Import Libraries
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import os
from utils import stability_calc, sep_calc_parallel, normalize_dataset
from pytorch_config import *


def main(dataset_name,shuffle_num, norm='L2'):
    if dataset_name == 'CIFAR_RGB':
        train_loader, val_loader, test_loader, data = preprosses_CIFAR(dataset_name, shuffle_num)
        net = CIFAR_net()

    elif dataset_name == 'MNIST':
        train_loader, val_loader, test_loader, data = preprosses_MNIST(dataset_name, shuffle_num)
        net = MNIST_net()

    elif dataset_name == 'GTSRB_RGB':
        train_loader, val_loader, test_loader, data = preprosses_GTSRB(dataset_name, shuffle_num)
        resn50 = resnet50(pretrained=True, progress=True)
        net = GTSRB_net(resn50)

    elif dataset_name == 'SignLanguage':
        train_loader, val_loader, test_loader, data = preprosses_SignLanguage(dataset_name, shuffle_num)
        net = SignLanguage_net()

    elif dataset_name == 'Fashion':
        train_loader, val_loader, test_loader, data = preprosses_Fashion(dataset_name, shuffle_num)
        net = Fashion_net()

    num_epochs = 30
    learning_rate = 0.001

    if torch.cuda.is_available():
        net = net.cuda()
    print(net)
    print('=================================================================')

    # CrossEntropyLoss - Loss function
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    PATH_model = f'./{dataset_name}/{shuffle_num}/pytorch/'
    # training and validating
    # model_exists = False # to remove
    model_exists = os.path.exists(PATH_model + 'model.pth')
    if not model_exists:
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
    Y_pred_train = model_testing(net, criterion, train_loader, 'train')

    acc_test = accuracy_score(test_loader.dataset.label, Y_pred_test)
    np.save(PATH_model + f'acc_test.npy', acc_test)

    # calc logits
    logits_val = compute_logits(net, val_loader)
    logits_test = compute_logits(net, test_loader)
    logits_train = compute_logits(net, train_loader)

    # predict proba (after softmax)
    all_predictions_val = softmax(logits_val, axis=1)
    all_predictions_test = softmax(logits_test, axis=1)
    all_predictions_train = softmax(logits_train, axis=1)

    L_to_string = {'L1':'manhattan', 'Linf':'chebyshev', 'L2':'euclidean'}
    
    stability_val = stability_calc(data.X_train, data.X_val, data.y_train, Y_pred_val, data.num_labels, L_to_string[norm])
    stability_test = stability_calc(data.X_train, data.X_test, data.y_train, Y_pred_test, data.num_labels, L_to_string[norm])
    print('stability-done')

    PATH_data = f'./{dataset_name}/{shuffle_num}/data/'
    sep_val = sep_calc_parallel(data.X_val, Y_pred_val, PATH_data, norm)
    sep_test = sep_calc_parallel(data.X_test, Y_pred_test, PATH_data, norm)
    print('sep-done')

    np.save(PATH_model + f'Y_pred_test.npy', Y_pred_test)
    np.save(PATH_model + f'Y_pred_val.npy', Y_pred_val)
    np.save(PATH_model + f'Y_pred_train.npy', Y_pred_train)
    np.save(PATH_model + f'logits_val.npy', logits_val)
    np.save(PATH_model + f'logits_test.npy', logits_test)
    np.save(PATH_model + f'logits_train.npy', logits_train)
    np.save(PATH_model + f'all_predictions_val.npy', all_predictions_val)
    np.save(PATH_model + f'all_predictions_test.npy', all_predictions_test)
    np.save(PATH_model + f'all_predictions_train.npy', all_predictions_train)
    np.save(PATH_model + f'stability_test_{norm}.npy', stability_test)
    np.save(PATH_model + f'stability_val_{norm}.npy', stability_val)
    np.save(PATH_model + f'sep_test_{norm}.npy', sep_test)
    np.save(PATH_model + f'sep_val_{norm}.npy', sep_val)

if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2], sys.argv[3])
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    #    dataset_name, model_name , norm

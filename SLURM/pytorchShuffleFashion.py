# Import Libraries
import sys
from calibrators import *
from Data import *
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import random
import os
from utils import stability_calc,sep_calc_parallel
from pytorch_config import *



def preprosses_Fashion(dataset_name, shuffle_num):
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

    train_mean = X_train.mean() / 255.
    train_std = X_train.std() / 255.

    isRGB = "RGB" in dataset_name
    data = Data(X_train, X_test, X_val, y_train, y_test, y_val, len(set(y_train)), isRGB)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])
    # Also use X_train in normalize since train/val sets should have same distribution
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std]),
    ])

    trainset = Fashion_from_array(data=X_train, label=y_train, transform=train_transform)
    valset = Fashion_from_array(data=X_val, label=y_val, transform=val_transform)
    testset = Fashion_from_array(data=X_test, label=y_test, transform=test_transform)

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




def val_per_epoch(model, loss_fn, dataloader, verbose):
    # In validation, we only compute loss value
    model.eval()
    epoch_loss = 0.0
    acc = 0.0
    val_size = 0
    with torch.no_grad():
        for i, (feature, target) in enumerate(dataloader):

            # feature, target = feature.to(device), target.to(device)
            if torch.cuda.is_available():
                feature = feature.cuda()
                target = target.cuda()

            output = model(feature)  # outputs.data.shape= batches_num * num_class

            # compute acc
            _, pred = torch.max(output.data, dim=1)
            correct = (pred == target).sum().item()  # convert to number
            val_size += target.size(0)
            acc += correct

            loss = loss_fn(output, target)
            epoch_loss += loss.item()

            idx = i
            length = len(dataloader)

            # display progress
            if verbose:
                update_info(idx, length, epoch_loss, acc / val_size, 'validating')

        acc = acc / val_size
    print('')
    return epoch_loss / len(dataloader), acc


def update_info(idx, length, epoch_loss, acc, mode):
    if length >= 250:
        update_size = int(length / 250)
    else:
        update_size = 5

    if idx % update_size == 0 and idx != 0:
        # print ('=', end="")
        finish_rate = idx / length * 100
        print("\r   {} progress: {:.2f}%  ......  loss: {:.4f} , acc: {:.4f}".
              format(mode, finish_rate, epoch_loss / idx, acc), end="", flush=True)


def train_per_epoch(model, loss_fn, dataloader, optimizer, verbose):
    # train mode
    model.train()

    # initialize loss
    epoch_loss = 0.0
    acc = 0.0
    train_size = 0

    for i, (feature, target) in enumerate(dataloader):
        if torch.cuda.is_available():
            feature = feature.cuda()
            target = target.cuda()

        # set zero to the parameter gradients for initialization
        optimizer.zero_grad()
        output = model(feature)
        loss = loss_fn(output, target)

        # compute acc
        _, pred = torch.max(output.data, dim=1)
        correct = (pred == target).sum().item()  # convert to number
        train_size += target.size(0)
        acc += correct

        # compute current loss. Loss is a 0-dim tensor, so use tensor.item() to get the scalar value
        epoch_loss += loss.item()

        # backward propagation
        loss.backward()

        # this represents one update on the weight/bias for a mini-batch(16 images in our case):
        # weights[k] + alpha * d_weights[k]
        optimizer.step()

        # show the update information
        idx = i
        length = len(dataloader)

        # display progress
        if verbose:
            update_info(idx, length, epoch_loss, acc / train_size, '  training')

    acc = acc / train_size
    print('')
    return epoch_loss / len(dataloader), acc


def model_training(num_epochs, model, loss_fn, train_loader, optimizer, val_loader=None, verbose=True):
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []
    history['train_acc'] = []
    history['val_acc'] = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        train_loss, train_acc = train_per_epoch(model, loss_fn, train_loader, optimizer, verbose=verbose)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        if val_loader is not None:
            val_loss, val_acc = val_per_epoch(model, loss_fn, val_loader, verbose=verbose)
            print('\n        Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(train_loss, val_loss))
            print('         Training acc: {:.4f},  Validation acc: {:.4f}\n'.format(train_acc, val_acc))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        else:
            print('\n        Training Loss: {:.4f}\n'.format(train_loss))
            print('\n         Training acc: {:.4f}\n'.format(train_acc))

    return history

def compute_logits(net, loader):
    logits_lst = []
    with torch.no_grad():
        for input, label in loader:
            if torch.cuda.is_available():
                input = input.cuda()
            logits = net(input)
            logits_lst.append(logits)
        logits_lst = torch.cat(logits_lst)
    return logits_lst.cpu().detach().numpy()


def model_testing(model, loss_fn, dataloader, s, verbose=True):
    Y_pred = []
    correct = 0
    total = 0
    epoch_loss = 0.0
    acc = 0.0
    test_size = 0
    with torch.no_grad():
        for i, (feature, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                feature = feature.cuda()
                target = target.cuda()

            outputs = model(feature)  # outputs.data.shape= batches_num * num_class

            # compute acc
            _, pred = torch.max(outputs.data, 1)
            correct = (pred == target).sum().item()  # convert to number
            test_size += target.size(0)
            # print(test_size)
            acc += correct

            loss = loss_fn(outputs, target)
            epoch_loss += loss.item()

            idx = i
            length = len(dataloader)

            # if torch.cuda.is_available():
            #    pred = pred.cuda()

            # Pred labels
            Y_pred += pred.cpu().numpy().tolist()

            if verbose:
                update_info(idx, length, epoch_loss, acc / test_size, 'testing')

    acc = acc / test_size
    print(f'\n\n {s} of the network on the {test_size} test images: {100 * acc}%')

    return Y_pred



def main(dataset_name,shuffle_num):
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
   
    sep_val = sep_calc_parallel(data.X_val,Y_pred_val,PATH_data)
    sep_test = sep_calc_parallel(data.X_test,Y_pred_test,PATH_data)
    np.save(PATH_model + f'sep_test.npy', sep_test)
    np.save(PATH_model + f'sep_val.npy', sep_val)
    
    print('sep-done')

    ECEs = compute_all_ece(test_proba, Y_pred_test, Y_pred_val, logits_val, val_proba, data, test_loader, val_loader,
                       stability_val, stability_test, sep_test, sep_val)
    np.save(PATH_model + f'ECEs.npy', ECEs)


if __name__ == "__main__":
    print(sys.argv[1], sys.argv[2])
    main(sys.argv[1], sys.argv[2])
    #    dataset_name, model_name

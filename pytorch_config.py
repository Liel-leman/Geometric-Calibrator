# Build the CNN
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from calibrators import TSCalibrator, EnsembleTSCalibrator, StabilityCalibrator, IsotonicCalibrator, PlattCalibrator, \
    SBCCalibrator, HBCalibrator
from utils import ECE_calc

# classes of different networks:

class SignLanguage_net(nn.Module):

    def __init__(self, stride=1, dilation=1, n_classes=24):
        super(SignLanguage_net, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            # input=(batch, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=self.stride,
                      dilation=self.dilation),
            nn.BatchNorm2d(8),
            # (batch, 8, 28, 28)
            nn.AvgPool2d(2),
            # (batch, 8, 14, 14)
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=self.stride,
                      dilation=self.dilation),
            nn.BatchNorm2d(16),
            # (batch, 16, 14, 14)
            nn.AvgPool2d(2),
            # (batch, 16, 7, 7)
            nn.ReLU()
        )

        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
        # (batch, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.n_classes)
        # (batch, 25)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)

        return x

class MNIST_net(nn.Module):
    def __init__(self):
        super(MNIST_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class GTSRB_net(nn.Module):
    def __init__(self, pretrained_model):
        super(GTSRB_net, self).__init__()
        self.rn50 = pretrained_model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256, 43)

    def forward(self, X):
        X = self.rn50(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X

class Fashion_net(nn.Module):

    def __init__(self):
        super(Fashion_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

class CIFAR_net(nn.Module):
    def __init__(self, dropout=0.2):
        super(CIFAR_net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # 3*32*32 -> 32*32*32
        self.dropout1 = nn.Dropout(p=dropout)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 32*32*32 -> 16*16*32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        # 16*16*32 -> 16*16*64
        self.dropout2 = nn.Dropout(p=dropout)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 16*16*64 -> 8*8*64
        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout4 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.dropout1(self.conv1(x))
        x = self.pool1(F.relu(x))
        x = self.dropout2(self.conv2(x))
        x = self.pool2(F.relu(x))
        x = x.view(-1, self.num_flat_features(x))
        # self.num_flat_features(x) = 8*8*64 here.
        # -1 means: get the rest a row (in this case is 16 mini-batches)
        # pytorch nn only takes mini-batch as the input

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# load data to dataloader:

class SignLanguage_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data.reshape(-1, 28, 28, 1)
        self.label = label
        self.transform = transform
        self.img_shape = data.shape

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
        return img.float(), label

    def __len__(self):
        return len(self.data)

class MNIST_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data.reshape(-1, 28, 28, 1)
        self.label = label
        self.transform = transform
        self.img_shape = data.shape

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
        return img.float(), label

    def __len__(self):
        return len(self.data)

class GTSRB_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data.reshape(-1, 30, 30, 3)
        self.label = label
        self.transform = transform
        self.img_shape = data.shape

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
        return img.float(), label

    def __len__(self):
        return len(self.data)

class Fashion_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data.reshape(-1, 28, 28, 1)
        self.label = label
        self.transform = transform
        self.img_shape = data.shape

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
        return img.float(), label

    def __len__(self):
        return len(self.data)

class CIFAR_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data.reshape(-1, 32, 32, 3)
        self.label = label
        self.transform = transform
        self.img_shape = data.shape

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
        return img.float(), label

    def __len__(self):
        return len(self.data)

######################## helper functions:

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

def compute_all_ece(test_proba, Y_pred_test,Y_pred_val, logits_val, val_proba, data, test_loader, val_loader, stability_val, stability_test, sep_test, sep_val):
    ECEs = {}
    # ['TSCalibrator','EnsembleTSCalibrator','HBCalibrator','SBCCalibrator','StabilityCalibrator']

    ECEs['base'] = ECE_calc(test_proba, Y_pred_test, test_loader.dataset.label, bins=15)


    # TSCalibrator
    tsCal = TSCalibrator()
    tsCal.fit(logits_val, val_loader.dataset.label)
    test_proba_tsCal = tsCal.calibrate(test_proba)
    ECEs['TSCalibrator'] = ECE_calc(test_proba_tsCal, Y_pred_test, test_loader.dataset.label, bins=15)

    # EnsembleTSCalibrator
    etsCal = EnsembleTSCalibrator()
    etsCal.fit(logits_val, val_loader.dataset.label)
    test_proba_etsCal = etsCal.calibrate(test_proba)
    ECEs['EnsembleTSCalibrator'] = ECE_calc(test_proba_etsCal, Y_pred_test, test_loader.dataset.label, bins=15)

    # HB_toplabel
    HBcali = HBCalibrator()
    HBcali.fit(val_proba, data.y_val + 1)
    prob_HB = HBcali.calibrate(test_proba)
    ECEs['HBCalibrator'] = ECE_calc(prob_HB, Y_pred_test, data.y_test, bins=15)

    # SBCCalibrator
    SBCcali = SBCCalibrator()
    SBCcali.fit(val_proba, data.y_val)
    SBC_probs_test = SBCcali.calibrate(test_proba)
    y_SBC_test = np.argmax(SBC_probs_test,axis=1)
    ECEs['SBCCalibrator'] = ECE_calc(SBC_probs_test, y_SBC_test, data.y_test, bins=15)

    #IsotonicCalibrator
    IsoCal = IsotonicCalibrator()
    IsoCal.fit(val_proba, Y_pred_val == data.y_val)
    test_proba_iso = IsoCal.calibrate(test_proba)
    ECEs['IsotonicCalibrator'] = ECE_calc(test_proba_iso, Y_pred_test, data.y_test, bins=15)

    # PlattCalibrator
    plattCal = PlattCalibrator()
    plattCal.fit(val_proba, Y_pred_val == data.y_val)
    test_proba_platt = plattCal.calibrate(test_proba)
    ECEs['PlattCalibrator'] = ECE_calc(test_proba_platt, Y_pred_test, data.y_test, bins=15)

    # StabilityCalibrator
    StabilityCal = StabilityCalibrator()
    StabilityCal.fit(stability_val, data.y_val, Y_pred_val)
    prob_test_stab = StabilityCal.calibrate(stability_test)
    ECEs['StabilityCalibrator'] = ECE_calc(prob_test_stab, Y_pred_test, data.y_test, bins=15)

    # SeperationCalibrator
    SepCal = StabilityCalibrator()
    SepCal.fit(sep_val, data.y_val, Y_pred_val)
    prob_test_sep = SepCal.calibrate(sep_test)
    ECEs['SeperationCalibrator'] = ECE_calc(prob_test_sep, Y_pred_test, data.y_test, bins=15)

    # # stab->sbc
    # prob_val_stab = StabilityCal.calibrate(stability_val)
    # hot_encoded_val_probs = hot_padding(prob_val_stab, Y_pred_val, data.num_labels)
    # hot_encoded_test_probs = hot_padding(prob_test_stab, Y_pred_test, data.num_labels)
    #
    # calibrator = SBC.PlattBinnerMarginalCalibrator(len(val_proba), num_bins=15)
    # calibrator.train_calibration(hot_encoded_val_probs, data.y_val)
    #
    # SBCStab_probs_test = calibrator.calibrate(hot_encoded_test_probs)
    # ECEs['stab->SBC-Calibrator'] = ECE_calc(SBCStab_probs_test, Y_pred_test, data.y_test, bins=15)

    return ECEs

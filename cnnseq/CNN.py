import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from cnnseq import utils
import os
import matplotlib.pyplot as plt
import json
from cnnseq.utils_models import set_optimizer_parameters, load_json, save_json
from cnnseq.CNFN2D import CNFN_2MF, test as test_cnfn, fuzzy_update


# Convolutional neural network (two convolutional layers) for HSL (3 x 100 x 100)
class ConvNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        # Input and output channels differ
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 9, kernel_size=6, stride=1, padding=2),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(9, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(12 * 3 * 3, num_classes)
        '''
        # Same input and output channels (3: HSL)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=6, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(3 * 6 * 6, 64)
        # self.fc2 = nn.Linear(64, num_classes)
        self.fc = nn.Linear(3 * 6 * 6, num_classes)
        # self.act = nn.ReLU()
        self.act = nn.Softmax()

    def forward(self, x):
        # print("X: {}".format(x.shape))
        out = self.layer1(x)
        # print("Layer1: {}".format(out.shape))
        out = self.layer2(out)
        # print("Layer2: {}".format(out.shape))
        out = self.layer3(out)
        # print("Layer3: {}".format(out.shape))
        out = self.dropout(out)
        hidden = out.reshape(out.size(0), -1)
        # print("Reshape hidden: {}".format(hidden.shape))
        out = self.fc(hidden)
        # out = self.fc2(out)
        # print("FC: {}".format(out.shape))
        out = self.act(out)
        # print("Act: {}".format(out.shape))
        return out, hidden


# Convolutional neural network (two convolutional layers) for HSL histogram (3 x 32)
class ConvNetHist(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(ConvNetHist, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = nn.Linear(3 * 8, num_classes)

    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        hidden = out.reshape(out.size(0), -1)
        # print(hidden.shape)
        out = self.fc(hidden)
        # print(out.shape)
        return out, hidden


# Convolutional neural network (two convolutional layers) for HSL histogram flattened to 1 x 96
class ConvNetHistFlatten(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super(ConvNetHistFlatten, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = nn.Linear(24 * 6, num_classes)

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        hidden = out.reshape(out.size(0), -1)
        #print(hidden.shape)
        out = self.fc(hidden)
        #print(out.shape)
        return out, hidden


def train(model, args, dataloader_label_train, dataloader_train, dataloader_label_test, dataloader_test, bs=1,
          target_bin=True, criterion_type='MSE'):

    if criterion_type == 'MSE':
        criterion = nn.MSELoss()  # .cuda()
    else:
        criterion = nn.CrossEntropyLoss(size_average=True)  # .cuda()

    partial_name = 'res_{}_{}_{}_{}_{}_ep_{}_bs_{}_lr_{}'.format(args.model_type, args.data_type, args.target_bin,
                                                              args.emotion_res, criterion_type, str(args.num_epochs),
                                                              str(args.batch_size), str(args.learning_rate))

    params = vars(args)
    mfc_lr = float(params['LR']["FUZZ"])
    wc_lr = params['LR']["CONV"]
    mfdc_lr = float(params['LR']["DEFUZZ"])

    lr_fuzz = Variable(torch.tensor(mfc_lr).float(), requires_grad=True)
    lr_conv1 = Variable(torch.tensor(wc_lr['conv1']).float(), requires_grad=True)
    lr_conv2 = Variable(torch.tensor(wc_lr['conv2']).float(), requires_grad=True)
    lr_conv3 = Variable(torch.tensor(wc_lr['conv3']).float(), requires_grad=True)
    lr_defuzz = Variable(torch.tensor(mfdc_lr).float(), requires_grad=True)

    #optimizer, partial_name = set_optimizer_parameters([model.parameters(), {lr_fuzz, lr_conv1, lr_conv2, lr_conv3, lr_defuzz}],
    #                                                   params, partial_name=partial_name)

    optimizer, partial_name = set_optimizer_parameters(model.parameters(), params, partial_name=partial_name)
    optimizer_fuzz = torch.optim.Adam({lr_fuzz, lr_conv1, lr_conv2, lr_conv3, lr_defuzz}, lr=params['learning_rate'],
                                      weight_decay=params['weight_decay'])

    args.results_dir = args.results_dir + partial_name + '/'
    utils.ensure_dir(args.results_dir)

    # Save args in json file so model can be fully loaded independently
    save_json(args)
    log_file = open(args.results_dir + args.log_filename, 'w')
    log_file_best_train = open(args.results_dir + args.log_filename_best_train, 'w')
    log_file_best_test = open(args.results_dir + args.log_filename_best_test, 'w')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor,
                                                           verbose=True)
    scheduler_fuzz = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fuzz, mode='min',
                                                                factor=args.scheduler_factor, verbose=True)
    # Train
    loss_arr = []
    epoch_arr = []
    first_iter = True
    train_acc_arr = []
    train_acc_ep_arr = []
    test_acc_arr = []
    test_acc_ep_arr = []
    for epoch in range(args.num_epochs):
        for d, l in zip(dataloader_train.keys(), dataloader_label_train.keys()):
            label = dataloader_label_train[l]
            data = dataloader_train[d]
            img = data
            lab = label
            with torch.no_grad():
                img = Variable(img, requires_grad=False).cuda(args.gpu_num[0])
                lab = Variable(lab, requires_grad=False).cuda(args.gpu_num[0])
            # print(np.shape(img))
            # ===================forward=====================
            output, _ = model(img)
            if not target_bin:
                output = output.squeeze()
            # print(np.shape(output))
            loss = criterion(output, lab)
            # ===================backward====================
            optimizer.zero_grad()
            optimizer_fuzz.zero_grad()
            loss.backward()
            # scheduler.step(loss)  # Test use of scheduler to change lr after it plateaus
            optimizer.step()
            optimizer_fuzz.step()

            # If fuzzy update fuzzy variables
            if args.model_type == 'fuzzy':
                # fuzzy_update(model, params)
                fuzzy_update(model, params, mfc_lr=lr_fuzz,
                             wc_lr={'conv1': lr_conv1, 'conv2': lr_conv2, 'conv3': lr_conv3}, mfdc_lr=lr_defuzz)

        scheduler.step(loss)
        scheduler_fuzz.step(loss)
        loss_arr.append(loss.item())
        epoch_arr.append(epoch + 1)

        # Save steps
        if epoch % args.save_step == 0:
            # ===================log========================
            # Save model
            torch.save(model.state_dict(), args.results_dir + args.saved_model)

            # Save best model
            if criterion_type == 'MSE':
                train_acc, _ = test_with_model(model, dataloader_label_train, dataloader_train, target_bin=target_bin)
                test_acc, _ = test_with_model(model, dataloader_label_test, dataloader_test, target_bin=target_bin)
            else:
                train_acc = test_cnfn(model, dataloader_label_train, dataloader_train)
                test_acc = test_cnfn(model, dataloader_label_test, dataloader_test)

            if first_iter:
                max_train_acc = train_acc
                train_acc_arr.append(max_train_acc)
                max_test_acc = test_acc
                test_acc_arr.append(max_test_acc)
                max_train_acc_ep = epoch + 1
                train_acc_ep_arr.append(max_train_acc_ep)
                max_test_acc_ep = epoch + 1
                test_acc_ep_arr.append(max_test_acc_ep)
                torch.save(model.state_dict(), args.results_dir + args.saved_model_best_train)
                torch.save(model.state_dict(), args.results_dir + args.saved_model_best_test)
                first_iter = False
                log_str_best_train = 'Train: epoch [{}/{}], loss {:.4f}, acc {:.4f}'.\
                    format(epoch + 1, args.num_epochs, loss.item(), max_train_acc)
                log_str_best_test = 'Test: epoch [{}/{}], loss {:.4f}, acc {:.4f}'.\
                    format(epoch + 1, args.num_epochs, loss.item(), max_test_acc)
            else:
                if (max_test_acc < test_acc) and (test_acc <= train_acc):
                    max_test_acc = test_acc
                    test_acc_arr.append(max_test_acc)
                    max_test_acc_ep = epoch + 1
                    test_acc_ep_arr.append(max_test_acc_ep)
                    torch.save(model.state_dict(), args.results_dir + args.saved_model_best_test)
                    log_str_best_test = 'Test: epoch [{}/{}], loss {:.4f}, acc {:.4f}'.\
                        format(epoch + 1, args.num_epochs, loss.item(), max_test_acc)

                if max_train_acc < train_acc:
                    max_train_acc = train_acc
                    train_acc_arr.append(max_train_acc)
                    max_train_acc_ep = epoch + 1
                    train_acc_ep_arr.append(max_train_acc_ep)
                    torch.save(model.state_dict(), args.results_dir + args.saved_model_best_train)
                    log_str_best_train = 'Train: epoch [{}/{}], loss {:.4f}, acc {:.4f}'.\
                        format(epoch + 1, args.num_epochs, loss.item(), max_train_acc)

            log_str = 'Training {}%, epoch [{}/{}], loss: {:.4f}, train_acc {:.4f} at {} ep, test_acc: {:.4f} at {} ep,' \
                      ' lr: [fuzz {:.4f}, conv1 {:.4f}, conv2 {:.4f}, conv3 {:.4f}, defuzz {:.4f}, fc {:.4f}]'.\
                format(100*(epoch + 1)/args.num_epochs, epoch + 1, args.num_epochs, loss.item(), max_train_acc,
                        max_train_acc_ep, max_test_acc, max_test_acc_ep, lr_fuzz, lr_conv1, lr_conv2, lr_conv3,
                       lr_defuzz, args.learning_rate)
            #log_str = 'Training {}%, epoch [{}/{}], loss: {:.4f}, train_acc {:.4f} at {} ep, test_acc: {:.4f} at {} ep'\
            #    .format(100 * (epoch + 1) / args.num_epochs, epoch + 1, args.num_epochs, loss.item(), max_train_acc,
            #            max_train_acc_ep, max_test_acc, max_test_acc_ep)
            print(log_str)

            log_file.write(log_str + '\n')
            log_file_best_train.write(log_str_best_train + '\n')
            log_file_best_test.write(log_str_best_test + '\n')
            # print("Shapes - label: {}, data: {}".format(np.shape(lab), np.shape(img)))

    # Save model
    torch.save(model.state_dict(), args.results_dir + args.saved_model)

    # Save training loss
    plt.figure(figsize=[6, 6])
    plt.plot(epoch_arr, loss_arr, '*-')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('on')
    plt.savefig("{}training_loss.svg".format(args.results_dir))
    plt.savefig("{}training_loss.png".format(args.results_dir))
    plt.cla()

    plt.figure(figsize=[6, 6])
    plt.plot(train_acc_ep_arr, train_acc_arr, '*-')
    plt.title('Model Train Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid('on')
    plt.savefig("{}train_acc.svg".format(args.results_dir))
    plt.savefig("{}train_acc.png".format(args.results_dir))
    plt.cla()

    plt.figure(figsize=[6, 6])
    plt.plot(train_acc_ep_arr, train_acc_arr, '*-')
    plt.title('Model Test Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid('on')
    plt.savefig("{}test_acc.svg".format(args.results_dir))
    plt.savefig("{}test_acc.png".format(args.results_dir))
    plt.cla()

    # Get weights
    # print("\nModel keys: {}".format(model.state_dict().keys()))
    return lab, img, args.results_dir


def load_model(params, results_dir_model, saved_model_path, trim_model_name=False):
    num_classes = params['num_classes']
    if params['model_type'] == 'vanilla':
        if params['data_type'] == 'histFlatten':
            model = ConvNetHistFlatten(num_classes=num_classes, dropout=params['DR']).cuda()
        elif params['data_type'] == 'hist':
            model = ConvNetHist(num_classes=num_classes, dropout=params['DR']).cuda()
        else:
            model = ConvNet(num_classes=num_classes, dropout=params['DR']).cuda()
    else:
        # np.random.seed(params['SEED'])
        # torch.manual_seed(params['SEED'])
        # torch.cuda.manual_seed(params['SEED'])
        model = CNFN_2MF(params).cuda()

    pretrained_state = torch.load(results_dir_model + saved_model_path)
    if trim_model_name:
        pretrained_state = utils.fix_unexpected_keys_error(pretrained_state)
    model.load_state_dict(pretrained_state)
    # print("Loading model from: " + results_dir_model + saved_model_path)

    # model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    return model


def test_with_model(model, dataloader_label_test, dataloader_test, target_bin=True, save_feats_pickle_filename=None):
    correct_samples = 0
    euc_dist = 0
    num_samples = 0
    hidden_dict = {}
    output_dict = {}
    label_dict = {}
    for d, l in zip(dataloader_test.keys(), dataloader_label_test.keys()):
        label = dataloader_label_test[l]
        data = dataloader_test[d]
        # print("Shapes1 - label: {}, data: {}".format(np.shape(label), np.shape(data)))
        img = data
        lab = label
        with torch.no_grad():
            img = Variable(img, requires_grad=False).cuda()
        # print("Shapes2 - label: {}, img: {}".format(np.shape(lab), np.shape(img)))
        output, hidden = model(img)
        if not target_bin:
            output = output.squeeze()
        # print("Shapes3 - label: {}, out: {}, img: {}".format(np.shape(lab), np.shape(output), np.shape(img)))
        pic = output.cpu().data
        s = np.shape(pic)

        gen_out = pic.numpy()
        output_dict[d] = gen_out
        hidden_dict[d] = hidden.data.cpu().numpy()
        label_dict[d] = label.data.cpu().numpy()
        lab = lab.numpy()
        # Performance evaluation by comparing target and obtained output
        for i, c in zip(lab, gen_out):
            # Accuracy
            if target_bin:
                i_max_idx = np.argmax(i)
                c_max_idx = np.argmax(c)
            else:
                i_max_idx = i
                c_max_idx = round(c)
            if i_max_idx == c_max_idx:
                correct_samples += 1

            # Euclidian distance:
            # print("Shapes = i: {}, c: {}".format(np.shape(i), np.shape(c)))
            euc_dist += np.linalg.norm(i - c)
            num_samples += 1

    # Save hidden state in pickle for t-SNE plotting
    if save_feats_pickle_filename is not None:
        utils.save_feats_pickle(hidden_dict, output_dict, label_dict, save_feats_pickle_filename)

    correct_samples_perc = 100 * correct_samples / num_samples
    euc_dist_mean = euc_dist / num_samples
    return correct_samples_perc, euc_dist_mean


def test(args, dataloader_label_test, dataloader_test, target_bin=True, best_model=False, best_train=False):
    if best_model:
        if best_train:
            saved_model_path = args.saved_model_best_train
        else:
            saved_model_path = args.saved_model_best_test
    else:
        saved_model_path = args.saved_model

    results_dir = args.results_dir

    params = load_json(results_dir, args.saved_model_parameters)
    model = load_model(params, results_dir, saved_model_path)
    correct_samples_perc, euc_dist_mean = test_with_model(model, dataloader_label_test, dataloader_test,
                                                          target_bin=target_bin,
                                                          save_feats_pickle_filename=args.results_dir +
                                                                                     args.feature_pickle_filename)

    return correct_samples_perc, euc_dist_mean


def test_with_cnfn_test_func(args, dataloader_label_test, dataloader_test, best_model=False, best_train=False):
    if best_model:
        if best_train:
            saved_model_path = args.saved_model_best_train
        else:
            saved_model_path = args.saved_model_best_test
    else:
        saved_model_path = args.saved_model

    results_dir = args.results_dir

    params = load_json(results_dir, args.saved_model_parameters)
    model = load_model(params, results_dir, saved_model_path)
    test_acc = test_cnfn(model, dataloader_label_test, dataloader_test,
                         save_feats_pickle_filename=args.results_dir+args.feature_pickle_filename)
    return test_acc

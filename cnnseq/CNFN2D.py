import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from cnnseq.utils import save_feats_pickle


torch.cuda.manual_seed_all(1)
torch.manual_seed(1)


"""Modification of code by Linh Nguyen"""


class FuzzyConv(nn.Sequential):
    def __init__(self, pars):
        super(FuzzyConv, self).__init__()

        Ci = pars['IN_CHANNEL']  # Channel in
        Co = pars["KERNELS_NUM"]  # Channel out 100
        Ks = pars['KERNELS_SIZE']

        self.add_module('conv1', nn.Conv2d(Ci, Co[0], Ks[0], stride=2, padding=1, bias=False))
        self.add_module('bnorm1', nn.BatchNorm2d(Co[0]))    # BatchNorm1d(Co)) ???

        self.add_module('conv2', nn.Conv2d(Co[0], Co[1], Ks[1], stride=1, padding=2, bias=False))
        self.add_module('bnorm2', nn.BatchNorm2d(Co[1]))

        self.add_module('conv3', nn.Conv2d(Co[1], Co[2], Ks[2], stride=1, padding=2, bias=False))
        self.add_module('bnorm3', nn.BatchNorm2d(Co[2]))


class CNFN_2MF(nn.Module):

    def __init__(self, pars):
        super(CNFN_2MF, self).__init__()
        dtype = torch.cuda.FloatTensor

        self.eps = pars['EPS']

        C = pars['CLASS_NUM']  # 2
        Ci = pars['IN_CHANNEL']  # 3
        Si = pars['IN_SIZE']  # 100 x 100
        Co = pars['KERNELS_NUM']  # Channel out 100
        Ks = pars['KERNELS_SIZE']  # 4, 6, 6
        self.FC = pars['FC_WT']

        sigmaf = pars['SIGMAF']

        sigmac = pars['SIGMAC']
        mf_name = ['lo', 'me', 'hi']
        # Train: min -13.6956520081, max 275.547668457
        mf_params = {
            'lo':   {
                    'f_center_init_min': 0.2499, 'f_center_init_max': 0.2501,
                    # 'f_center_min': 0, 'f_center_max': 0.5,
                    'f_center_min': -13.7, 'f_center_max': 130.,
                    'conv_center_init_min': 0, 'conv_center_init_max': 0.1,
                    'conv_center_min': -0.75, 'conv_center_max': 0.25,
                    'df_center_init_min': 0.2499, 'df_center_init_max': 0.2501,
                    'df_center_min': 0, 'df_center_max': 0.5,
                    # 'fixed_cent': 0.25,  # 0
                    'fixed_cent': 58.15,  # 0
                    },

            'hi':   {
                    'f_center_init_min': 0.7499, 'f_center_init_max': 0.7501,
                    # 'f_center_min': 0.5, 'f_center_max': 1,
                    'f_center_min': 130., 'f_center_max': 275.5,

                    'conv_center_init_min': 0.9, 'conv_center_init_max': 1.0,
                    'conv_center_min': 0.75, 'conv_center_max': 1.75,

                    'df_center_init_min': 0.7499, 'df_center_init_max': 0.7501,
                    'df_center_min': 0, 'df_center_max': 0.5,

                    # 'fixed_cent': 0.75,  # 1.0,
                    'fixed_cent': 202.75,  # 1.0,
                    },
            }

        self.mfc_lr = float(pars['LR']["FUZZ"])
        self.wc_lr = pars['LR']["CONV"]
        self.mfdc_lr = float(pars['LR']["DEFUZZ"])

        self.fcnn = {}
        for mf_key, mf_val in mf_params.items():
            self.fcnn[mf_key] = {
                'fuzz': {
                        'lr': Variable(torch.tensor(self.mfc_lr).float(), requires_grad=True),
                        'delta_center': Variable(torch.FloatTensor(Ci, Si[0], Si[1]).fill_(0).type(dtype), requires_grad=False),
                        'center': Variable(torch.FloatTensor(Ci, Si[0], Si[1]).uniform_(mf_val['f_center_init_min'],
                                                                              mf_val['f_center_init_max']).type(dtype),
                                           requires_grad=True),
                        'delta_sigma': Variable(torch.FloatTensor(Ci, Si[0], Si[1]).fill_(0).type(dtype),
                                      requires_grad=False),
                        'sigma': Variable(torch.FloatTensor(Ci, Si[0], Si[1]).fill_(sigmaf).type(dtype),
                                           requires_grad=True),
                        'min': mf_val['f_center_min'], 'max': mf_val['f_center_max'],
                        },

                'conv1': {
                        'lr': Variable(torch.tensor(self.wc_lr['conv1']).float(), requires_grad=True),
                        'delta_center': Variable(torch.FloatTensor(Co[0], Ci, Ks[0], Ks[0]).fill_(0).type(dtype), requires_grad=False),
                        'center': Variable(
                                   torch.FloatTensor(Co[0], Ci, Ks[0], Ks[0]).uniform_(mf_val['conv_center_init_min'], mf_val['conv_center_init_max']).type(dtype),
                                                    requires_grad=False),
                        'delta_sigma': Variable(torch.FloatTensor(Co[0], Ci, Ks[0], Ks[0]).fill_(0).type(dtype),
                                              requires_grad=False),
                        'sigma': Variable(torch.FloatTensor(Co[0], Ci, Ks[0], Ks[0]).fill_(sigmac).type(dtype),
                                             requires_grad=False),
                        'min': mf_val['conv_center_min'], 'max': mf_val['conv_center_max'], 'cent':mf_val['fixed_cent']
                            },
                'conv2': {
                        'lr': Variable(torch.tensor(self.wc_lr['conv2']).float(), requires_grad=True),
                        'delta_center': Variable(torch.FloatTensor(Co[1], Co[0], Ks[1], Ks[1]).fill_(0).type(dtype),
                                                    requires_grad=False),
                        'center': Variable(
                                   torch.FloatTensor(Co[1], Co[0], Ks[1], Ks[1]).uniform_(mf_val['conv_center_init_min'], mf_val['conv_center_init_max']).type(dtype),
                                   requires_grad=False),
                        'delta_sigma': Variable(torch.FloatTensor(Co[1], Co[0], Ks[1], Ks[1]).fill_(0).type(dtype),
                                            requires_grad=False),
                        'sigma': Variable(torch.FloatTensor(Co[1], Co[0], Ks[1], Ks[1]).fill_(sigmac).type(dtype),
                                             requires_grad=False),
                        'min': mf_val['conv_center_min'], 'max': mf_val['conv_center_max'], 'cent':mf_val['fixed_cent']
                        },

                'conv3': {
                        'lr': Variable(torch.tensor(self.wc_lr['conv3']).float(), requires_grad=True),
                        'delta_center': Variable(torch.FloatTensor(Co[2], Co[1], Ks[2], Ks[2]).fill_(0).type(dtype),
                                                    requires_grad=False),
                        'center': Variable(
                                   torch.FloatTensor(Co[2], Co[1], Ks[2], Ks[2]).uniform_(mf_val['conv_center_init_min'], mf_val['conv_center_init_max']).type(dtype),
                                                    requires_grad=False),

                        'delta_sigma': Variable(torch.FloatTensor(Co[2], Co[1], Ks[2], Ks[2]).fill_(0).type(dtype),
                                                requires_grad=False),
                        'sigma': Variable(torch.FloatTensor(Co[2], Co[1], Ks[2], Ks[2]).fill_(sigmac).type(dtype),
                                                requires_grad=False),
                        'min': mf_val['conv_center_min'], 'max': mf_val['conv_center_max'], 'cent': mf_val['fixed_cent']
                        },

                'defuzz': {
                        'lr': Variable(torch.tensor(self.mfdc_lr).float(), requires_grad=True),
                        'delta_center': Variable(torch.FloatTensor(self.FC[0]).fill_(0).type(dtype),
                                                    requires_grad=False),
                        'center': Variable(
                               torch.FloatTensor(self.FC[0]).uniform_(mf_val['df_center_init_min'], mf_val['df_center_init_max']).type(dtype),
                                                    requires_grad=True),
                        }
                }


            self.fconv = nn.Sequential()
            for mf_key, mf_val in mf_params.items():
                self.fconv.add_module(mf_key, FuzzyConv(pars))

            self.dropout = nn.Dropout(pars['DR'])
            self.fc1 = nn.Linear(self.FC[0], self.FC[1])
            self.fc2 = nn.Linear(self.FC[1], self.FC[2])

            self.fc3 = nn.Linear(self.FC[0], self.FC[2])

    def conv_and_pool(self, x, conv):

        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool2d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        num = 0
        den = 0
        for mf_idx in range(len(self.fconv)):
            mf_name = list(self.fcnn.keys())[mf_idx]
            test = self.fcnn[mf_name]['fuzz']['center']
            test = test[:, :]
            center_arr = test.expand_as(x)
            sigma_arr = self.fcnn[mf_name]['fuzz']['sigma'][:, :].expand_as(x)
            delta = x - center_arr
            Xf = (-((delta).pow(2.0)) / sigma_arr.pow(2.0)).exp()
            # print("Xf: {}".format(np.shape(Xf)))
            x_1 = F.relu(self.fconv[mf_idx].bnorm1(self.fconv[mf_idx].conv1(Xf)))
            x_1 = F.max_pool2d(x_1, kernel_size=(2))
            # print("x1: {}".format(np.shape(x_1)))
            x_2 = F.relu(self.fconv[mf_idx].bnorm2(self.fconv[mf_idx].conv2(x_1)))
            x_2 = F.max_pool2d(x_2, kernel_size=(2))
            # print("x2: {}".format(np.shape(x_2)))
            x_3 = F.relu(self.fconv[mf_idx].bnorm3(self.fconv[mf_idx].conv3(x_2)))
            x_3 = F.max_pool2d(x_3, kernel_size=(2))
            # x_conv = torch.cat((x_1, x_4, x_5), 1)  # (N,len(Ks)*Co)
            # print("x3: {}".format(np.shape(x_3)))

            x_conv = self.dropout(x_3)
            # print("x_conv: {}".format(np.shape(x_conv)))
            x_conv = x_conv.view(-1, self.FC[0])
            # print("x_conv: {}".format(np.shape(x_conv)))

            den = den + x_conv
            num += x_conv * self.fcnn[mf_name]['defuzz']['center'].expand_as(x_conv)
        den += self.eps
        x_out = num / den  # self.h0_defuzz = x
        hidden = x_out.reshape(x_out.size(0), -1)
        # print("x_out: {}".format(np.shape(x_out)))
        x_fc1 = self.fc1(x_out)  # (N,C)
        # print("x_fc1: {}".format(np.shape(x_fc1)))
        logit = self.fc2(x_fc1)
        # logit = self.fc3(x_out)
        # print("logit: {}".format(np.shape(logit)))
        return logit, hidden


def fuzzy_update(model, params, mfc_lr=None, wc_lr=None, mfdc_lr=None):
    # Fuzzy params update
    if mfc_lr is None:
        mfc_lr = float(params['LR']["FUZZ"])
    if wc_lr is None:
        wc_lr = params['LR']["CONV"]
    if mfdc_lr is None:
        mfdc_lr = float(params['LR']["DEFUZZ"])

    layers_lr = 0.001

    momentum = params["momentum"]
    eps = params["EPS"]
    for mf_key, mf_val in model.fcnn.items():
        for keys, values in model.fcnn[mf_key].items():
            if keys == 'fuzz':
                # model.fcnn[mf_key][keys]['lr'].data = layers_lr * model.fcnn[mf_key][keys]['lr'].grad.data
                lr = mfc_lr.cpu().data.numpy()  # model.fcnn[mf_key][keys]['lr'].data
                # print("{} lr: {}".format(keys, lr))
                grad = model.fcnn[mf_key][keys]['center'].grad.data

                lr_grad = float(lr) * grad
                model.fcnn[mf_key][keys]['delta_center'].data = lr_grad + \
                    momentum * model.fcnn[mf_key][keys]['delta_center'].data

                model.fcnn[mf_key][keys]['center'].data = \
                    torch.clamp(model.fcnn[mf_key][keys]['center'].data - model.fcnn[mf_key][keys]['delta_center'].data,
                                model.fcnn[mf_key][keys]['min'], model.fcnn[mf_key][keys]['max']
                                )

            if 'conv' in keys:
                for name, module in model.fconv._modules[mf_key].named_children():
                    if name == keys:
                        # model.fcnn[mf_key][keys]['lr'].data = layers_lr * model.fcnn[mf_key][keys]['lr'].grad.data
                        lr = wc_lr[keys].cpu().data.numpy()  # model.fcnn[mf_key][keys]['lr'].data
                        # print("{} lr: {}".format(keys, lr))
                        moment = momentum * model.fcnn[mf_key][keys]['delta_center'].data
                        grad = model.fconv._modules[mf_key]._modules[keys].weight.grad.data
                        lr_grad = float(lr) * grad
                        model.fcnn[mf_key][keys]['delta_center'].data = lr_grad + moment
                        model.fcnn[mf_key][keys]['center'].data = \
                            torch.clamp(
                                model.fcnn[mf_key][keys]['center'].data - model.fcnn[mf_key][keys]['delta_center'].data,
                                model.fcnn[mf_key][keys]['min'], model.fcnn[mf_key][keys]['max'])
                        model.fconv._modules[mf_key]._modules[keys].weight.data = (-(
                            (model.fcnn[mf_key][keys]['cent'] - model.fcnn[mf_key][keys]['center'].data).pow(2.0)) /
                                                                                   (model.fcnn[mf_key][keys][
                                                                                        'sigma'].data.pow(
                                                                                       2.0) + eps)).exp()
                        # print("{} lr: {}".format(keys, lr))

            if keys == 'defuzz':
                # model.fcnn[mf_key][keys]['lr'].data = layers_lr * model.fcnn[mf_key][keys]['lr'].grad.data
                lr = mfdc_lr.cpu().data.numpy()  # model.fcnn[mf_key][keys]['lr'].data
                # print("{} lr: {}".format(keys, lr))
                grad = model.fcnn[mf_key][keys]['center'].grad.data
                lr_grad = float(lr) * grad
                model.fcnn[mf_key][keys]['delta_center'].data = lr_grad + momentum *\
                                                                          model.fcnn[mf_key][keys]['delta_center'].data
                model.fcnn[mf_key][keys]['center'].data = model.fcnn[mf_key][keys]['center'].data - \
                                                          model.fcnn[mf_key][keys][
                                                              'delta_center'].data


def train(model, loss, optimizer, x_val, y_val, params):
    model.train()  # toggle the model into training mode
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()
    #    print(x.size())
    # Forward
    fx, _ = model.forward(x)
    # print(fx.size())
    output = loss.forward(fx, y)

    # Backward
    output.backward()
    #    print(model.bn1_1.weight.grad.data.max())

    #    print(model.bn1_1.weight.grad.data.max())
    # Update parameters
    optimizer.step()

    fuzzy_update(model, params)

    return output.data[0]


def test(model, dataloader_label_test, dataloader_test, save_feats_pickle_filename=None):
    num_batches = 0
    test_acc = 0.
    hidden_dict = {}
    output_dict = {}
    label_dict = {}
    for d, l in zip(dataloader_test.keys(), dataloader_label_test.keys()):
        # print("l: {}, d: {}".format(l, d))
        label = dataloader_label_test[l]
        data = dataloader_test[d]
        img = data
        lab = label
        with torch.no_grad():
            img = Variable(img).cuda()
            lab = Variable(lab).cuda()
        test_acc_tmp, hidden, output = predict(model, img, lab)
        hidden_dict[d] = hidden
        output_dict[d] = output
        label_dict[d] = label.data.cpu().numpy()
        test_acc += test_acc_tmp
        num_batches += 1

    # Save hidden state in pickle for t-SNE plotting
    if save_feats_pickle_filename is not None:
        save_feats_pickle(hidden_dict, output_dict, label_dict, save_feats_pickle_filename)

    test_acc = test_acc / max(1, num_batches)
    return test_acc


def predict(model, x_val, y_val, bsize=0):
    model.eval()

    # hidden_dict = {}
    if bsize == 0:
        # valX = torch.from_numpy(x_val).float().cuda(GPU)
        x = Variable(x_val, requires_grad=False)
        output, hidden = model.forward(x)
        # hidden_dict[0] = hidden
        hidden = hidden.data.cpu().numpy()
        output = output.data.cpu().numpy()
        pred = output.argmax(axis=1)
        val_acc = 100. * np.mean(pred == y_val.cpu().numpy())
        # print("Feature shape: {}".format(np.shape(hidden)))  # 30, 108
    else:
        val_acc = 0
        num_batches = len(x_val) // bsize
        for k in range(num_batches):
            start, end = k * bsize, (k + 1) * bsize

            b_valX = Variable(x_val[start:end], requires_grad=False)
            b_valY = y_val[start:end]
            output, hidden = model.forward(b_valX).data.cpu().numpy()
            # hidden_dict[k] = hidden
            pred = output.argmax(axis=1)
            val_acc += 100. * np.mean(pred == b_valY.cpu().numpy())
        val_acc = val_acc / num_batches

    return val_acc, hidden, output

import torch
import numpy as np
from cnnseq import utils
import argparse
from cnnseq.CNN import ConvNetHistFlatten, ConvNetHist, ConvNet, train, test, test_with_cnfn_test_func
from cnnseq.CNFN2D import CNFN_2MF
from cnnseq.utils_models import CLASS, CLASS2D, get_num_classes
import os
import json

"""
    CNN Classifier for Scene classification with emotion label 1/0 (positive/negative)

    data_type:
        hist/histFlatten:
            batch_size=90
            data_filename='video_feats_HSLhist_data'
        HSL:
            batch_size=30
            data_filename='video_feats_HSL_10fps_data'

    Fuzzy is only supported with 2 classes, HSL data input

    Organized version of *ANFIS-RNN/cnn_classifier_hist.py*
    Author: Gwena Cunha
"""


def get_parser():
    parser = argparse.ArgumentParser(description="Convolutional AutoEncoder")

    parser.add_argument('--model_type', type=str, default='vanilla', help='Options: [vanilla, fuzzy]')

    # wav2mid2wav audio - 2 epochs: 70min, 20 epochs:78min, 40 epochs
    # Original audio (no norm) - 2 epochs: 5.4min, 40 epochs: 51.46min (best_test: 76.7837%)
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs to save results')
    parser.add_argument('--save_step', type=int, default=1, help='Save results every N steps')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--gpu_num', type=list, default=[0, 1])
    parser.add_argument('--optimizer_name', type=str, default='adam',
                        help='Optimizer: [adam, sgd, asgd, adagrad, adadelta, rmsprop, rprop]')
    # lr 0.1: 91%, 60%, 97%, 94%, 97%, 60% 80%
    # lr 0.01: asgd 98%
    # lr 0.001 (at 4ep): adadelta (57.43% CNN, 57.43% CNFN), adam (95.9% CNN, 58.93% CNFN)
    # lr 0.01 (at 4ep): adadelta (87.86% CNN, 57.43% CNFN)
    # lr 0.1 (at 4ep): adam (57.4318% CNFN), sgd (42.5682% CNFN), asgd (57.4318% CNFN), adagrad (57.4318% CNFN),
    #  adadelta (57.2413% CNFN), rmsprop (57.4318% CNFN), rprop (57.4318% CNFN)
    # OrigAudio: epochs 40 - test 68.9834, best_test 70.5767, best_train 69.8631%, 6.0537 min
    # Instrumental: test 68.2307, best_test 70.7527, best_train 70.0293%, 5.9623 min
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 1e-3)  # 0.029
    parser.add_argument('--criterion_type', type=str, default='CrossEntropy', help='Options: [MSE, CrossEntropy]')
    parser.add_argument('--momentum', type=float, default=0.98)  # 0.7, 0.98
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 5e-4
    parser.add_argument('--scheduler_factor', type=float,
                        default=0.8, help='Factor by which the learning rate will be changed after plateauing')

    # CNFN
    parser.add_argument('--LR', type=json.loads, default={"FUZZ": 0.4250,
                                                          "CONV": {"conv1": 0.4276, "conv2": 14.3721, "conv3": 70.4043},
                                                          "DEFUZZ": 0.0956,
                                                          "FC": 0.00259})  # 0.029, FC: learning rate used in optimizer

    parser.add_argument('--IN_CHANNEL', type=int, default=3)
    parser.add_argument('--IN_SIZE', type=list, default=[100, 100])
    parser.add_argument('--CLASS_NUM', type=int, default=2)
    parser.add_argument('--KERNELS_SIZE', type=list, default=[4, 6, 5])
    parser.add_argument('--KERNELS_NUM', type=list, default=[3, 3, 3])
    parser.add_argument('--SIGMAF', type=float, default=0.13)
    parser.add_argument('--SIGMAC', type=float, default=0.13)
    parser.add_argument('--NUM_MF', type=int, default=2)
    parser.add_argument('--FC_WT', type=list, default=[108, 64, 2])
    parser.add_argument('--DR', type=float, default=0.2)  # 0.2
    parser.add_argument('--NORM_LIMIT', type=int, default=3)
    parser.add_argument('--EPS', type=float, default=1e-20)
    parser.add_argument('--SEED', type=int, default=1)
    parser.add_argument('--MODE', type=str, default='List')

    # Target emotion
    parser.add_argument('--target_bin', type=str, default='bin', help='Use only for emotion_res=1D or 2D. Options: [bin, num]')
    parser.add_argument('--emotion_res', type=str, default='1D', help='Emotion resolution. Options: [1D, 2D, 2D_raw (TODO)]')

    parser.add_argument('--num_classes', type=int, default=-1, help='Number of classes')

    # Data
    parser.add_argument('--data_dir', type=str,
                        # default='/media/ceslea/DATA/VideoEmotion/DataWithEmotionTags_noText_correctedAudio_hsio/',
                        # default=utils.project_dir_name() + 'data/',
                        # default='/media/root/DATA/PycharmProjects/Scene2Wav/datasets/data_npz/',
                        default='/mnt/Gwena/Scene2Wav/datasets/data_npz/',
                        help='Data directory')
    parser.add_argument('--data_type', type=str, default='HSL', help='HSL histogram or pure HSL. Options: [hist, histFlatten, HSL]')
    # parser.add_argument('--data_filename', type=str, default='video_feats_HSL_10fps_data', help='Data file')
    #parser.add_argument('--data_filename', type=str, default='video_feats_HSLboth_10fps_data_divPerVideo',
    #                    help='Data file')
    # parser.add_argument('--data_filename', type=str, default='video_feats_HSL_10fps_origAudio_10secs_pad',
    #                    help='Data file')
    parser.add_argument('--data_filename', type=str, default='video_feats_HSL_10fps_3secs_intAudio_pad',
                        help='Data file')
    parser.add_argument('--results_dir', type=str, default='./results_cnn4_3secs/', help='Directory to save results')
    #parser.add_argument('--results_dir', type=str,
    #                    # default='./results_cnn/res_fuzzy_HSL_bin_1D_MSE_ep_4_bs_30_lr_0.001_we_1e-05_asgd/',
    #                    default='./results_cnn/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_94.22perf/',
    #                    help='Directory to save results')
    parser.add_argument('--saved_model', type=str, default='cnn_model.pth', help='Path to save model')
    parser.add_argument('--saved_model_parameters', type=str, default='parameters.json',
                        help='Path to save models parameters')
    parser.add_argument('--log_filename', type=str, default='log.txt', help='Log text file')
    parser.add_argument('--saved_model_best_train', type=str, default='cnn_model_best_train.pth', help='Path to save model')
    parser.add_argument('--log_filename_best_train', type=str, default='log_best_train.txt', help='Log text file')
    parser.add_argument('--saved_model_best_test', type=str, default='cnn_model_best_test.pth', help='Path to save model')
    parser.add_argument('--log_filename_best_test', type=str, default='log_best_test.txt', help='Log text file')
    # parser.add_argument('--save_ext', type=str, default='npz', help='Method of Options: [npz, json, pkl]')
    parser.add_argument('--feature_pickle_filename', type=str, default='feats.pkl',
                        help='Pickle with saved features and its respective labels')
    parser.add_argument('--mode', type=str, default='train', help='Options: [train, test]')
    args = parser.parse_args()
    return args


def label_tensor(label, bs=1, tensor_type='float'):
    # print("Label batch")
    label_shape = np.shape(label)
    # print("label_shape: {}".format(label_shape))
    c = label_shape[0]  # c, a = text_shape
    dataloader_text = {}
    # for i in range(0, c-bs+1, bs):
    for i in range(0, c, bs):
        idx_fin = min(i+bs, c)
        d = label[i:idx_fin]
        # print("Bs: {}-{}, samples: {}, shape: {}".format(i, idx_fin, idx_fin-i, np.shape(d)))
        d_tensor = torch.tensor(d)
        if tensor_type == 'float':
            dataloader_text[i] = d_tensor.float()
        else:
            dataloader_text[i] = d_tensor.long()
    # print("dataset label: {}".format(np.shape(dataloader_text.keys())))
    return dataloader_text


def feats_tensor(attributes, bs=1, data_type='hist'):
    # print("Feats tensor")
    dataset = attributes
    dataloader = {}

    # print("np.shape(dataset): {}".format(np.shape(dataset)))
    if data_type == 'histFlatten':
        c, a = np.shape(dataset)  # d_num, 96
    elif data_type == 'hist':
        c, a, b = np.shape(dataset)  # d_num, 3, 32
    else:
        c, a, d1, d2 = np.shape(dataset)  # d_num, 3, 100, 100

    for i in range(0, c, bs):
        idx_fin = min(i+bs, c)
        d = dataset[i:idx_fin]
        # print("Bs: {}-{}, samples: {}, d shape: {}".format(i, idx_fin, idx_fin-i, np.shape(d)))
        if data_type == 'histFlatten':
            d_reshape = np.reshape(d, (idx_fin - i, 1, a))
        elif data_type == 'hist':
            d_reshape = np.reshape(d, (idx_fin - i, a, b))
        else:
            d_reshape = np.reshape(d, (idx_fin - i, a, d1, d2))
        d_tensor = torch.tensor(d_reshape)
        dataloader[i] = d_tensor.float()
    # print("dataloader keys: {}".format(np.shape(dataloader.keys())))
    return dataloader


def load_data1(data, target_bin=True, data_type='hist', emotion_res='1D', do_normalize_data=False):
    if 'label' in data:
        y, y2D, y2D_raw = data['label'], data['label2D'], data['label2D_raw']
    else:  # doesn't support 2D emotion label
        y, y2D, y2D_raw = data['emotion'], data['emotion'], data['emotion']
    if 'hist' in data_type:
        X = data['hist_data']
    else:
        X = data['HSL_data']
    # X = (splices x num_frames x hist_bins x channels) = (624, 90, 3, 32)
    print("Loading data - X: {}, y: {}, y2D: {}, y2D_raw: {}".format(np.shape(X), np.shape(y), np.shape(y2D), np.shape(y2D_raw)))

    s = X.shape
    if do_normalize_data:
        X = utils.normalize_data(X, s)

    if data_type == 'histFlatten':
        att_train = X.reshape(s[0] * s[1], s[2] * s[3])
    elif data_type == 'hist':
        att_train = X.reshape(s[0] * s[1], s[2], s[3])
    else:
        att_train = X.reshape(s[0] * s[1], s[2], s[3], s[4])

    # Target train
    # Expand labels so that each splice frame has a label
    #   Before: 1 video, 38 frames -> 1 label
    #   After expansion: 1 video, 38 frames -> 38 labels
    print("Loading label with data type {}".format(data_type))
    if emotion_res == '2D':
        y = y2D
    else:
        if emotion_res == '2D_raw':
            y = y2D_raw
    if target_bin and (emotion_res == '1D' or emotion_res == '2D'):
        y_name = []
        y_tmp = y
        for i, y_t in enumerate(y):
            y_t = int(y_t)
            if emotion_res == '1D':
                y_name.append(CLASS[y_t])
            else:
                y_t -= 1  # classes ranging from 1 to 4 (change to 0 to 3)
                y_tmp[i] = y_t
                y_name.append(CLASS2D[y_t])
        y = y_tmp
        num_classes = max(y).astype(np.int) + 1
        print("Num classes: {}".format(num_classes))
        y_bin = np.zeros([len(y), num_classes])
        print("y_bin shape: {}".format(np.shape(y_bin)))
        for i in range(0, len(y)):
            y_bin[i, y[i].astype(np.int)] = 1

        label_train = utils.expand_array(y_bin, X.shape[1])
    else:
        label_train = utils.expand_array(y, X.shape[1])
    return att_train, label_train


def load_data(data_dir, data_filename, target_bin=True, data_type='hist', emotion_res='1D', do_normalize_data=False):
    """ Loads data, assumes that train and test .npz files already exits (see main.py to create it)
    """
    train_data_path = "{data_dir}{path}_train.npz".format(data_dir=data_dir, path=data_filename)
    test_data_path = "{data_dir}{path}_test.npz".format(data_dir=data_dir, path=data_filename)

    # Load data
    data_train = np.load(train_data_path)
    att_train, label_train = load_data1(data_train, target_bin=target_bin, data_type=data_type, emotion_res=emotion_res, do_normalize_data=do_normalize_data)

    data_test = np.load(test_data_path)
    att_test, label_test = load_data1(data_test, target_bin=target_bin, data_type=data_type, emotion_res=emotion_res, do_normalize_data=do_normalize_data)

    return att_train, label_train, att_test, label_test


if __name__ == '__main__':

    from timeit import default_timer as timer

    start_time = timer()

    # Settings
    args = get_parser()
    utils.ensure_dir(args.results_dir)

    if args.mode is 'test':
        # Load json file and get model's information
        json_path = os.path.join(args.results_dir, args.saved_model_parameters)
        with open(json_path, 'r') as fp:
            print("Loading JSON file for testing: " + json_path)
            params = json.load(fp)
            args.target_bin = params['target_bin']
            args.data_type = params['data_type']
            args.emotion_res = params['emotion_res']
            args.batch_size = params['batch_size']
            args.data_filename = params['data_filename']
            args.data_dir = params['data_dir']
            args.num_classes = params['num_classes']

    # Load data, separate into test and train and return
    print("\nLoad data...")
    target_bin = True if args.target_bin == 'bin' else False
    # att_train, _, label_train, _, att_test, _, label_test, _ = load_data(args.data_dir, args.data_filename,
    #                                                                     target_bin=target_bin,
    #                                                                     data_type=args.data_type,
    #                                                                     emotion_res=args.emotion_res,
    #                                                                     do_normalize_data=False)

    # IF using MSEloss, use tensor_type='float'. If using CrossEntropy, use , tensor_type='long'
    target_bin2 = target_bin
    tensor_type = 'float'
    if args.criterion_type == 'CrossEntropy':
        target_bin2 = False
        tensor_type = 'long'
    att_train, label_train, att_test, label_test = load_data(args.data_dir, args.data_filename,
                                                             target_bin=target_bin2, data_type=args.data_type,
                                                             emotion_res=args.emotion_res, do_normalize_data=False)
    print("\nLoaded data - train: {}, {}, test: {}, {}".format(np.shape(att_train), np.shape(label_train),
                                                               np.shape(att_test), np.shape(label_test)))
    print("\nTrain: min {}, max {}".format(np.amin(np.amin(np.amin(np.amin(att_train)))),
                                           np.amax(np.amax(np.amax(np.amax(att_train))))))

    # Dataset: (num_lines, num_words_embedded, size_embedding)
    dataloader_train = feats_tensor(att_train, args.batch_size, data_type=args.data_type)
    dataloader_label_train = label_tensor(label_train, args.batch_size, tensor_type=tensor_type)
    dataloader_test = feats_tensor(att_test, args.batch_size, data_type=args.data_type)
    dataloader_label_test = label_tensor(label_test, args.batch_size, tensor_type=tensor_type)
    print("Tensor shape train - label: {}, att: {}".format(np.shape(dataloader_label_train.keys()),
                                                           np.shape(dataloader_train.keys())))
    print("Tensor shape test - label: {}, att: {}".format(np.shape(dataloader_label_test.keys()),
                                                           np.shape(dataloader_test.keys())))
    # for l, d in zip(dataloader_label_train.keys(), dataloader_train.keys()):
    #     print("l: {}, d: {}".format(np.shape(dataloader_label_train[l]), np.shape(dataloader_train[d])))

    if args.num_classes == -1:
        args.num_classes = get_num_classes(target_bin, args.emotion_res)

    # Train CNN
    max_train_acc = 0
    if args.mode == 'train':
        if args.model_type == 'vanilla':
            print("\nCNN model...")
            if args.data_type == 'histFlatten':
                model = ConvNetHistFlatten(num_classes=args.num_classes, dropout=args.DR).cuda(args.gpu_num[0])
            elif args.data_type == 'hist':
                model = ConvNetHist(num_classes=args.num_classes, dropout=args.DR).cuda(args.gpu_num[0])
            else:
                model = ConvNet(num_classes=args.num_classes, dropout=args.DR).cuda(args.gpu_num[0])
        else:
            print("\nFuzzy-CNN model...")
            #np.random.seed(args.SEED)
            #torch.manual_seed(args.SEED)
            #torch.cuda.manual_seed(args.SEED)
            model = CNFN_2MF(vars(args)).cuda(args.gpu_num[0])

        lab, img, res_dir, max_train_acc = train(model, args, dataloader_label_train, dataloader_train, dataloader_label_test,
                                                 dataloader_test, bs=1, target_bin=target_bin, criterion_type=args.criterion_type)

        ## Test the model with last sample
        # last_test_label = {'0': lab.cpu()}
        # last_test_img = {'0': img.cpu()}
        # acc, euc_dist_mean = test(args, last_test_label, last_test_img, bs=1, target_bin=target_bin, num_classes=args.num_classes)
        # print("Last sample - Accuracy: {}%, Euclidian distance: {}".format(acc, euc_dist_mean))

    # Test model with test data
    #acc, euc_dist_mean = test(args, dataloader_label_test, dataloader_test, bs=1, target_bin=target_bin,
    #                          num_classes=args.num_classes, best_model=True)
    #print("Test data - Accuracy: {}%, Euclidian distance: {}".format(acc, euc_dist_mean))

    if args.criterion_type == 'MSE':
        print("Test")
        test_acc, _ = test(args, dataloader_label_test, dataloader_test, target_bin=target_bin, best_model=False)
        test_acc_best, _ = test(args, dataloader_label_test, dataloader_test, target_bin=target_bin, best_model=True)
    else:
        print("Test with CNFN test func")  # 57.02726473175022%
        test_acc = test_with_cnfn_test_func(args, dataloader_label_test, dataloader_test, best_model=False)
        test_acc_best = test_with_cnfn_test_func(args, dataloader_label_test, dataloader_test, best_model=True)
        train_acc_best = test_with_cnfn_test_func(args, dataloader_label_test, dataloader_test, best_model=True,
                                                  best_train=True)

    print("Test data - Accuracy: test {:.4f}, best_test {:.4f}, best_train {:.4f}%, max_train_acc {:.4f}%".
          format(test_acc, test_acc_best, train_acc_best, max_train_acc))

    print("Time elapsed: {:.4f} min".format((timer()-start_time)/60))

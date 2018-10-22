import torch
import numpy as np
from cnnseq import utils
import argparse
from cnnseq.CNNSeq2Seq2_2 import CNNSeq2Seq, train, test
from cnnseq.CNN import (load_model as load_model_cnn, test_with_model as test_cnn)
from cnnseq.utils_models import CLASS, CLASS2D, get_num_classes, load_json


"""
    Model I.A (See models.jpg in assets folder)
    CNN-Seq2Seq for Music generation from Scene and music video emotion score
    Emotion label 1/0 (positive/negative)
    
    Assume pre-trained CNN already exists
    Load CNN model into CNN-Seq2Seq which calls Seq2Seq (CNN fine-tuning happend)
        Parameters are being updated when any of the cost functions are called
        Two Cost functions (emotion with MSELoss and audio with L1Loss)
    
    Author: Gwena Cunha
"""


def get_parser():
    parser = argparse.ArgumentParser(description="Convolutional AutoEncoder")
    parser.add_argument('--cnn_model_path', type=str,
                        default='./cnn_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.83perf/',
                        help='Directory containing CNNs saved model and parameters, pre-trained model')
    parser.add_argument('--cnnseq2seq_model_path',
                        type=str,
                        default='./cnnseq2seq_HSL_bin_1D_res_stepPred_8_ep_2_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/',
                        help='Directory containing CNN-Seq2Seqs saved model and parameters, pre-trained model')
    return parser.parse_args()


def load_data1(data, audio_n_prediction, target_bin=True, data_type='hist', emotion_res='1D', normalize_data=False):
    y, y2D, y2D_raw, audio = data['label'], data['label2D'], data['label2D_raw'], data['audio']
    # X = (splices x num_frames x hist_bins x channels) = (624, 90, 3, 32)
    if 'hist' in data_type:
        X = data['hist_data']
    else:
        X = data['HSL_data']
    # X = (splices x num_frames x hist_bins x channels) = (624, 90, 3, 32)
    print("X: {}, y: {}, y2D: {}, y2D_raw: {}, audio: {}".format(np.shape(X), np.shape(y), np.shape(y2D), np.shape(y2D_raw), np.shape(audio)))

    s = np.shape(X)
    # s = X.shape
    if normalize_data:
        X = utils.normalize_data(X, s)

    if data_type == 'histFlatten':
        att_train = X.reshape(s[0], s[1], s[2] * s[3])
    elif data_type == 'hist':
        att_train = X.reshape(s[0], s[1], s[2], s[3])
    else:
        att_train = X.reshape(s[0], s[1], s[2], s[3], s[4])

    # Target train
    # Expand labels so that each splice frame has a label
    #   Before: 1 video, 38 frames -> 1 label
    #   After expansion: 1 video, 38 frames -> 38 labels
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

        y_train = y_bin
    else:
        y_train = y

    frames_label_train = []
    for a in y_train:
        audio_label_train_exdim = np.expand_dims(a, axis=0)
        frames_label_train.append(utils.expand_array(audio_label_train_exdim, s[1]))

    label_train = utils.expand_array(y_train, X.shape[1])

    # Divide audio into groups of "audio_n_prediction"
    import math
    y_audio_dim = int(math.ceil(np.shape(audio)[1] / audio_n_prediction))
    audio_train = np.zeros([np.shape(audio)[0], audio_n_prediction, y_audio_dim])
    for i in range(0, np.shape(audio_train)[0]):
        k = 0
        j_final = np.shape(audio)[1]  # -audio_n_prediction-1
        for j in range(0, j_final, audio_n_prediction):
            splice_audio = audio[i, j:min(np.shape(audio)[1], j + audio_n_prediction)]
            audio_train[i, :, k] = splice_audio
            k += 1

    return att_train, label_train, frames_label_train, audio_train


def load_data(data_dir, data_filename, audio_n_prediction, target_bin=True, data_type='hist', emotion_res='1D', normalize_data=False):
    """ Loads data, assumes that train and test .npz files already exits (see main.py to create it)
    """
    train_data_path = "{data_dir}{path}_train.npz".format(data_dir=data_dir, path=data_filename)
    test_data_path = "{data_dir}{path}_test.npz".format(data_dir=data_dir, path=data_filename)

    # Load data
    data_train = np.load(train_data_path)
    att_train, label_train, frames_label_train, audio_train = load_data1(data_train, audio_n_prediction,
                                                                         target_bin=target_bin, data_type=data_type,
                                                                         emotion_res=emotion_res, normalize_data=normalize_data)

    data_test = np.load(test_data_path)
    att_test, label_test, frames_label_test, audio_test = load_data1(data_test, audio_n_prediction,
                                                                     target_bin=target_bin, data_type=data_type,
                                                                     emotion_res=emotion_res, normalize_data=normalize_data)

    return att_train, label_train, frames_label_train, audio_train,\
           att_test, label_test, frames_label_test, audio_test


def label_tensor(label):
    # print("Label batch")
    label_shape = np.shape(label)
    # print("label_shape: {}".format(label_shape))
    c = label_shape[0]  # c, a = text_shape
    dataloader_text = {}
    # for i in range(0, c-bs+1, bs):
    for i, l in enumerate(label):
        d_tensor = torch.tensor(l)
        dataloader_text[i] = d_tensor.float()
    # print("dataset label: {}".format(np.shape(dataloader_text.keys())))
    return dataloader_text


def feats_tensor_input(attributes, data_type='hist'):
    # print("Feats tensor")
    dataset = attributes
    dataloader = {}

    # print("np.shape(dataset): {}".format(np.shape(dataset)))
    if data_type == 'histFlatten':
        c, f, a = np.shape(dataset)  # samples_num, n_frames, 96
    elif data_type == 'hist':
        c, f, a, b = np.shape(dataset)  # samples_num, n_frames, 3, 32
    else:
        c, f, a, d1, d2 = np.shape(dataset)  # samples_num, n_frames, 3, 100, 100

    for i, d in enumerate(dataset):
        if data_type == 'histFlatten':
            d_reshape = np.reshape(d, (f, 1, a))
        elif data_type == 'hist':
            d_reshape = np.reshape(d, (f, a, b))
        else:
            d_reshape = np.reshape(d, (f, a, d1, d2))
        d_tensor = torch.tensor(d_reshape)
        dataloader[i] = d_tensor.float()
    return dataloader


def feats_tensor_audio(attributes):
    # print("Feats tensor")
    dataset = attributes
    dataloader = {}

    # print("np.shape(dataset): {}".format(np.shape(dataset)))
    c, b, a = np.shape(dataset)  # d_num, 224

    for i, d in enumerate(dataset):
        # print("Bs: {}-{}, samples: {}, d shape: {}".format(i, idx_fin, idx_fin-i, np.shape(d)))
        d_reshape = np.reshape(d, (1, 1, b, a))
        d_tensor = torch.tensor(d_reshape)
        dataloader[i] = d_tensor.float()
    # print("dataloader keys: {}".format(np.shape(dataloader.keys())))
    return dataloader


def cnn_seq2seq():
    # Settings
    args = get_parser()
    utils.ensure_dir(args.results_dir)

    # Get CNN models information
    # Load json file and get model's information
    cnn_params = load_json(args.cnn_model_path, 'parameters.json')

    # Seq2Seq params
    seq2seq_params = load_json(args.cnnseq2seq_model_path, 'parameters.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained CNN and CNN-Seq2Seq
    cnn_model = load_model_cnn(cnn_params, seq2seq_params['cnn_model_path'], 'cnn_model.pth')


if __name__ == '__main__':
    cnn_seq2seq()

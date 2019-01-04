
import os
import numpy as np
from cnnseq.utils_models import CLASS, CLASS2D
from sklearn.preprocessing.data import QuantileTransformer, MinMaxScaler
from collections import OrderedDict
import pickle
import json


def fix_unexpected_keys_error(pretrained_state):
    new_pretrained_state = OrderedDict()

    for k, v in pretrained_state.items():
        # layer_name = k.replace("model.", "")
        layer_name = k
        if "num_batches_tracked" not in layer_name:
            new_pretrained_state[layer_name] = v
    return new_pretrained_state


def project_dir_name():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    project_dir = os.path.abspath(current_dir + "/../") + "/"
    return project_dir


def ensure_dir(file_path):
    # directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def expand_array(arr, expansion_coefficient):
    # print("expansion_coefficient: {}".format(expansion_coefficient))
    # print("arr shape: {}".format(arr.shape))
    if len(np.shape(arr)) == 1:
        s = (arr.shape[0] * expansion_coefficient)
    else:
        s = (arr.shape[0] * expansion_coefficient, arr.shape[1])
    # print("s: {}".format(s))
    arr_expanded = np.ones(s)
    # print("arr_expanded shape: {}".format(arr_expanded.shape))

    count = 0
    for i in range(arr.shape[0]):
        if len(np.shape(arr)) == 1:
            arr_expanded[count:count + expansion_coefficient] *= arr[i]
        else:
            arr_expanded[count:count + expansion_coefficient, :] *= arr[i, :]
        count += expansion_coefficient

    return arr_expanded


def normalize_data(X, s, verbose=True):
    if verbose:
        print("Normalizing data...")
    trans_norm = QuantileTransformer(output_distribution='normal', subsample=1000)
    trans_uni = MinMaxScaler()
    if verbose:
        print(trans_uni)
    Xtrain_reshaped = X.reshape(s[0], -1)
    if verbose:
        print("Xtrain_reshaped: {}".format(np.shape(Xtrain_reshaped)))
    X_train_norm = trans_norm.fit_transform(Xtrain_reshaped)
    if verbose:
        print("X_train_norm: {}".format(np.shape(X_train_norm)))
    trans_uni.fit(X_train_norm)
    X = trans_uni.transform(X_train_norm).reshape(s)
    if verbose:
        print("X: {}".format(np.shape(X)))
    return X


def normalize(arr, min=0, max=1):
    return min + (max - min) * (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))


# Compatible with CNN-Seq2Seq
# Normalizing data takes too long -> not used
def load_data1(data, audio_n_prediction=1, target_bin=True, data_type='hist', emotion_res='1D', normalize_data=False):
    y, y2D, y2D_raw, X, audio = data['label'], data['label2D'], data['label2D_raw'], data['hist_data'], data['audio']
    # X = (splices x num_frames x hist_bins x channels) = (624, 90, 3, 32)
    print("X: {}, y: {}, y2D: {}, y2D_raw: {}, audio: {}".format(np.shape(X), np.shape(y), np.shape(y2D), np.shape(y2D_raw), np.shape(audio)))
    s = np.shape(X)

    if normalize_data:
        trans_norm = QuantileTransformer(output_distribution='normal')
        trans_uni = MinMaxScaler()
        Xtrain_reshaped = X.reshape([s[0], -1])
        X_train_norm = trans_norm.fit_transform(Xtrain_reshaped)
        trans_uni.fit(X_train_norm)
        X = trans_uni.transform(X_train_norm).reshape(s)

    if data_type == 'histFlatten':
        att_train = X.reshape(s[0], s[1], s[2] * s[3])
    #else:
    #    att_train = X.reshape(s)
    elif data_type == 'hist':
        att_train = X.reshape(s[0], s[1], s[2], s[3])
    else:
        att_train = X.reshape(s[0], s[1], s[2], s[3], s[4])

    if normalize_data:
        print()

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
        frames_label_train.append(expand_array(audio_label_train_exdim, s[1]))

    label_train = expand_array(y_train, X.shape[1])

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


def load_data(data_dir, data_filename, audio_n_prediction=1, target_bin=True, data_type='hist', emotion_res='1D',
              normalize_data=False):
    """ Loads data, assumes that train and test .npz files already exits (see main.py to create it)
    """
    train_data_path = "{data_dir}{path}_train.npz".format(data_dir=data_dir, path=data_filename)
    test_data_path = "{data_dir}{path}_test.npz".format(data_dir=data_dir, path=data_filename)

    # Load data
    data_train = np.load(train_data_path)
    att_train, label_train, frames_label_train, audio_train = load_data1(data_train, audio_n_prediction=audio_n_prediction,
                                                                         target_bin=target_bin, data_type=data_type,
                                                                         emotion_res=emotion_res,
                                                                         normalize_data=normalize_data)

    data_test = np.load(test_data_path)
    att_test, label_test, frames_label_test, audio_test = load_data1(data_test, audio_n_prediction=audio_n_prediction,
                                                                     target_bin=target_bin, data_type=data_type,
                                                                     emotion_res=emotion_res,
                                                                     normalize_data=normalize_data)

    return att_train, label_train, frames_label_train, audio_train,\
           att_test, label_test, frames_label_test, audio_test


def save_feats_pickle(hidden_dict, output_dict, label_dict, save_feats_pickle):
    # Save hidden state in pickle for t-SNE plotting
    with open(save_feats_pickle, 'wb') as f:
        pickle.dump({"feat": hidden_dict, "label": label_dict, "gen_label": output_dict}, f)


def load_json(results_dir, saved_model_parameters):
    # Load model parameters from JSON file, not really needed here
    json_path = os.path.join(results_dir, saved_model_parameters)
    with open(json_path, 'r') as fp:
        # print("JSON file: " + json_path)
        params = json.load(fp)
    return params


def save_json(args):
    # Save args in json file so model can be fully loaded independently
    json_path = os.path.join(args.results_dir, args.saved_model_parameters)
    with open(json_path, 'w') as fp:
        print("Saving model parameters in " + json_path)
        json.dump(vars(args), fp, sort_keys=True, indent=4)

import torch
import numpy as np
from cnnseq import utils
import argparse
from cnnseq.CNNSeq2Seq2 import CNNSeq2Seq, train, test
from cnnseq.CNN import (load_model as load_model_cnn, test_with_model as test_cnn)
from cnnseq.utils import CLASS, CLASS2D, load_json
from timeit import default_timer as timer


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
    # Shared CNN and Seq2Seq hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to save results')
    parser.add_argument('--save_step', type=int, default=1, help='Save results every N steps')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--gpu_num', type=list, default=[0, 1])
    # parser.add_argument('--optimizer_name', type=str, default='asgd',
    parser.add_argument('--optimizer_name', type=str, default='asgd',
                        help='Optimizer: [adam, sgd, asgd, adagrad, adadelta, rmsprop, rprop]')
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 1e-3)
    parser.add_argument('--momentum', type=float, default=0.98)
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # 5e-4
    parser.add_argument('--scheduler_factor', type=float,
                        default=0.8, help='Factor by which the learning rate will be changed after plateauing')
    parser.add_argument('--cost_function_type', type=str, default='audio',
                        help='Which cost functions to consider. Options: [audio, emotion, both].'
                             'audio: L1Loss with seq2seq generated audio'
                             'emotion: MSELoss with CNNs emotion output'
                             'both: audio and emotion (considers two cost functions)')

    # Seq2Seq hyper-parameters
    parser.add_argument('--sequence_length', type=int, default=48000, help='3 seconds at 16 kHz is 48000, 10secs is 160000')
    parser.add_argument('--audio_n_prediction', type=int, default=8, help='Step prediction')
    parser.add_argument('--input_size', type=int, default=30, help='90 frames of video in 3 seconds of audio, 30 if at 10fps')
    parser.add_argument('--output_size', type=int, default=8, help='Same as audio_n_prediction')
    parser.add_argument('--hidden_size', type=int, default=128)  # 128
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--activation_function', type=str, default='relu',
                        help='Activation function: [softmax, sigmoid, softplus, tanh, relu]')
    parser.add_argument('--model_type', type=str, default='seq2seq_gru', help='Options: [seq2seq, seq2seq_gru, lstm]')

    # Data
    parser.add_argument('--train_samples_size', type=int, default=-1,  # 200
                        help='Number of samples used for training, if -1 it means use all')
    parser.add_argument('--test_samples_size', type=int, default=-1,  # 100
                        help='Number of samples used for testing, if -1 it means use all')
    parser.add_argument('--data_dir', type=str,
                        # default='/media/ceslea/DATA/VideoEmotion/DataWithEmotionTags_noText_correctedAudio_hsio/',
                        # default=utils.project_dir_name() + 'data/',
                        # default='/media/root/DATA/PycharmProjects/Scene2Wav/datasets/data_npz/',
                        default='/mnt/Gwena/Scene2Wav/datasets/data_npz/',
                        help='Data directory')
    # parser.add_argument('--data_filename', type=str, default='video_feats_HSLhist_data', help='Data file')
    # parser.add_argument('--data_filename', type=str, default='video_feats_HSLboth_10fps_data_divPerVideo', help='Data file')
    # parser.add_argument('--data_filename', type=str, default='video_feats_HSL_10fps_10secs_intAudio_pad', help='Data file')
    parser.add_argument('--data_filename', type=str, default='video_feats_HSL_10fps_3secs_intAudio_pad',
                        help='Data file')

    # Pre-trained CNN/CNFN
    parser.add_argument('--cnn_model_path', type=str,
                        # default='./results_cnn/res_histFlatten_num_1D_ep_40_bs_90_lr_0.001_we_1e-05_asgd/',
                        # default='./results_cnn/res_HSL_num_1D_ep_40_bs_30_lr_0.001_we_1e-05_asgd_96.195/',
                        # default='./results_cnn/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.83perf/',
                        # default='./results_cnn2/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf/',
                        # default='./results_cnn2_origAudio/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_76.78perf/',
                        # default='./results_cnn2_10secs/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/',
                        # default='./results_cnn2/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf/',
                        # default='./results_cnn3/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/',
                        # default='./results_cnn4/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/',
                        default='./results/cnn2_3secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf/',
                        help='Directory containing CNNs saved model and parameters')

    # Results
    parser.add_argument('--results_dir', type=str, default='./results_cnn_seq2seq2_gru_stepLossPlot/', help='Directory to save results')
    # parser.add_argument('--results_dir', type=str, default='./results_cnn_seq2seq3/HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3182_testSize_1139_cost_audio/', help='Directory to save results')
    # parser.add_argument('--results_dir', type=str, default='./results_cloud_server_test/cnnseq2seq2_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/', help='Directory to save results')
    # parser.add_argument('--results_dir', type=str, default='./results_cnn_seq2seq/histFlatten_num_1D_res_stepPred_8_ep_10_bs_90_relu_layers_2_size_128_lr_0.01_we_1e-05_asgd_trainSize_3883_testSize_431_cost_both/', help='Directory to save results')
    # parser.add_argument('--results_dir', type=str,
    #                    default='./results_cnn_seq2seq/HSL_bin_1D_res_stepPred_8_ep_2_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/',
    #                    help='Directory to save results')
    parser.add_argument('--log_filename', type=str, default='log.txt', help='Log text file')
    parser.add_argument('--saved_model', type=str, default='seq2seq_model.pth', help='Path to save model')
    parser.add_argument('--saved_model_best', type=str, default='seq2seq_model_best.pth', help='Path to save model')
    parser.add_argument('--saved_model_parameters', type=str, default='seq2seq_parameters.json',
                        help='Path to save models parameters')
    parser.add_argument('--saved_model_cnn', type=str, default='cnn_model.pth', help='Path to save model')
    parser.add_argument('--saved_model_cnn_best', type=str, default='cnn_model_best.pth', help='Path to save model')
    parser.add_argument('--saved_model_parameters_cnn', type=str, default='cnn_parameters.json',
                        help='Path to save models parameters')

    # parser.add_argument('--save_ext', type=str, default='npz', help='Method of Options: [npz, json, pkl]')
    parser.add_argument('--mode', type=str, default='train', help='Options: [train, test]')

    return parser.parse_args()


def load_data1(data, audio_n_prediction, target_bin=True, data_type='hist', emotion_res='1D', normalize_data=False):
    if 'label' in data:
        y, y2D, y2D_raw, audio = data['label'], data['label2D'], data['label2D_raw'], data['audio']
    else:
        y, y2D, y2D_raw, audio = data['emotion'], data['emotion'], data['emotion'], data['audio']
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
    audio_train = np.reshape(audio, [np.shape(audio)[0], audio_n_prediction, y_audio_dim])
    '''
    audio_train = np.zeros([np.shape(audio)[0], audio_n_prediction, y_audio_dim])
    for i in range(0, np.shape(audio_train)[0]):
        k = 0
        j_final = np.shape(audio)[1]  # -audio_n_prediction-1
        for j in range(0, j_final, audio_n_prediction):
            splice_audio = audio[i, j:min(np.shape(audio)[1], j + audio_n_prediction)]
            audio_train[i, :, k] = splice_audio
            k += 1
    '''
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


if __name__ == '__main__':
    # Timer
    tic = timer()

    # Settings
    args = get_parser()
    utils.ensure_dir(args.results_dir)

    # Get CNN models information
    # Load json file and get model's information
    cnn_params = load_json(args.cnn_model_path, 'parameters.json')

    # Change this
    # args.batch_size = cnn_params['batch_size']
    # args.data_filename = cnn_params['data_filename']
    # args.data_dir = cnn_params['data_dir']

    # Seq2Seq params
    seq2seq_params = vars(args)
    seq2seq_params['target_bin'] = cnn_params['target_bin']
    seq2seq_params['data_type'] = cnn_params['data_type']
    seq2seq_params['emotion_res'] = cnn_params['emotion_res']
    seq2seq_params['num_classes'] = cnn_params['num_classes']

    # Load data, separate into test and train and return
    print("\nLoad data...")
    target_bin = True if seq2seq_params['target_bin'] == 'bin' else False
    att_train, label_train, frames_label_train, audio_train, att_test, label_test, frames_label_test, audio_test =\
        load_data(seq2seq_params['data_dir'], seq2seq_params['data_filename'], seq2seq_params['audio_n_prediction'],
                  target_bin=target_bin, data_type=seq2seq_params['data_type'],
                  emotion_res=seq2seq_params['emotion_res'], normalize_data=False)
    print("\nLoaded data - train: {}, {}, {} - test: {}, {}, {}".format(np.shape(att_train), np.shape(label_train),
                                                                        np.shape(frames_label_train), np.shape(audio_train),
                                                                        np.shape(att_test), np.shape(label_test),
                                                                        np.shape(frames_label_test), np.shape(audio_test)))

    # Dataset: (num_lines, num_words_embedded, size_embedding)
    if args.train_samples_size == -1:
        args.train_samples_size = np.shape(audio_train)[0]
    if args.test_samples_size == -1:
        args.test_samples_size = np.shape(audio_test)[0]

    dataloader_train = feats_tensor_input(att_train[0:args.train_samples_size], data_type=seq2seq_params['data_type'])
    dataloader_label_train = label_tensor(frames_label_train[0:args.train_samples_size])
    dataloader_audio_train = feats_tensor_audio(audio_train[0:args.train_samples_size])

    dataloader_test = feats_tensor_input(att_test[0:args.test_samples_size], data_type=seq2seq_params['data_type'])
    dataloader_label_test = label_tensor(frames_label_test[0:args.test_samples_size])
    dataloader_audio_test = feats_tensor_audio(audio_test[0:args.test_samples_size])

    print("Tensor shape train - label: {}, att: {}, audio: {}".format(np.shape(dataloader_label_train.keys()),
                                                                      np.shape(dataloader_train.keys()),
                                                                      np.shape(dataloader_audio_train.keys())))
    print("Tensor shape test - label: {}, att: {}, audio: {}".format(np.shape(dataloader_label_test.keys()),
                                                                     np.shape(dataloader_test.keys()),
                                                                     np.shape(dataloader_audio_test.keys())))
    # for i, (l, a, d) in enumerate(zip(dataloader_label_train.keys(), dataloader_audio_train.keys(), dataloader_train.keys())):
    #    print("[{}] l: {}, a: {}, d: {}".format(i, np.shape(dataloader_label_train[l]),
    #                                            np.shape(dataloader_audio_train[a]),
    #                                            np.shape(dataloader_train[d])))

    print("num_classes: {}".format(seq2seq_params['num_classes']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained CNN
    cnn_model = load_model_cnn(cnn_params, seq2seq_params['cnn_model_path'], 'cnn_model.pth')

    acc, euc_dist_mean = test_cnn(cnn_model, dataloader_label_test, dataloader_test, target_bin=target_bin)

    # Train CNN-Seq2Seq
    if seq2seq_params['mode'] is 'train':
        print("\nTraining CNN-Seq2Seq model...")
        cnn_seq2seq_model = CNNSeq2Seq(seq2seq_params, cnn_model, device).cuda(args.gpu_num[0])

        # params2 = cnn_seq2seq_model.named_parameters()
        # for name1, param1 in params2:
        #     print("name: {}".format(name1))

        lab, img, res_dir = train(cnn_seq2seq_model, seq2seq_params, dataloader_label_train, dataloader_train,
                                  dataloader_audio_train, dataloader_label_test, dataloader_test,
                                  dataloader_audio_test, target_bin=target_bin)

    # Test model with test data
    print("\nTesting CNN-Seq2Seq model...")
    print("\nTest data before fine tuning - CNN accuracy: {:.2f}% and euclidian distance: {:.2f}".
          format(acc, euc_dist_mean))
    acc_fine_tuning, euc_dist_mean_fine_tuning, p_mean, r_mean = test(seq2seq_params, dataloader_label_test,
                                                                      dataloader_test, dataloader_audio_test,
                                                                      target_bin=target_bin, device=torch.device('cuda'))
    print("\nTest data after fine tuning - CNN accuracy: {:.2f}% and euclidian distance: {:.2f} - CNNSeq2Seq p {:.4f} and r {:.4f}".
          format(acc_fine_tuning, euc_dist_mean_fine_tuning, p_mean, r_mean))

    acc_fine_tuning, euc_dist_mean_fine_tuning, p_mean, r_mean = test(seq2seq_params, dataloader_label_train,
                                                                      dataloader_train, dataloader_audio_train,
                                                                      target_bin=target_bin, device=torch.device('cuda'), use_best_checkpoint=True)
    print("\nTest data after fine tuning (BEST CKPT) - CNN accuracy: {:.2f}% and euclidian distance: {:.2f} - CNNSeq2Seq p {:.4f} and r {:.4f}".
          format(acc_fine_tuning, euc_dist_mean_fine_tuning, p_mean, r_mean))

    #print("\nGenerate data after fine tuning (BEST CKPT) - without target")
    #generate(seq2seq_params, dataloader_test, device=torch.device('cuda'), use_best_checkpoint=True)

    toc = timer()
    hours = (toc - tic)/3600
    print("Code ran for {} seconds, or {} hours".format(toc-tic, hours))

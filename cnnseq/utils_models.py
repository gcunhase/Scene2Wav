import torch
import numpy as np
import os
import json


CLASS = {0: 'Negative', 1: 'Positive'}
CLASS2 = {'Negative': 0, 'Positive': 1}
CLASS2D = {0: 'PositiveHigh', 1: 'PositiveLow', 2: 'NegativeHigh', 3: 'NegativeLow'}


def flatten_audio(samples, args):
    audio = np.zeros([1, args.sequence_length])
    for k, j in enumerate(range(0, args.sequence_length, args.audio_n_prediction)):
        j_final = min(args.sequence_length, j + args.audio_n_prediction)
        s_sample = samples[0:j_final-j, k]
        audio[0, j:j_final] = s_sample
    return audio


def flatten_audio_with_params(samples, seq_length, audio_n_pred):
    audio = np.reshape(samples, [-1, seq_length])
    #audio = np.zeros([1, seq_length])
    #for k, j in enumerate(range(0, seq_length, audio_n_pred)):
    #    j_final = min(seq_length, j + audio_n_pred)
    #    s_sample = samples[0:j_final-j, k]
    #    audio[0, j:j_final] = s_sample
    return audio


def get_num_classes(target_bin, emotion_res):
    if not target_bin:
        num_classes = 1
    else:
        if emotion_res == '1D':
            num_classes = 2
        else:
            num_classes = 4
    return num_classes


def set_optimizer(model, args, partial_name=None):
    optimizer_name = args.optimizer_name
    if partial_name is None:
        partial_name = 'res_stepPred_{}_ep_{}_bs_{}_{}_layers_{}_size_{}_lr_{}'.\
            format(args.audio_n_prediction, args.num_epochs, args.batch_size, args.activation_function, args.num_layers,
                   args.hidden_size, args.learning_rate)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        partial_name += '_we_{}_{}'.format(str(args.weight_decay), optimizer_name)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        partial_name += '_we_{}_mo_{}_{}'.format(args.weight_decay, str(args.momentum), optimizer_name)
    elif optimizer_name == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        partial_name += '_we_{}_{}'.format(args.weight_decay, optimizer_name)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, lr_decay=0.001,
                                        weight_decay=args.weight_decay)
        partial_name += '_we_{}_{}'.format(str(args.weight_decay), optimizer_name)
    elif optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        partial_name += '_we_{}_{}'.format(str(args.weight_decay), optimizer_name)
    elif optimizer_name == 'rprop':  # very slow
        optimizer = torch.optim.Rprop(model.parameters(), lr=args.learning_rate)
        partial_name += '_{}'.format(optimizer_name)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        partial_name += '_we_{}_{}'.format(str(args.weight_decay), optimizer_name)

    return optimizer, partial_name


def set_optimizer_parameters(model_parameters, params, partial_name=None):
    optimizer_name = params['optimizer_name']
    lr = params['learning_rate']
    weight_decay = params['weight_decay']
    if partial_name is None:
        partial_name = 'res_stepPred_{}_ep_{}_bs_{}_{}_layers_{}_size_{}_lr_{}'.\
            format(params['audio_n_prediction'], params['num_epochs'], params['batch_size'],
                   params['activation_function'], params['num_layers'], params['hidden_size'], lr)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
        partial_name += '_we_{}_{}'.format(weight_decay, optimizer_name)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=params['momentum'],
                                    weight_decay=weight_decay)
        partial_name += '_we_{}_mo_{}_{}'.format(weight_decay, params['momentum'], optimizer_name)
    elif optimizer_name == 'asgd':
        optimizer = torch.optim.ASGD(model_parameters, lr=lr, weight_decay=weight_decay)
        partial_name += '_we_{}_{}'.format(weight_decay, optimizer_name)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model_parameters, lr=lr, lr_decay=0.001,
                                        weight_decay=weight_decay)
        partial_name += '_we_{}_{}'.format(weight_decay, optimizer_name)
    elif optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(model_parameters, lr=lr, weight_decay=weight_decay)
        partial_name += '_we_{}_{}'.format(weight_decay, optimizer_name)
    elif optimizer_name == 'rprop':  # very slow
        optimizer = torch.optim.Rprop(model_parameters, lr=lr)
        partial_name += '_{}'.format(optimizer_name)
    else:
        optimizer = torch.optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)
        partial_name += '_we_{}_{}'.format(weight_decay, optimizer_name)

    return optimizer, partial_name


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


def save_json_with_params(params):
    # Save args in json file so model can be fully loaded independently
    json_path = os.path.join(params['results_dir'], params['saved_model_parameters'])
    with open(json_path, 'w') as fp:
        print("Saving model parameters in " + json_path)
        json.dump(params, fp, sort_keys=True, indent=4)

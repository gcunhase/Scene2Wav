import torch
import torch.nn as nn
import numpy as np
from cnnseq import utils
from cnnseq.utils_models import set_optimizer, flatten_audio
from skimage.io import imsave
import json
import os


# Recurrent neural network (many-to-one)
class Seq2Seq(nn.Module):
    def __init__(self, params, device=torch.device('cpu')):
        super(Seq2Seq, self).__init__()
        self.params = params
        input_size = self.params['input_size']
        output_size = self.params['output_size']
        hidden_size = self.params['hidden_size']
        num_layers = self.params['num_layers']
        activation_function = self.params['activation_function']
        self.model_type = self.params['model_type']
        self.device = device
        self.hidden_size = self.params['hidden_size']
        self.num_layers = self.params['num_layers']
        self.lstm_enc = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        if self.model_type == 'seq2seq':
            self.lstm_dec = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
            self.decoder_linear = nn.Linear(hidden_size, output_size)
        else:
            self.decoder_linear = nn.Linear(hidden_size, input_size)

        # self.softmax = nn.LogSoftmax(dim=1)
        if activation_function == 'softmax':
            self.softmax = nn.Softmax(dim=num_layers)
        elif activation_function == 'sigmoid':
            self.softmax = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.softmax = nn.Tanh()
        elif activation_function =='relu':
            self.softmax = nn.ReLU()
        else:
            self.softmax = nn.Softplus()

    def forward(self, x, y):
        # print("x: {}".format(np.shape(x)))

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Encoder: Forward propagate LSTM
        out, hidden = self.lstm_enc(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # print("Enc shapes: out {}, hidden: {}, h0: {}, c0: {}".format(np.shape(out), np.shape(hidden), np.shape(h0), np.shape(c0)))

        # Decoder: Decode the hidden state of the last time step
        if self.model_type == 'seq2seq':
            out, _ = self.lstm_dec(y, hidden)
            # print("Decoder: {}".format(np.shape(out)))
        out = self.softmax(self.decoder_linear(out))
        # print("Out: {}".format(np.shape(out)))

        return out, hidden


def train(model, dataloader_train, dataloader_label_train, args, device):
    # Loss and optimizer
    criterion = nn.L1Loss()  # SmoothL1Loss, NLLLoss(), CrossEntropyLoss()
    optimizer, partial_name = set_optimizer(model, args)

    # New results dir based on model's parameters
    res_dir = args.results_dir + '{}_trainSize_{}_testSize_{}/'.format(partial_name, args.train_samples_size,
                                                                       args.test_samples_size)
    args.results_dir = res_dir
    utils.ensure_dir(res_dir)
    # print("res_dir: {}".format(res_dir))
    log_file = open(res_dir + 'log.txt', 'w')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor,
                                                           verbose=True)

    total_step = len(dataloader_train.keys())
    loss_arr = []
    epoch_arr = []
    for epoch in range(args.num_epochs):
        for i, (im, la) in enumerate(zip(dataloader_train.keys(), dataloader_label_train.keys())):
            labels = dataloader_label_train[la]
            images = dataloader_train[im]

            # print("Shape images: {}, labels: {}".format(np.shape(images), np.shape(labels)))

            images = images.reshape(-1, np.shape(images)[-1], args.input_size).to(device)  # bsx28x28
            labels = labels.reshape(-1, np.shape(labels)[-1], args.output_size).to(device)  # labels.to(device)

            # print("Shape after reshape images: {}, labels: {}".format(np.shape(images), np.shape(labels)))

            # Forward pass
            target = labels  # images  # 1-images
            outputs, _ = model(images, target)  # (2, 96, 90), (2, 6000, 8)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                log_str = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs,
                                                                             i + 1, total_step, loss.item())
                print(log_str)
                log_file.write(log_str + '\n')

        scheduler.step(loss)
        loss_arr.append(loss.item())
        epoch_arr.append(epoch + 1)
        if (epoch + 1) % args.save_step == 0:

            # Input images
            input_data = images.cpu().data.numpy()[0]
            input_reshaped = np.reshape(input_data, [np.shape(input_data)[1], np.shape(input_data)[0]])

            # Target audio
            images_white_data = target.cpu().data.numpy()[0]
            im_reshaped = np.reshape(images_white_data,
                                     [np.shape(images_white_data)[1], np.shape(images_white_data)[0]])
            im_reshaped = flatten_audio(im_reshaped, args)

            # Generated audio
            outputs_data = outputs.cpu().data.numpy()[0]
            out_reshaped = np.reshape(outputs_data, [np.shape(outputs_data)[1], np.shape(outputs_data)[0]])
            out_reshaped = flatten_audio(out_reshaped, args)

            # Save audio, 16KHz
            from scipy.io.wavfile import write
            scaled = -1.0 + (1.0 - (-1.0)) * (input_reshaped - np.min(input_reshaped)) / (
                    np.max(input_reshaped) - np.min(input_reshaped))
            imsave('{}{}_input.jpg'.format(res_dir, epoch + 1), scaled)

            scaled = -1.0 + (1.0 - (-1.0)) * (im_reshaped - np.min(im_reshaped)) / (
                    np.max(im_reshaped) - np.min(im_reshaped))
            scaled = np.int16(scaled / np.max(np.abs(scaled)) * 32767)
            write('{}{}_target.wav'.format(res_dir, epoch + 1), 16000, scaled[0])

            scaled2 = np.int16(out_reshaped / np.max(np.abs(out_reshaped)) * 32767)
            write('{}{}_gen.wav'.format(res_dir, epoch + 1), 16000, scaled2[0])
            # imsave('{}{}_target.jpg'.format(res_dir, epoch + 1), images_white_data[0])
            # imsave('{}{}_gen.jpg'.format(res_dir, epoch + 1), outputs_data[0])

    # Save the model checkpoint
    torch.save(model.state_dict(), res_dir + args.saved_model)

    # Plot loss_epochs.svg file
    import matplotlib.pyplot as plt
    plt.figure(figsize=[6, 6])
    plt.plot(epoch_arr, loss_arr, '*-')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('on')
    # plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("{}loss_epochs.svg".format(res_dir))
    plt.cla()

    # Save args in json file so model can be fully loaded independently
    with open(os.path.join(res_dir, args.saved_model_parameters), 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    log_file.close()
    return res_dir


def test(dataloader_test, dataloader_label_test, args, device):
    # Test the model
    print("\nTesting model")

    # Load model parameters from JSON file
    with open(os.path.join(args.results_dir, args.saved_model_parameters), 'r') as fp:
        print(os.path.join(args.results_dir, args.saved_model_parameters))
        params = json.load(fp)

    # model = CNNSeq2Seq(args.input_size, args.output_size, args.hidden_size, args.num_layers).to(device)
    model = Seq2Seq(params, device).to(device)
    model_path = args.results_dir + args.saved_model
    print(model_path)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        correct = 0
        total = 0
        for im, la in zip(dataloader_test.keys(), dataloader_label_test.keys()):
            labels = dataloader_label_test[la]
            images = dataloader_test[im]
            images = images.reshape(-1, np.shape(images)[-1], params['input_size']).to(device)
            labels = labels.reshape(-1, np.shape(labels)[-1], params['output_size']).to(device)
            target = labels  # 1 - images
            outputs, _ = model(images, target)

            input_data_tmp = images.cpu().data.numpy()
            images_white_data_tmp = target.cpu().data.numpy()
            outputs_data_tmp = outputs.cpu().data.numpy()
            print(np.size(images_white_data_tmp))
            for i in range(0, min(2, params['batch_size'])):
                input_data = input_data_tmp[i]
                images_white_data = images_white_data_tmp[i]
                outputs_data = outputs_data_tmp[i]
                im_reshaped = np.reshape(images_white_data,
                                         [np.shape(images_white_data)[1], np.shape(images_white_data)[0]])
                im_reshaped = flatten_audio(im_reshaped, args)
                input_reshaped = np.reshape(input_data, [np.shape(input_data)[1], np.shape(input_data)[0]])
                out_reshaped = np.reshape(outputs_data, [np.shape(outputs_data)[1], np.shape(outputs_data)[0]])
                out_reshaped = flatten_audio(out_reshaped, args)
                print(np.shape(out_reshaped))

                # Save audio, 16KHz
                from scipy.io.wavfile import write

                scaled = -1.0 + (1.0 - (-1.0)) * (input_reshaped - np.min(input_reshaped)) / (
                        np.max(input_reshaped) - np.min(input_reshaped))
                imsave('{}test_input.jpg'.format(args.results_dir), scaled)

                scaled = -1.0 + (1.0 - (-1.0)) * (im_reshaped - np.min(im_reshaped)) / (
                        np.max(im_reshaped) - np.min(im_reshaped))
                scaled = np.int16(scaled / np.max(np.abs(scaled)) * 32767)
                write('{}test_target.wav'.format(args.results_dir), 16000, scaled[0])

                scaled = np.int16(out_reshaped / np.max(np.abs(out_reshaped)) * 32767)
                write('{}test_gen_{}.wav'.format(args.results_dir, i), 16000, scaled[0])

            break

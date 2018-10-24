import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage.io import imsave
from cnnseq import utils
import numpy as np
import json
import os

from cnnseq.utils_models import set_optimizer_parameters, flatten_audio_with_params, load_json, save_json_with_params
from cnnseq.Seq2Seq import Seq2Seq
from cnnseq.CNN import load_model as load_model_cnn
import matplotlib
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip

SOS_token = 10
#EOS_token = -10

# Convolutional neural network (two convolutional layers) for HSL (3 x 100 x 100)
class CNNSeq2Seq(nn.Module):
    def __init__(self, params, cnn_model, device=torch.device('cpu')):
        super(CNNSeq2Seq, self).__init__()

        self.params = params
        self.cnn_model = cnn_model

        # histFlatten
        self.fc = nn.Linear(144, 96)

        self.seq2seq_model = Seq2Seq(params, device=device).cuda(self.params['gpu_num'][0])

    def forward(self, x, y):
        # print("X: {}".format(x.shape))
        cnn_out, hidden = self.cnn_model(x)
        # print("CNN layer: {}".format(cnn_out.shape))
        # hidden = hidden.reshape(hidden.size(0), -1)
        # print("Reshape hidden: {}".format(hidden.shape))
        # hidden = self.fc(hidden)
        # print("Reshape hidden: {}".format(hidden.shape))
        # hidden = hidden.cpu().numpy()
        hidden = hidden.reshape(-1, np.shape(hidden)[-1], self.params['input_size'])
        hidden = hidden.to(self.params['gpu_num'][0])
        # print("Reshape hidden for Seq2Seq: {}".format(hidden.shape))
        out, hidden_enc = self.seq2seq_model(hidden, y)
        # print("Seq2Seq layer: {}".format(out.shape))
        return out, self.cnn_model, cnn_out, hidden_enc


def train(model, params, dataloader_label_train, dataloader_train, dataloader_audio_train,
          dataloader_label_test, dataloader_test, dataloader_audio_test, target_bin=True):

    cost_function_type = params['cost_function_type']

    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()

    optimizer, partial_name = set_optimizer_parameters(model.parameters(), params)
    # optimizer, partial_name = set_optimizer_parameters(cnn_model.parameters() + seq2seq_model.parameters())

    res_dir = params['results_dir'] + '{}_{}_{}_{}_trainSize_{}_testSize_{}_cost_{}/'.\
        format(params['data_type'], params['target_bin'], params['emotion_res'], partial_name,
               params['train_samples_size'], params['test_samples_size'], params['cost_function_type'])

    params['results_dir'] = res_dir
    utils.ensure_dir(params['results_dir'])

    # Save args in json file so model can be fully loaded independently
    save_json_with_params(params)
    log_file = open(res_dir + params['log_filename'], 'w')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['scheduler_factor'],
                                                           verbose=True)
    # Train
    ep_min_loss = 0
    # ep_max_acc = 0
    first_iter = True
    loss_arr = []
    epoch_arr = []
    max_num_epochs = params['num_epochs']
    for epoch in range(max_num_epochs):
        for d, l, a in zip(dataloader_train.keys(), dataloader_label_train.keys(), dataloader_audio_train.keys()):
            label = dataloader_label_train[l]
            data = dataloader_train[d]
            audio = dataloader_audio_train[a]
            img = data
            lab = label
            lab_audio = audio
            target = audio.reshape(-1, np.shape(audio)[-1], params['output_size']).to(params['gpu_num'][0])  # labels.to(device)
            with torch.no_grad():
                img = Variable(img).cuda(params['gpu_num'][0])
                lab = Variable(lab).cuda(params['gpu_num'][0])
            # print(np.shape(img))
            # ===================forward=====================
            #out, cnn_model, cnn_out, hidden_enc = model(img, [[SOS_token]])  # target
            out, cnn_model, cnn_out, hidden_enc = model(img, target)  # np.array([[SOS_token]])
            if not target_bin:
                cnn_out = cnn_out.squeeze()
            # Invert hidden dimensions (bs, 90, 96) -> (bs, 96, 90), where 90 is number of frames in 30 fps
            # images = hidden.reshape(-1, np.shape(hidden)[-1], args.input_size).cuda(args.gpu_num[0])  # bsx28x28
            # output = seq2seq_model(images, target)
            # print(np.shape(out))
            loss1 = criterion2(out, target)
            loss2 = criterion(cnn_out, lab)
            if cost_function_type == 'audio':
                loss = loss1
            elif cost_function_type == 'emotion':
                loss = loss2
            else:
                loss = loss1 + loss2
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            # scheduler.step(loss)  # Test use of scheduler to change lr after it plateaus
            optimizer.step()

        scheduler.step(loss)
        loss_arr.append(loss.item())
        epoch_arr.append(epoch + 1)

        # Save steps
        if epoch % params['save_step'] == 0:
            # ===================log========================
            # Save model
            torch.save(model.state_dict(), res_dir + params['saved_model'])
            torch.save(cnn_model.state_dict(), res_dir + params['saved_model_cnn'])

            #last_test_label = {'0': lab.cpu()}
            #last_test_img = {'0': img.cpu()}
            #last_test_audio = {'0': lab_audio.cpu()}
            #acc, euc_dist_mean = test_with_model(model, last_test_label, last_test_img, last_test_audio, args,
            #                                     target_bin=target_bin, epoch=epoch+1)
            acc, euc_dist_mean = test_with_model(model, dataloader_label_test, dataloader_test, dataloader_audio_test,
                                                 params, target_bin=target_bin, epoch=epoch+1)

            if first_iter:
                min_loss = loss.item()
                max_acc = acc
                torch.save(model.state_dict(), res_dir + params['saved_model_best'])
                torch.save(cnn_model.state_dict(), res_dir + params['saved_model_cnn_best'])
                first_iter = False
            if loss.item() < min_loss:  # early stop (best loss), should be with validation set
                torch.save(model.state_dict(), res_dir + params['saved_model_best'])
                min_loss = loss.item()
                ep_min_loss = epoch + 1
            # if acc > max_acc:
                torch.save(cnn_model.state_dict(), res_dir + params['saved_model_cnn_best'])
                # max_acc = acc
                # ep_max_acc = epoch + 1

            log_str = 'Epoch [{:.2f}%][{}/{}], Total loss: {:.4f}, Seq2Seq loss: {:.4f}, CNN [loss: {}, acc: {}, euc: {}]'.\
                format(100*(epoch + 1)/max_num_epochs, epoch + 1, max_num_epochs, loss.item(), loss1.item(),
                       loss2.item(), acc, euc_dist_mean)
            print(log_str)
            log_file.write(log_str + '\n')

    # Save model
    torch.save(model.state_dict(), res_dir + params['saved_model'])
    torch.save(cnn_model.state_dict(), res_dir + params['saved_model_cnn'])

    # Save log file
    best_model_str = "\nModel: ep {}, min loss: {}, CNN acc: {:.2f}\n".format(ep_min_loss, min_loss, acc)
    log_file.write(best_model_str)
    log_file.close()

    # Save training loss
    plt.figure(figsize=[6, 6])
    plt.plot(epoch_arr, loss_arr, '*-')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('on')
    plt.savefig("{}training_loss.svg".format(res_dir))
    plt.cla()

    # Get weights
    # print("\nModel keys: {}".format(model.state_dict().keys()))
    return lab, img, res_dir


def test_with_model(model, dataloader_label_test, dataloader_test, dataloader_audio_test, params, target_bin=True, epoch=0, verbose=False):
    correct_samples_cnn = 0
    euc_dist_cnn = 0
    num_samples = 0
    test_size = len(dataloader_test.keys())
    res_dir = params['results_dir']
    number_samples = 0
    max_num_samples = 2
    for bs_test, (d, l, a) in enumerate(zip(dataloader_test.keys(), dataloader_label_test.keys(), dataloader_audio_test.keys())):
        label = dataloader_label_test[l]
        data = dataloader_test[d]
        audio = dataloader_audio_test[a]
        # print("Shapes1 - label: {}, data: {}".format(np.shape(label), np.shape(data)))
        img = data
        lab = label
        lab_audio = audio
        target = audio.reshape(-1, np.shape(audio)[-1], params['output_size']).to(params['gpu_num'][0])  # labels.to(device)
        with torch.no_grad():
            img = Variable(img).cuda()
        #out, cnn_model, cnn_out, hidden_enc = model(img, np.array([[SOS_token]]))  # target
        out, cnn_model, cnn_out, hidden_enc = model(img, target)  # target
        if not target_bin:
            cnn_out = cnn_out.squeeze()
        # print("Shapes3 - label: {}, out: {}, img: {}".format(np.shape(lab), np.shape(output), np.shape(img)))

        input_data_tmp = img.cpu().data.numpy()
        target_data_tmp = target.squeeze().cpu().data.numpy()
        outputs_data_tmp = out.squeeze().cpu().data.numpy()
        #print("input_data_tmp: {}, target_data_tmp: {}, outputs_data_tmp: {}".
        #      format(np.shape(input_data_tmp), np.shape(target_data_tmp), np.shape(outputs_data_tmp)))

        # input_reshaped = np.reshape(input_data, [np.shape(input_data)[1], np.shape(input_data)[0]])
        input_reshaped = np.reshape(input_data_tmp, [-1, np.shape(input_data_tmp)[2], np.shape(input_data_tmp)[3],
                                    np.shape(input_data_tmp)[1]])

        target_reshaped = flatten_audio_with_params(target_data_tmp, params['sequence_length'],
                                                    params['audio_n_prediction'])

        out_reshaped = flatten_audio_with_params(outputs_data_tmp, params['sequence_length'],
                                                 params['audio_n_prediction'])

        #print("input_reshaped: {}, out_reshaped: {}, target_reshaped: {}".
        #      format(np.shape(input_reshaped), np.shape(out_reshaped), np.shape(target_reshaped)))

        if number_samples < max_num_samples:
            # Save audio, 16KHz
            from scipy.io.wavfile import write
            scaled = -1.0 + (1.0 - (-1.0)) * (target_reshaped - np.amin(target_reshaped)) / (
                np.amax(target_reshaped) - np.amin(target_reshaped))
            test_target_audio_scaled = np.int16(scaled / np.max(np.abs(scaled)) * 32767)
            write('{}test_target_ep{}_s{}.wav'.format(res_dir, epoch, number_samples), 16000, test_target_audio_scaled[0])

            test_gen_audio_scaled = np.int16(out_reshaped / np.max(np.abs(out_reshaped)) * 32767)
            write('{}test_gen_ep{}_s{}.wav'.format(res_dir, epoch, number_samples), 16000, test_gen_audio_scaled[0])

            # TODO: save with target and generated audio (RGB and HSV versions, total 4 videos)
            # TODO: save as video
            #scaled = -1.0 + (1.0 - (-1.0)) * (input_reshaped - np.amin(input_reshaped)) / (
            #    np.amax(input_reshaped) - np.amin(input_reshaped))
            scaled = input_reshaped

            from skimage import color
            frame_arr = []
            frame_arr2 = []
            for frame in scaled:
                frame = utils.normalize(frame, min=-1, max=1)
                frame_rgb = color.hsv2rgb(frame)
                frame_arr.append(frame_rgb * 255)
                frame_arr2.append(frame_rgb)
            print("frame_arr: {}, min: {}, max: {}".
                  format(np.shape(frame_arr), np.amin(frame_arr), np.amax(frame_arr)))
            print("frame_arr2: {}, min: {}, max: {}".
                  format(np.shape(frame_arr2), np.amin(frame_arr2), np.amax(frame_arr2)))
            clip = ImageSequenceClip(np.array(frame_arr), fps=10)  # 3-second clip, .tolist()
            clip.set_audio(AudioArrayClip(test_target_audio_scaled, fps=16000))
            clip.write_videofile('{}test_rgbInputWithTargetAudio_ep{}_s{}.mp4'.format(res_dir, epoch, number_samples),
                                 fps=10)  # export as video
            # .set_audio(AudioClip)
            # TODO: transform from HSV to RGB and save it as video
            # img_rgb = matplotlib.colors.hsv_to_rgb(scaled)
            # clip = ImageSequenceClip(np.array(img_rgb), fps=10)  # 3-second clip, .tolist()
            # clip.set_audio(AudioArrayClip(test_target_audio_scaled, fps=16000))
            # clip.write_videofile('{}test_rgbInputWithTargetAudio2_ep{}_s{}.mp4'.format(res_dir, epoch, number_samples),
            #                      fps=10)  # export as video

            number_samples += 1

        '''
        for i in range(0, min(2, np.shape(target_data_tmp)[0])):
            input_data = input_data_tmp[i]
            target_data = target_data_tmp[i]
            outputs_data = outputs_data_tmp[i]
            print("input_data: {}, target_data: {}, outputs_data: {}".
                  format(np.shape(input_data), np.shape(target_data), np.shape(outputs_data)))

            # input_reshaped = np.reshape(input_data, [np.shape(input_data)[1], np.shape(input_data)[0]])
            input_reshaped = np.reshape(input_data, [np.shape(input_data)[1], np.shape(input_data)[2],
                                                     np.shape(input_data)[0]])

            target_reshaped = np.reshape(target_data, [np.shape(target_data)[1], np.shape(target_data)[0]])
            target_reshaped = flatten_audio_with_params(target_reshaped, params['sequence_length'],
                                                        params['audio_n_prediction'])

            out_reshaped = np.reshape(outputs_data, [np.shape(outputs_data)[1], np.shape(outputs_data)[0]])
            out_reshaped = flatten_audio_with_params(out_reshaped, params['sequence_length'],
                                                     params['audio_n_prediction'])

            # print(np.shape(out_reshaped))
            print("input_reshaped: {}, out_reshaped: {}, target_reshaped: {}".
                  format(np.shape(input_reshaped), np.shape(out_reshaped), np.shape(target_reshaped)))

            # Save audio, 16KHz
            from scipy.io.wavfile import write
            scaled = -1.0 + (1.0 - (-1.0)) * (target_reshaped - np.min(target_reshaped)) / (
                np.max(target_reshaped) - np.min(target_reshaped))
            test_target_audio_scaled = np.int16(scaled / np.max(np.abs(scaled)) * 32767)
            write('{}test_target_ep{}_{}.wav'.format(res_dir, epoch, i), 16000, test_target_audio_scaled[0])

            test_gen_audio_scaled = np.int16(out_reshaped / np.max(np.abs(out_reshaped)) * 32767)
            write('{}test_gen_ep{}_{}.wav'.format(res_dir, epoch, i), 16000, test_gen_audio_scaled[0])

            # TODO: save with target and generated audio (RGB and HSV versions, total 4 videos)
            # TODO: save as video
            scaled = -1.0 + (1.0 - (-1.0)) * (input_reshaped - np.amin(input_reshaped)) / (
                    np.amax(input_reshaped) - np.amin(input_reshaped))
            # imsave('{}test_input_ep{}_{}.jpg'.format(res_dir, epoch, i), scaled)
            def make_frame_rgb(t):
                """ returns a numpy array of the frame at time t """
                frame_for_time_t = scaled[t, :, :, :]
                return frame_for_time_t
            clip = VideoClip(make_frame_rgb, duration=3)  # 3-second clip
            # clip.set_audio(AudioFileClip(test_target_audio_scaled, fps=16000))
            clip.write_videofile('{}test_inputWithTargetAudio_ep{}_{}.mp4'.format(res_dir, epoch, i), fps=10)  # export as video
            # .set_audio(AudioClip)
            # TODO: transform from HSV to RGB and save it as video
            # img_rgb = matplotlib.colors.hsv_to_rgb(img_hsv.reshape([100, 100, 3]))
        '''

        # TODO: calculate NLL of generated and target audio

        # CNN accuracy
        pic = cnn_out.cpu().data
        gen_out = pic.numpy()
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
                correct_samples_cnn += 1

            # Euclidian distance:
            # print("Shapes = i: {}, c: {}".format(np.shape(i), np.shape(c)))
            euc_dist_cnn += np.linalg.norm(i - c)
            num_samples += 1
        if verbose:
            print("[Tested {}%] acc: {}%, euc {}".format(round(100 * (bs_test+1)/test_size, 2),
                                                         round(100 * correct_samples_cnn/num_samples, 2),
                                                         round(euc_dist_cnn / num_samples, 2)))
    correct_samples_perc_cnn = 100 * correct_samples_cnn / num_samples
    euc_dist_mean_cnn = euc_dist_cnn / num_samples

    return correct_samples_perc_cnn, euc_dist_mean_cnn


def get_hidden_state(model, dataloader_test, dataloader_audio_test, params, epoch=0):
    hidden_enc_arr = []
    for bs_test, (d, a) in enumerate(zip(dataloader_test.keys(), dataloader_audio_test.keys())):
        data = dataloader_test[d]
        audio = dataloader_audio_test[a]
        img = data
        target = audio.reshape(-1, np.shape(audio)[-1], params['output_size']).to(params['gpu_num'][0])  # labels.to(device)
        with torch.no_grad():
            img = Variable(img).cuda()
        out, _, _, hidden_enc = model(img, target)
        hidden_enc_arr.append(hidden_enc)

    return hidden_enc_arr


def load_cnnseq2seq(use_best_checkpoint=True):
    cnn_model_path = utils.project_dir_name() + 'cnnseq/cnn_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.83perf/'
    cnn_params = load_json(cnn_model_path, 'parameters.json')
    cnnseq2seq_model_path = utils.project_dir_name() + 'cnnseq/cnnseq2seq_HSL_bin_1D_res_stepPred_8_ep_2_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/'
    cnnseq2seq_params = load_json(cnnseq2seq_model_path, 'seq2seq_parameters.json')

    saved_model = cnnseq2seq_params['saved_model']
    saved_model_cnn = cnnseq2seq_params['saved_model_cnn']
    # saved_model_cnn = cnn_params['saved_model']
    if use_best_checkpoint:
        saved_model = cnnseq2seq_params['saved_model_best']
        saved_model_cnn = cnnseq2seq_params['saved_model_cnn_best']
        # saved_model_cnn = cnn_params['saved_model_best_test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trim_model_name = True
    model = load_model(cnn_params, cnnseq2seq_params, cnnseq2seq_model_path, saved_model, saved_model_cnn,
                       device=device, trim_model_name=trim_model_name)
    ## Load CNN before fine-tuning (error loading fine-tuned)
    # cnn_model = load_model_cnn(cnn_params, cnn_model_path, saved_model_cnn)
    ## Load CNNSeq2Seq
    #model = CNNSeq2Seq(cnnseq2seq_params, cnn_model, device).cuda(cnnseq2seq_params['gpu_num'][0])
    #model.load_state_dict(torch.load(cnnseq2seq_model_path + saved_model))
    return model, cnnseq2seq_params


def load_model(cnn_params, cnnseq2seq_params, results_dir, saved_model_path, saved_model_path_cnn,
               device=torch.device('cpu'), trim_model_name=False):

    # Load fine-tuned CNN with old parameters
    cnn_model = load_model_cnn(cnn_params, results_dir, saved_model_path_cnn, trim_model_name=trim_model_name)

    # Load CNNSeq2Seq
    model = CNNSeq2Seq(cnnseq2seq_params, cnn_model, device).cuda(cnnseq2seq_params['gpu_num'][0])
    model_state_dict = model.state_dict()
    # for k, v in model_state_dict.items():
    #    print(k)

    pretrained_state = torch.load(results_dir + saved_model_path)
    if trim_model_name:
        pretrained_state = utils.fix_unexpected_keys_error(pretrained_state)
    model.load_state_dict(pretrained_state)
    # print("Loading model from: " + results_dir + saved_model_path)

    return model


def test(params, dataloader_label_test, dataloader_test, dataloader_audio_test, target_bin=True, device=torch.device('cpu'), use_best_checkpoint=False):

    cnn_model_path = params['cnn_model_path']
    cnn_params = load_json(cnn_model_path, 'parameters.json')
    cnnseq2seq_params = load_json(params['results_dir'], params['saved_model_parameters'])

    saved_model = params['saved_model']
    saved_model_cnn = params['saved_model_cnn']
    if use_best_checkpoint:
        saved_model = params['saved_model_best']
        saved_model_cnn = params['saved_model_cnn_best']
    model = load_model(cnn_params, cnnseq2seq_params, params['results_dir'], saved_model, saved_model_cnn, device=device)

    correct_samples_perc_cnn, euc_dist_mean_cnn = test_with_model(model, dataloader_label_test, dataloader_test,
                                                                  dataloader_audio_test, params, target_bin=target_bin,
                                                                  epoch='Test', verbose=False)

    return correct_samples_perc_cnn, euc_dist_mean_cnn


# TODO: generate without target
def generate_with_model(model, dataloader_test, params, epoch=0, verbose=True):
    res_dir = params['results_dir']
    number_samples = 0
    max_num_samples = 2
    for bs_test, d in enumerate(dataloader_test.keys()):
        data = dataloader_test[d]
        img = data
        with torch.no_grad():
            img = Variable(img).cuda()
        out, cnn_model, cnn_out, hidden_enc = model(img, None)  # target=None

        input_data_tmp = img.cpu().data.numpy()
        outputs_data_tmp = out.squeeze().cpu().data.numpy()

        input_reshaped = np.reshape(input_data_tmp, [-1, np.shape(input_data_tmp)[2], np.shape(input_data_tmp)[3],
                                    np.shape(input_data_tmp)[1]])
        out_reshaped = flatten_audio_with_params(outputs_data_tmp, params['sequence_length'],
                                                 params['audio_n_prediction'])

        if verbose:
            print("input_data_tmp: {}, outputs_data_tmp: {}".format(np.shape(input_data_tmp), np.shape(outputs_data_tmp)))
            print("input_reshaped: {}, out_reshaped: {}".format(np.shape(input_reshaped), np.shape(out_reshaped)))

        if number_samples < max_num_samples:
            # Save audio, 16KHz
            from scipy.io.wavfile import write
            test_gen_audio_scaled = np.int16(out_reshaped / np.max(np.abs(out_reshaped)) * 32767)
            write('{}test_gen_ep{}_s{}.wav'.format(res_dir, epoch, number_samples), 16000, test_gen_audio_scaled[0])

            scaled = input_reshaped

            from skimage import color
            frame_arr = []
            frame_arr2 = []
            for frame in scaled:
                frame = utils.normalize(frame, min=-1, max=1)
                frame_rgb = color.hsv2rgb(frame)
                frame_arr.append(frame_rgb * 255)
                frame_arr2.append(frame_rgb)
            if verbose:
                print("frame_arr: {}, min: {}, max: {}".
                      format(np.shape(frame_arr), np.amin(frame_arr), np.amax(frame_arr)))
                print("frame_arr2: {}, min: {}, max: {}".
                      format(np.shape(frame_arr2), np.amin(frame_arr2), np.amax(frame_arr2)))
            clip = ImageSequenceClip(np.array(frame_arr), fps=10)  # 3-second clip, .tolist()
            clip.set_audio(AudioArrayClip(test_gen_audio_scaled, fps=16000))
            clip.write_videofile('{}test_rgbInputWithGenAudio_ep{}_s{}.mp4'.format(res_dir, epoch, number_samples),
                                 fps=10)  # export as video
            number_samples += 1


def generate(params, dataloader_test, device=torch.device('cpu'), use_best_checkpoint=False):

    cnn_model_path = params['cnn_model_path']
    cnn_params = load_json(cnn_model_path, 'parameters.json')
    cnnseq2seq_params = load_json(params['results_dir'], params['saved_model_parameters'])

    saved_model = params['saved_model']
    saved_model_cnn = params['saved_model_cnn']
    if use_best_checkpoint:
        saved_model = params['saved_model_best']
        saved_model_cnn = params['saved_model_cnn_best']
    model = load_model(cnn_params, cnnseq2seq_params, params['results_dir'], saved_model, saved_model_cnn, device=device)

    generate_with_model(model, dataloader_test, params, epoch='Gen', verbose=True)

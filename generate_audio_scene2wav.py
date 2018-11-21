from model import SampleRNN
import torch
from collections import OrderedDict
import os
import json
from trainer.plugins import GeneratorPlugin, GeneratorCNNSeq2SamplePlugin
import numpy as np
from model import CNNSeq2SampleRNN
from train import make_data_loader

from librosa.output import write_wav
from moviepy.editor import ImageSequenceClip
from skimage import color
from cnnseq.utils import normalize


'''Other comments: https://github.com/deepsound-project/samplernn-pytorch/issues/8'''

# Paths
# RESULTS_PATH = 'results/exp:TEST_END2END-frame_sizes:16,4-n_rnn:2-dataset:data_npz/'
RESULTS_PATH = 'results/exp:TEST2_END2END_FIXED_AUDIO_RESHAPE-frame_sizes:16,4-n_rnn:2-dataset:data_npz/'
# RESULTS_PATH = 'results/exp:TEST_END2END-frame_sizes:16,4-n_rnn:2-dataset:data_npz (copy)/'
PRETRAINED_DIR = RESULTS_PATH + 'checkpoints/'
# PRETRAINED_CKP = 'best-ep38-it39292'
PRETRAINED_CKP = 'best-ep10-it11280'  # Audio orig
PRETRAINED_CKP = 'best-ep5-it5640'  # Audio
PRETRAINED_PATH = PRETRAINED_DIR + PRETRAINED_CKP
PRETRAINED_PATH_CNNSEQ2SAMPLE = PRETRAINED_DIR + 'cnnseq2sample-' + PRETRAINED_CKP
# GENERATED_PATH = RESULTS_PATH + 'generated/'
GENERATED_PATH = RESULTS_PATH + 'generated_cnnseq2sample/'
if not os.path.exists(GENERATED_PATH):
    os.mkdir(GENERATED_PATH)

TEST_PATH = RESULTS_PATH + 'test_cnnseq2sample/'
if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)


def test_emotion(chosen_emotion):
    e = 0
    for data in test_data_loader:
        batch_hsl = data[0]
        batch_audio = data[1]
        batch_emotion = data[2]
        batch_text = data[3]
        for b, a, em in zip(batch_hsl, batch_audio, batch_emotion):
            if e >= num_samples:
                break
            if em == chosen_emotion:
                b = np.array(b.numpy()).squeeze()
                a = np.array(np.reshape(a, [-1, sample_length]), dtype="float32").squeeze()
                print("Shape audio {}, video {}, emotion {}".format(np.shape(a), np.shape(b), em))
                filename_target_audio = TEST_PATH + 'test{}_em{}.wav'.format(e, em)
                write_wav(filename_target_audio, a, sr=16000, norm=True)

                input_reshaped = np.array(b).squeeze()

                frame_arr = []
                for frame in input_reshaped:
                    frame = np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2)  # np.transpose(frame)  # 100, 100, 3
                    frame = normalize(frame, min=0, max=1)
                    frame_rgb = color.hsv2rgb(frame)
                    frame_arr.append(frame_rgb * 255)

                clip = ImageSequenceClip(np.array(frame_arr), fps=10)  # 3-second clip, .tolist()
                filename = TEST_PATH + 'test{}_em{}.avi'.format(e, em)
                clip.write_videofile(filename, fps=10, codec='png', audio_fps=16000,
                                     audio=filename_target_audio)  # export as video
                e += 1
        if e >= num_samples:
            break


# Load model parameters from .json for audio generation
params_path = RESULTS_PATH + 'sample_rnn_params.json'
with open(params_path, 'r') as fp:
    params = json.load(fp)

# Create model with same parameters as used in training
model = SampleRNN(
    frame_sizes=params['frame_sizes'],
    n_rnn=params['n_rnn'],
    dim=params['dim'],
    learn_h0=params['learn_h0'],
    q_levels=params['q_levels'],
    weight_norm=params['weight_norm'],
    batch_size=params['batch_size']
)
model_cnnseq2sample = CNNSeq2SampleRNN().cuda()

# Delete "model." from key names since loading the checkpoint automatically attaches it to the key names
pretrained_state = torch.load(PRETRAINED_PATH)
new_pretrained_state = OrderedDict()

for k, v in pretrained_state.items():
    layer_name = k.replace("model.", "")
    new_pretrained_state[layer_name] = v
    # print("k: {}, layer_name: {}, v: {}".format(k, layer_name, np.shape(v)))

# Load pretrained model
model.load_state_dict(new_pretrained_state)
model = model.cuda()

pretrained_state_cnnseq2sample = torch.load(PRETRAINED_PATH_CNNSEQ2SAMPLE)
model_cnnseq2sample.load_state_dict(pretrained_state_cnnseq2sample)

# Generate Plugin
num_samples = 2  # params['n_samples']
sample_length = params['sample_length']
sample_rate = params['sample_rate']
print("Number samples: {}, sample_length: {}, sample_rate: {}".format(num_samples, sample_length, sample_rate))
generator = GeneratorCNNSeq2SamplePlugin(GENERATED_PATH, num_samples, sample_length, sample_rate)

# Call new register function to accept the trained model and the cuda setting
generator.register_generate(model, model_cnnseq2sample, params['cuda'])

# Test data
data_loader = make_data_loader(model.lookback, params, npz_filename='video_feats_HSL_10fps_pad_test.npz')
test_data_loader = data_loader(0, 1, eval=False)

# Test target audio with emotion 0 and 1
test_emotion(0)
test_emotion(1)

# Generate new audio
generator.epoch(test_data_loader, 'Test_cnnseq2sample')



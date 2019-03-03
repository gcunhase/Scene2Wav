## More notes on Training
* Training Scene2Wav using [pre-trained encoder](https://tinyurl.com/y8rkkw4z): run `train.py` with settable hyperparemeters.
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset data_npz --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_pad_test.npz --cnn_pretrain cnnseq/cnn4_3secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_asgd/ --cnn_seq2seq_pretrain cnnseq/cnnseq2seq4_3secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_asgdCNN_trainSize_3182_testSize_1139_cost_audio/
```

```
--exp DEBUG_DATA --frame_sizes 16 4 --n_rnn 2 --dataset data_npz --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_intAudio_pad_test.npz --cnn_pretrain cnnseq/cnn4_3secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_asgd/ --cnn_seq2seq_pretrain cnnseq/cnnseq2seq4_3secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_asgdCNN_trainSize_3182_testSize_1139_cost_audio/
--exp DEBUG_DATA --frame_sizes 16 4 --n_rnn 2 --dataset data_npz --npz_filename video_feats_HSL_10fps_origAudio_3secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_origAudio_3secs_intAudio_pad_test.npz --cnn_pretrain cnnseq/cnn4_3secs_origAudio_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_asgd/ --cnn_seq2seq_pretrain cnnseq/cnnseq2seq4_3secs_origAudio_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_asgdCNN_trainSize_3182_testSize_1139_cost_audio/
CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --dataset piano3
CUDA_VISIBLE_DEVICES=1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset COGNIMUSE_eq_eq_pad
CUDA_VISIBLE_DEVICES=2 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --q_levels 512 --dataset COGNIMUSE_eq_eq_pad

CUDA_VISIBLE_DEVICES=0,1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset splices_audio_BMI_16000_c1_16bits_music_eq

CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST_3SECS_CNNSEQ2SEQ_CORRECTED_ORIG_N3 --frame_sizes 16 4 --n_rnn 3 --dataset data_npz
CUDA_VISIBLE_DEVICES=1 python train.py --exp TEST_3SECS_CNNSEQ2SEQ_CORRECTED_N3 --frame_sizes 16 4 --n_rnn 3 --dataset data_npz --npz_filename video_feats_HSL_10fps_pad_train.npz --npz_filename_test video_feats_HSL_10fps_pad_test.npz --cnn_pretrain cnnseq/cnn2_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf/ --cnn_seq2seq_pretrain cnnseq/cnnseq2seq2_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/
```

## More detailed dependencies
```bash
apt-get install ffmpeg
pip install --upgrade pip
pip install -U numpy scipy matplotlib natsort scikit-image librosa

git clone https://github.com/librosa/librosa
cd librosa
python setup.py build
python setup.py install

pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36mu-linux_x86_64.whl

pip install moviepy requests

pip install pandas seaborn datashader plotnine umap-learn
```

## Notes
* **RuntimeError loading state_dict**: Addition of *module.* at the beginning of parameters' keys makes throws an *unexpected keys* error
```
RuntimeError: Error(s) in loading state_dict for ConvNet:
	Unexpected key(s) in state_dict: "layer1.1.num_batches_tracked", "layer2.1.num_batches_tracked", "layer3.1.num_batches_tracked". 
```

```python
from collections import OrderedDict
pretrained_state = torch.load(PRETRAINED_PATH)
new_pretrained_state = OrderedDict()

for k, v in pretrained_state.items():
    layer_name = k.replace("model.", "")
    new_pretrained_state[layer_name] = v
    print("k: {}, layer_name: {}, v: {}".format(k, layer_name, np.shape(v)))
    
# Load pretrained model
model.load_state_dict(new_pretrained_state)
model = model.cuda()
```

* In *moviepy*'s *ImageSequenceClip.py*:
```
if isinstance(sequence, list) or isinstance(sequence, np.ndarray)
```
# Scene2Wav
* A PyTorch implementation of *Scene2Wav: A Deep Neural Network for Emotional Scene Musicalization*
* Model has 3 stages: visual feature extraction with CNN, visual feature encoding with Deep RNN Encoder and music generation with SampleRNN Decoder.

![A visual representation of the Scene2Wav architecture]()

## Datasets
* Obtain data in `npz` format: Download the COGNIMUSE dataset and organize it and pre-process following instructions in [AnnotatedMV-PreProcessing](https://github.com/gcunhase/AnnotatedMV-PreProcessing) 
* Your dataset should be in a subfolder in `datasets/` filled with equal-length wav files.

## Pre-processing
```
python audio_preprocessing.py --folder FOLDER_NAME
```

## Training
1. Train your visual extraction and Encoder with CNN-Seq2Seq
    * Pre-train CNN with Scene frames and Emotion scores
    ```bash
    python CNN_main.py --mode=train
    ```
    * Pre-train CNN-Seq2Seq end-to-end with the Scene frames and Audio
    ```bash
    python CNNSeq2Seq_main.py --mode=train
    ```
2. To train the full CNNSeq2SampleRNN (Scene2Wav) model you need to run `train.py`. All model hyperparameters are settable in the command line. Most hyperparameters have sensible default values, so you don't need to provide all of them. Run `python train.py -h` for details. To train on the `piano` dataset using the best hyperparameters we've found, run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --dataset piano3
CUDA_VISIBLE_DEVICES=1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset COGNIMUSE_eq_eq_pad
CUDA_VISIBLE_DEVICES=2 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --q_levels 512 --dataset COGNIMUSE_eq_eq_pad

CUDA_VISIBLE_DEVICES=0,1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset splices_audio_BMI_16000_c1_16bits_music_eq

CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST_3SECS_CNNSEQ2SEQ_CORRECTED_ORIG_N3 --frame_sizes 16 4 --n_rnn 3 --dataset data_npz
CUDA_VISIBLE_DEVICES=1 python train.py --exp TEST_3SECS_CNNSEQ2SEQ_CORRECTED_N3 --frame_sizes 16 4 --n_rnn 3 --dataset data_npz --npz_filename video_feats_HSL_10fps_pad_train.npz --npz_filename_test video_feats_HSL_10fps_pad_test.npz --cnn_pretrain cnnseq/cnn2_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf/ --cnn_seq2seq_pretrain cnnseq/cnnseq2seq2_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/
```

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

We also have an option to monitor the metrics using [CometML](https://www.comet.ml/). To use it, just pass your API key as `--comet_key` parameter to `train.py`.


## Dependencies

This code requires Python 3.5+ and PyTorch 0.1.12+ (try last three options below). Installation instructions for PyTorch are available on their website: http://pytorch.org/. You can install the rest of the dependencies by running `pip install -r requirements.txt`.
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

pip install pandas seaborn datashader umap plotnine
```

## RuntimeError loading state_dict
* Addition of *module.* at the beginning of parameters' keys makes throws an *unexpected keys* error
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

## Note
* In *moviepy*'s *ImageSequenceClip.py*:
```
if isinstance(sequence, list) or isinstance(sequence, np.ndarray)
```

## Reference
* [PyTorch implementation of SampleRNN](https://github.com/deepsound-project/samplernn-pytorch)
    * Only allows GRU units
    * [My fork](https://github.com/gcunhase/samplernn-pytorch)
        * DONE: save model parameters in JSON, generate audio after training
        * TODO: LSTM units
    
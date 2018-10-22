# Scene2Wav
* A PyTorch implementation of *Scene2Wav: A Deep Neural Network for Emotional Scene Musicalization*
* Model has 3 stages: visual feature extraction with CNN, visual feature encoding with Deep RNN Encoder and music generation with SampleRNN Decoder.

![A visual representation of the Scene2Wav architecture]()


## Dependencies

This code requires Python 3.5+ and PyTorch 0.1.12+ (try last three options below). Installation instructions for PyTorch are available on their website: http://pytorch.org/. You can install the rest of the dependencies by running `pip install -r requirements.txt`.
```bash
pip install --upgrade pip
pip install -U numpy scipy matplotlib natsort
pip install librosa

git clone https://github.com/librosa/librosa
cd librosa
python setup.py build
python setup.py install

pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36mu-linux_x86_64.whl

pip install moviepy requests
```

## Datasets

We provide a script for creating datasets from YouTube single-video mixes. It downloads a mix, converts it to wav and splits it into equal-length chunks. To run it you need youtube-dl (a recent version; the latest version from pip should be okay) and ffmpeg. To create an example dataset - 4 hours of piano music split into 8 second chunks, run:

```
cd datasets
./download-from-youtube.sh "https://www.youtube.com/watch?v=EhO_MrRfftU" 8 piano
```

You can also prepare a dataset yourself. It should be a directory in `datasets/` filled with equal-length wav files. Or you can create your own dataset format by subclassing `torch.utils.data.Dataset`. It's easy, take a look at `dataset.FolderDataset` in this repo for an example.

## Pre-processing
```
python audio_preprocessing.py --folder FOLDER_NAME
```

## Training

To train the model you need to run `train.py`. All model hyperparameters are settable in the command line. Most hyperparameters have sensible default values, so you don't need to provide all of them. Run `python train.py -h` for details. To train on the `piano` dataset using the best hyperparameters we've found, run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --dataset piano3
CUDA_VISIBLE_DEVICES=1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset COGNIMUSE_eq_eq_pad
CUDA_VISIBLE_DEVICES=2 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 3 --q_levels 512 --dataset COGNIMUSE_eq_eq_pad

CUDA_VISIBLE_DEVICES=0,1 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset splices_audio_BMI_16000_c1_16bits_music_eq
```

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

We also have an option to monitor the metrics using [CometML](https://www.comet.ml/). To use it, just pass your API key as `--comet_key` parameter to `train.py`.

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

## Reference
* [PyTorch implementation of SampleRNN](https://github.com/deepsound-project/samplernn-pytorch)
    * Only allows GRU units
    * [My fork](https://github.com/gcunhase/samplernn-pytorch)
        * DONE: save model parameters in JSON, generate audio after training
        * TODO: LSTM units
    
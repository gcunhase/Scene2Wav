# Scene2Wav
* A PyTorch implementation of *Scene2Wav: A Deep Neural Network for Emotional Scene Musicalization*
* Model has 3 stages: visual feature extraction with CNN, visual feature encoding with Deep RNN Encoder and music generation with Scene2Wav decoder (conditional SampleRNN Decoder).

## Datasets and Pre-Processing
* Download `data.npz` or make it from scratch:
    1. Download the [COGNIMUSE dataset](http://cognimuse.cs.ntua.gr/database)
    2. Organize it and pre-process following instructions in [AnnotatedMV-PreProcessing](https://github.com/gcunhase/AnnotatedMV-PreProcessing) 
* The `.npz` dataset should be copied in a subfolder in a `datasets/` folder in the root of the repository
    
        .Scene2Wav
        +-- datasets
        |   +-- data_npz
        |       +-- my_data_train.npz
        |       +-- my_data_test.npz


## Training
* Training Scene2Wav using [pre-trained encoder](https://tinyurl.com/y8rkkw4z): run `train.py` with settable hyperparemeters.
```
CUDA_VISIBLE_DEVICES=0 python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset data_npz --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_pad_test.npz --cnn_pretrain cnnseq/cnn4_3secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_asgd/ --cnn_seq2seq_pretrain cnnseq/cnnseq2seq4_3secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_asgdCNN_trainSize_3182_testSize_1139_cost_audio/
```

* The results (training log, loss plots, model checkpoints and generated samples) will be saved in `results/`.

* You can check some generated samples from [this link](https://tinyurl.com/y8rkkw4z).
    
* If you need to train encoder Scene2Wav with customized dataset (instead of using pre-trained one):
    * Pre-train CNN with Scene frames and Emotion scores
    ```bash
    python CNN_main.py --mode=train
    ```
    * Pre-train CNN-Seq2Seq end-to-end with the Scene frames and Audio
    ```bash
    python CNNSeq2Seq_main.py --mode=train
    ```

## Dependencies

This code requires Python 3.5+ and PyTorch 0.1.12+ (try last three options below). Installation instructions for PyTorch are available on their [website](http://pytorch.org/).
You can install the rest of the dependencies by running `pip install -r requirements.txt`.

#### More detailed dependencies
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

## References
* [COGNIMUSE dataset](http://cognimuse.cs.ntua.gr/database)
* [AnnotatedMV-PreProcessing](https://github.com/gcunhase/AnnotatedMV-PreProcessing) 
* [PyTorch implementation of SampleRNN](https://github.com/deepsound-project/samplernn-pytorch)
    * Only allows GRU units
    * [My fork](https://github.com/gcunhase/samplernn-pytorch)
        * DONE: save model parameters in JSON, generate audio after training
        * TODO: LSTM units

## Credits
* In case you wish to use this code, please credit this repository or send me an email with any requests or questions.  

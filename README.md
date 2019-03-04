# Scene2Wav
* A PyTorch implementation of *Scene2Wav: A Deep Convolutional Sequence-to-SampleRNN for Emotional Scene Musicalization*
* Model has 3 stages: visual feature extraction with CNN, visual feature encoding with Deep RNN Encoder and music generation with Scene2Wav decoder (conditional SampleRNN Decoder).
<p align="center">
<img src="https://github.com/gcunhase/Scene2Wav/blob/master/data_analysis/proposed_model.png" width="100" alt="Scene2Wav">
</p>

## Requirements
This code requires Python 3.5+ and PyTorch 0.1.12+.
You can install the rest of the dependencies by running `pip install -r requirements.txt`.

## Dataset and Pre-Processing
* [Download `data.npz`](https://data.mendeley.com/datasets/dsynj2sxnc/draft?a=35a88183-11cd-4a13-87ee-c9cabf9e7f86) or make it from scratch:
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

* If you need to train encoder Scene2Wav with customized dataset (instead of using pre-trained one):
    * Pre-train CNN with Scene frames and Emotion scores
    ```bash
    python CNN_main.py --mode=train
    ```
    * Pre-train CNN-Seq2Seq end-to-end with the Scene frames and Audio
    ```bash
    python CNNSeq2Seq_main.py --mode=train
    ```
    
## Results
* The results (training log, loss plots, model checkpoints and generated samples) will be saved in `results/`.

* You can check some generated samples from [this link](https://tinyurl.com/y8rkkw4z).

* Emotion evaluation
    1. Install requirements + Melodia plugin
    ```bash
    pip install music21 vamp librosa midiutil
    ```
    2. Transform wav to midi and detect chords
    ```bash
    python emotion_evaluation.py --data_dir [data dirname] --inflie [filename].wav --outfile [filename].mid
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

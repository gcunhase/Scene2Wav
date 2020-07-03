## Requirements
This code was tested with Python 3.5+ and PyTorch 0.4.1 (`pip install --default-timeout=1000 torch==0.4.1`)
`pip install --default-timeout=1000 torch==0.4.1.post2`

The rest of the dependencies can be installed with `pip install -r requirements.txt`.

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

* 10 secs:
    * 1 - Train CNN:
    > ep40/lr0.001/adam: test 31.6716, best_test 68.3284, best_train 31.6716%, max_train_acc 79.13%
    > ep100/lr0.001/adam: test 31.6716, best_test 68.3284, best_train 31.6716%, max_train_acc 55.6139%
    > ep500/lr0.001/adam (80.31min): test 31.6716, best_test 68.3284, best_train 31.6716%, max_train_acc 55.6139%
    > ep40/lr0.0001/adam: test 40.1466, best_test 40.1466, best_train 40.1466%, max_train_acc 76.7716%
    > ep100/lr0.0001/adam: test 50.2053, best_test 50.2053, best_train 50.2053%, max_train_acc 83.5327%
    > ep500/lr0.0001/adam (80.57min): test 53.4897, best_test 53.4897, best_train 53.4213%, max_train_acc 86.2749%
    > ep100/lr0.001/asgd (15.65min): test 40.8407, best_test 44.5161, best_train 40.8407%, max_train_acc 74.2882%
    > ep500/lr0.001/asgd (77.53min): test 69.4428, best_test 69.4428, best_train 69.4037%, max_train_acc 86.4568%
    > ep500/lr0.0005/asgd (87.49min): test 69.6774, best_test 69.6872, best_train 69.6872%, max_train_acc 83.2109%
    > ep100/lr0.0001/asgd (15.46min): test 29.1007, best_test 61.1046, best_train 29.1007%, max_train_acc 68.9087%
    > ep500/lr0.0001/asgd (71.22min): test 44.4575, best_test 61.1046, best_train 44.3793%, max_train_acc 75.5194%
    > **cnn4_10secs**: test 69.8143, best_test 70.5865, best_train 70.2639%, max_train_acc 89.4788%, loss: 0.3133
    ```
    CUDA_VISIBLE_DEVICES=5 python CNN_main.py --mode=train --num_epochs=500 --learning_rate=0.0005 --optimizer_name=asgd --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad --results_dir=./results/cognimuse_cnn_10secs_HSL_bin_1D/
    # cnn4_10secs test
    CUDA_VISIBLE_DEVICES=5 python CNN_main.py --mode=test --num_epochs=40 --learning_rate=0.001 --optimizer_name=adam --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad --results_dir=./results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/
    ``` 
    * 2 - Test CNNSeq2Seq:
    > **cnnseq2seq4_10secs**: min loss: 109.27816009521484, CNN accuracy: 54.62% and euclidian distance: 0.64
    ```
    CUDA_VISIBLE_DEVICES=5 python CNNSeq2Seq2_main.py --mode=test --sequence_length=160000 --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad --cnn_model_path=./results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --results_dir=./results/cnnseq2seq4_10secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_adamCNN_trainSize_953_testSize_341_cost_audio/
    ```
    * 3 - Train Scene2Wav:
    > Trained from scratch 1 (already had this one): training_loss: 1.1262   validation_loss: 1.0683    test_loss: 1.1027
    > Resumed above: GPU4: Started Jun 14th 18:21, ep14 Jun20th-18:04, ep15 Jun 21st-21:00, ep16 Jun22nd-21:52, ep18 Jun25th-17:24
    > From scratch 2: GPU3: Started Jun 14th 01:04, ep5 Jun20th-10:00
    > From scratch 3: LDS4 GPU1: Started Jun 16th 17:05
    ```
    CUDA_VISIBLE_DEVICES=4 python train.py --seq2seq_model_type=seq2seq --exp TEST_END2END_intAudio_10secs_CNNadam_adam_RESUME --epoch_limit 1000 --sample_length 160000 --frame_sizes 16 4 --dataset data_npz --batch_size 128 --cnn_pretrain cnnseq/results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/cnnseq2seq4_10secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_adamCNN_trainSize_953_testSize_341_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_10secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_10secs_intAudio_pad_test.npz
    CUDA_VISIBLE_DEVICES=3 python train.py --seq2seq_model_type=seq2seq --exp TEST2_END2END_intAudio_10secs_CNNadam_adam --epoch_limit 1000 --sample_length 160000 --frame_sizes 16 4 --dataset data_npz --batch_size 128 --cnn_pretrain cnnseq/results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/cnnseq2seq4_10secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_adamCNN_trainSize_953_testSize_341_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_10secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_10secs_intAudio_pad_test.npz
    # LDS4
    CUDA_VISIBLE_DEVICES=1 python train.py --seq2seq_model_type=seq2seq --exp TEST3_END2END_intAudio_10secs_CNNadam_adam --epoch_limit 1000 --sample_length 160000 --frame_sizes 16 4 --dataset data_npz --batch_size 128 --cnn_pretrain cnnseq/results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/cnnseq2seq4_10secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_adamCNN_trainSize_953_testSize_341_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_10secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_10secs_intAudio_pad_test.npz
    ```

* 10 secs (video size corrected - num of frames should be 100):
    * 1 - Train CNN:
    > bs30: ep40/lr0.001/adam (27.80 min): loss: 0.4405, test 68.3377, best_test 68.2938, best_train 31.6623%, max_train_acc 55.6080%
    >       ep100/lr0.001/adam (148.80 min): loss: 0.4977, test 68.3377, best_test 68.3377, best_train 31.6623%, max_train_acc 55.6080%
    >       ep200/lr0.001/adam (66.91 min): loss: 0.3313, test 68.3377, best_test 68.3377, best_train 31.6623%, max_train_acc 55.6080%
    >       ep40/lr0.0001/adam (28.56 min): loss: 0.3133, test 26.1829, best_test 31.6623, best_train 25.3621%, max_train_acc 67.0444%
    >       ep100/lr0.0001/adam (53.99): loss: 0.3266, test 68.3377, best_test 68.3377, best_train 31.6623%, max_train_acc 55.6080%
    >       ep200/lr0.0001/adam (70.72 min): loss: 0.8830, test 72.0000, best_test 77.9298, best_train 72.0000%, max_train_acc 83.3811%
    > bs100 (100 frames): ep40/lr0.001/adam (15.69 min): loss: 0.8523, test 31.6716, best_test 68.2258, best_train 31.6716%, max_train_acc 55.6139%
    >                     ep100/lr0.001/adam (35.47 min): loss: 0.7971, test 31.6716, best_test 68.2581, best_train 31.6716%, max_train_acc 55.6139%
    >                     ep200/lr0.001/adam (63.85 min): loss: 0.8148, test 31.6716, best_test 68.1994, best_train 31.6716%, max_train_acc 55.6139%
    >                     ep40/lr0.0001/adam (29.50 min): loss: 0.3494, test 34.0792, best_test 42.1994, best_train 41.1877%, max_train_acc 78.4187%
    >                     ep100/lr0.0001/adam (43.45 min): loss: 0.3838, test 67.5982, best_test 67.5982, best_train 67.5982%, max_train_acc 87.0714%
    >                     **ep200/lr0.0001/adam** (58.41 min): loss: 0.3534, test 71.6246, best_test 71.6246, best_train 71.6246%, max_train_acc 88.0735%
    ```
    CUDA_VISIBLE_DEVICES=5 python CNN_main.py --mode=train --num_epochs=200 --batch_size=100 --learning_rate=0.0001 --optimizer_name=adam --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected --results_dir=./results/cognimuse_videoFramesCorrected_cnn_10secs_HSL_bin_1D/
    ``` 
    * 2 - Test CNNSeq2Seq:
    > bs100
    ```
    # CNNSeq2Seq (bs30/ep20), CNN (bs30/ep40/lr0.001/adam)
    CUDA_VISIBLE_DEVICES=3 python CNNSeq2Seq2_main.py --mode=train --input_size=100 --num_epochs=20 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --sequence_length=160000 --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected --cnn_model_path=./results/cognimuse_videoFramesCorrected_cnn_10secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --results_dir=./results/cognimuse_videoFramesCorrected_cnnseq2seq_10secs_HSL_bin_1D/
    # CNNSeq2Seq (bs30/ep20), CNN (bs100/ep40/lr0.001/adam)
    CUDA_VISIBLE_DEVICES=3 python CNNSeq2Seq2_main2.py --mode=train --input_size=100 --batch_size=30 --num_epochs=20 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --sequence_length=160000 --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected --cnn_model_path=./results/cognimuse_videoFramesCorrected_cnn_10secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_100_lr_0.0001_we_0.0001_adam/ --results_dir=./results/cognimuse_videoFramesCorrected_cnnseq2seq_10secs_HSL_bin_1D_cnn_bs100_ep200/
    # CNNSeq2Seq (bs100/ep20), CNN (bs100/ep40/lr0.001/adam)
    CUDA_VISIBLE_DEVICES=6 python CNNSeq2Seq2_main3.py --mode=train --input_size=100 --batch_size=100 --num_epochs=20 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --sequence_length=160000 --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected --cnn_model_path=./results/cognimuse_videoFramesCorrected_cnn_10secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_100_lr_0.0001_we_0.0001_adam/ --results_dir=./results/cognimuse_videoFramesCorrected_cnnseq2seq_10secs_HSL_bin_1D_cnn_bs100_ep200/
    CUDA_VISIBLE_DEVICES=6 python CNNSeq2Seq2_main4.py --mode=train --input_size=100 --batch_size=100 --num_epochs=20 --learning_rate=0.0001 --optimizer_name=adam --num_layers=2 --sequence_length=160000 --data_dir=../datasets/data_npz/ --data_filename=video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected --cnn_model_path=./results/cognimuse_videoFramesCorrected_cnn_10secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_100_lr_0.0001_we_0.0001_adam/ --results_dir=./results/cognimuse_videoFramesCorrected_cnnseq2seq_10secs_HSL_bin_1D_cnn_bs100_ep200/
    ```
    * 3 - Train Scene2Wav:
    > Trained from scratch (Start Jun29th-17:35, ep): 
    ```
    CUDA_VISIBLE_DEVICES=5 python train.py --exp TEST_END2END_intAudio_10secs_CNNadam_adam_videoSizeCorrected10secs --seq2seq_model_type=seq2seq_gru --epoch_limit 25 --sample_length 160000 --frame_sizes 16 4 --dataset data_npz --batch_size 128 --cnn_pretrain cnnseq/results/cognimuse_videoFramesCorrected_cnn_10secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_100_lr_0.0001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/cognimuse_videoFramesCorrected_cnnseq2seq_10secs_HSL_bin_1D_cnn_bs100_ep200/HSL_bin_1D_res_stepPred_8_ep_20_bs_100_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_trainSize_953_testSize_341_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected_train.npz --npz_filename_test video_feats_HSL_10fps_10secs_intAudio_pad_videoFramesCorrected_test.npz
    ```

* 3 secs, DEAP Highlights/Raw (25 frames per splice):
    > (fps=25 to 10 (downsample factor=floor(2.5)), code was fixed so a 3sec video has 25 frames)
    * 1 - Train CNN:
    > Highlights: ep40/lr0.001/adam (4.90 min): test 73.6842, best_test 73.6842, best_train 73.6842%, loss: 1.3132, max_train_acc 73.6067%
    >             ep100/lr0.001/adam (10.6535 min): test 73.6842, best_test 73.6842, best_train 73.6842%, loss: 1.3131, max_train_acc 73.6067%
    >             ep100/lr0.0001/adam (9.989 min): test 81.7544, best_test 81.7895, best_train 81.7544%, loss: 1.3118, max_train_acc 84.1693%
    >             **ep200/lr0.0001/adam** (20.03 min): loss: 1.3131, test 82.2456, best_test 82.2456, best_train 82.2456%, max_train_acc 84.6372%
    >             ep100/lr0.00001/adam (10.81 min): loss: 1.2512, test 78.0702, best_test 78.0702, best_train 78.0702%, max_train_acc 78.5594%
    >             ep100/lr0.0001/asgd (9.83 min): loss: 1.1303, test 79.5789, best_test 79.5789, best_train 79.5789%, max_train_acc 81.4826%
    >             ep200/lr0.0001/asgd (19.43 min): loss: 1.1606, test 79.8246, best_test 79.9298, best_train 79.8246%, max_train_acc 82.1872%
    >             ep100/lr0.00001/asgd (10.0759 min): loss: 1.0391, test 74.5263, best_test 74.7719, best_train 74.2807%, max_train_acc 75.4259%
    > Raw: ep40/lr0.001/adam (18.67 min): loss: 0.3134, test 26.3158, best_test 34.0351, best_train 75.3684%, max_train_acc 73.3460%
    >      ep100/lr0.001/adam (47.14 min): loss: 1.2900, 73.6842, best_test 30.6667, best_train 73.6842%, max_train_acc 72.5965%
    >      ep100/lr0.0001/adam (51.78): loss: 1.2604, test 74.8421, best_test 75.6842, best_train 74.8421%, max_train_acc 89.0849%
    >      **ep200/lr0.0001/adam** (84.19 min): loss: 0.8161, test 74.9123, best_test 78.9123, best_train 74.9123%, max_train_acc 90.3607%
    >      ep200/lr0.00001/adam (143.3115 min): loss: 0.7608, test 69.7193, best_test 76.9825, best_train 69.7193%, max_train_acc 84.2512%
    >      ep400/lr0.00001/adam (216.7081 min): loss: 0.7677, test 71.0877, best_test 76.9474, best_train 71.0877%, max_train_acc 84.4239%
    >      ep40/lr0.001/asgd (25.46 min): loss: 1.3133, test 73.6491, best_test 73.6842, best_train 73.6842%, max_train_acc 75.1565%
    >      ep100/lr0.001/asgd: loss: 1.2696, test_acc: 76.8772, train_acc 80.0154 at 72 ep
    >      ep100/lr0.0001/asgd (100.53 min): loss: 0.9239, test 76.9825, best_test 77.5439, best_train 76.9825%, max_train_acc 80.5516%
    >      ep200/lr0.0001/asgd (136.31 min): loss: 0.8831, test 72.0000, best_test 77.9649, best_train 72.0000%, max_train_acc 83.3754%
    ```
    # Highlights
    CUDA_VISIBLE_DEVICES=4 python CNN_main.py --mode=train --num_epochs=200 --learning_rate=0.0001 --optimizer_name=adam --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames --results_dir=./results/deap_cnn_3secs_HSL_bin_1D_25frames/
    # Raw
    CUDA_VISIBLE_DEVICES=4 python CNN_main.py --mode=train --num_epochs=200 --learning_rate=0.0001 --optimizer_name=adam --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_raw --results_dir=./results/deap_cnn_3secs_HSL_bin_1D_25frames_raw/
    ```
    * 2 - Train CNNSeq2Seq (input_size=25 num_frames per splice)
    > Highlights: ep20/lr0.001/adam (7.67 hours): min loss: 347.1527404785156, CNN accuracy: 27.81% and euclidian distance: 0.91
    >             **ep100/lr0.001/adam** (28.98 hours): min loss: 288.213623046875, CNN accuracy: 54.31% and euclidian distance: 0.67
    >             ep20/lr0.0001/adam (3.08 hours): min loss: 493.8992004394531, CNN accuracy: 73.05% and euclidian distance: 0.47
    >             ep100/lr0.0001/adam (12.79 hours): min loss: 397.9322204589844, CNN accuracy: 63.49% and euclidian distance: 0.60
    > Raw: **ep20/lr0.001/adam** (18.93 hours): min loss: 146.0429229736328, CNN accuracy: 27.46% and euclidian distance: 1.02
    >      ep100/lr0.001/adam (68.76 hours): min loss: 140.58963012695312, CNN accuracy: 27.44% and euclidian distance: 1.02
    >      ep20/lr0.0001/adam (18.63 hours): min loss: 215.80377197265625, CNN accuracy: 27.98% and euclidian distance: 0.99
    >      ep100/lr0.0001/adam (68.98 hours): min loss: 155.93309020996094, CNN accuracy: 27.44% and euclidian distance: 1.02
    ```
    # Highlights
    CUDA_VISIBLE_DEVICES=6 python CNNSeq2Seq2_main.py --mode=train --input_size=25 --num_epochs=20 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames --cnn_model_path=./results/deap_cnn_3secs_HSL_bin_1D_25frames/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --results_dir=./results/deap_cnnseq2seq_3secs_HSL_bin_1D_25frames/
    # Raw
    CUDA_VISIBLE_DEVICES=6 python CNNSeq2Seq2_main.py --mode=train --input_size=25 --num_epochs=20 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_raw --cnn_model_path=./results/deap_cnn_3secs_HSL_bin_1D_25frames_raw/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --results_dir=./results/deap_cnnseq2seq_3secs_HSL_bin_1D_25frames_raw/
    ```
    * 3 - Train Scene2Wav
    > Highlights: CNN (ep200/lr0.0001/adam), CNNSeq2Seq (ep100/lr0.001/adam) (Start: Jun15th 23:28 to Jun17th 23:30, GPU5): 25 epochs, training_loss: 1.3057   validation_loss: 1.2277 test_loss: 1.3306
    > Highlights: CNN (ep200/lr0.0001/adam), CNNSeq2Seq (ep20/lr0.001/adam) (Start: Jun19th 13:42 to Jun21st-15:57, GPU5): 25 epochs, training_loss: 1.2943   validation_loss: 1.2276 test_loss: 1.3252
    > Highlights RESUME: CNN (ep200/lr0.0001/adam), CNNSeq2Seq (ep20/lr0.001/adam) (Start  Jun22nd-01:56: , ep35:Jun22nd-21:52, ep41 Jun23rd-15:11, ep45 Jun23rd-16:48): +25epochs, training_loss: 1.1060   validation_loss: 1.0451 test_loss: 1.1420
    > Highlights RESUME2: CNN (ep200/lr0.0001/adam), CNNSeq2Seq (ep20/lr0.001/adam) (Start  Jun25th-18:22, ep59 Jun26th-11:50)
    > Raw: CNN (ep200/lr0.0001/adam), CNNSeq2Seq (ep20/lr0.001/adam) (Start: Jun16th 16:46, LDS4 GPU1, ep10 Jun21st-14:00, ep13 Jun22nd 21:52, ep14 Jun23rd-15:11, ep17 Jun24th-18:21, ep19 Jun25th-17:24, ep20 Jun26th-11:51): 
    ```
    # Highlights - cnnseq2seq ep100
    CUDA_VISIBLE_DEVICES=5 python train.py --exp DEAP_25frames_END2END_intAudio_3secs_CNNadam_adam --seq2seq_model_type=seq2seq_gru --epoch_limit 1000 --sample_length 48000 --frame_sizes 16 4 --dataset data_npz_deap --batch_size 128 --cnn_pretrain cnnseq/results/deap_cnn_3secs_HSL_bin_1D_25frames/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/deap_cnnseq2seq_3secs_HSL_bin_1D_25frames/HSL_bin_1D_res_stepPred_8_ep_100_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_trainSize_760_testSize_114_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_test.npz
    # Highlights - cnnseq2seq ep20
    CUDA_VISIBLE_DEVICES=5 python train.py --exp DEAP_25frames_END2END_intAudio_3secs_CNNadam_adam_seq2seq_ep20 --seq2seq_model_type=seq2seq_gru --epoch_limit 25 --sample_length 48000 --frame_sizes 16 4 --dataset data_npz_deap --batch_size 128 --cnn_pretrain cnnseq/results/deap_cnn_3secs_HSL_bin_1D_25frames/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/deap_cnnseq2seq_3secs_HSL_bin_1D_25frames/HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_trainSize_760_testSize_114_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_test.npz
    CUDA_VISIBLE_DEVICES=5 python train.py --exp DEAP_25frames_END2END_intAudio_3secs_CNNadam_adam_seq2seq_ep20_RESUME --seq2seq_model_type=seq2seq_gru --epoch_limit 25 --sample_length 48000 --frame_sizes 16 4 --dataset data_npz_deap --batch_size 128 --cnn_pretrain cnnseq/results/deap_cnn_3secs_HSL_bin_1D_25frames/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/deap_cnnseq2seq_3secs_HSL_bin_1D_25frames/HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_trainSize_760_testSize_114_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_test.npz
    # Raw
    CUDA_VISIBLE_DEVICES=1 python train.py --exp DEAP_25frames_raw_END2END_intAudio_3secs_CNNadam_adam --seq2seq_model_type=seq2seq_gru --epoch_limit 1000 --sample_length 48000 --frame_sizes 16 4 --dataset data_npz_deap --batch_size 128 --cnn_pretrain cnnseq/results/deap_cnn_3secs_HSL_bin_1D_25frames_raw/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/deap_cnnseq2seq_3secs_HSL_bin_1D_25frames_raw/HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_trainSize_2850_testSize_114_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_raw_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_intAudio_pad_1D_25frames_raw_test.npz
    ```
    > CUDA_VISIBLE_DEVICES=5 python train.py --exp DEBUG --epoch_limit 1000 --sample_length 160000 --frame_sizes 16 4 --dataset data_npz --batch_size 128 --cnn_pretrain cnnseq/results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/cnnseq2seq4_10secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_adamCNN_trainSize_953_testSize_341_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_10secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_10secs_intAudio_pad_test.npz

  
* 3 secs, DEAP Highlights (16 frames per splice): 
    > (fps=25 to 10, but because of a coding error, a 3sec video has 16 frames):
    * 1 - Train CNN:
    > Highlights: ep40/lr0.001/adam: test 81.1475, best_test 81.1475, best_train 81.1475%
    >             ep100/lr0.001/adam: 73.4426, best_test 75.4098, best_train 77.1585%
    >             ep100/lr0.0001/adam: 82.4863, best_test 82.4863, best_train 82.4863%
    >             **ep200/lr0.0001/adam**: test 83.1557, best_test 83.1557, best_train 83.1557%, loss: 1.2026, train_acc 84.0805
    >             ep500/lr0.0001/adam: test 83.1557, best_test 83.1557, best_train 83.1557%, loss: 1.2014, train_acc 84.1133
    >             ep100/lr0.00001/adam: test 77.0492, best_test 77.1038, best_train 77.0492%
    >             ep200/lr0.00001/adam: 77.2678, best_test 77.3224, best_train 77.3224%
    >             ep100/lr0.0001/asgd: 79.1803, best_test 77.4590, best_train 79.1803%
    >             ep200/lr0.0001/asgd: test 79.7404, best_test 79.7404, best_train 79.7404%
    >             ep100/lr0.00001/asgd: 74.5902, best_test 74.5902, best_train 73.9891%
    ```
    CUDA_VISIBLE_DEVICES=5 python CNN_main.py --mode=train --num_epochs=200 --learning_rate=0.0001 --optimizer_name=asgd --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad --results_dir=./results/deap_cnn_3secs_HSL_bin_1D/
    ``` 
    * 2 - Train CNNSeq2Seq (input_size=16 num_frames per splice)
    > Highlights: ep20/lr0.001/adam: min loss: 346.9956, CNN [loss: 0.026022227481007576, acc: 26.42543859649123, euc: 0.9826706558836904]
    >             **ep100/lr0.001/adam (8.59 hours)**: min loss: 288.0337829589844, CNN accuracy: 26.69% and euclidian distance: 1.01
    >             ep100/lr0.0001/adam (9.1 hours): min loss: 397.8754577636719, CNN accuracy: 29.73% and euclidian distance: 0.87
    >             ep20/lr0.001/asgd (2.83 hours): min loss: 537.017822265625 CNN accuracy: 72.85% and euclidian distance: 0.53
    >             ep100/lr0.001/asgd (9.74 hours): min loss: 477.00408935546875 CNN accuracy: 72.46% and euclidian distance: 0.53
    >             ep20/lr0.0001/asgd (2.81 hours): min loss: 525.3634033203125, CNN accuracy: 72.83% and euclidian distance: 0.53
    >             ep100/lr0.0001/asgd (10.64 hours): min loss: 536.0626831054688 CNN accuracy: 72.43% and euclidian distance: 0.53
    >             ep20/lr0.001/adam/model_type=seq2seq(lstm) (8.82 hours): min loss: 358.66949462890625, CNN accuracy: 26.98% and euclidian distance: 0.97
    >             ep100/lr0.001/adam/model_type=seq2seq(lstm) (31.68 hours): min loss: 286.0079040527344, CNN accuracy: 26.32% and euclidian distance: 1.04
    ```
    CUDA_VISIBLE_DEVICES=6 python CNNSeq2Seq2_main.py --mode=train --model_type=seq2seq_gru --input_size=16 --num_epochs=20 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad --cnn_model_path=./results/deap_cnn_3secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --results_dir=./results/deap_cnnseq2seq_3secs_HSL_bin_1D_seq2seq/
    CUDA_VISIBLE_DEVICES=6 python CNNSeq2Seq2_main_3secs_deap.py --mode=train --model_type=seq2seq --input_size=16 --num_epochs=100 --learning_rate=0.001 --optimizer_name=adam --num_layers=2 --data_dir=../datasets/data_npz_deap/ --data_filename=video_feats_HSL_10fps_3secs_intAudio_pad --cnn_model_path=./results/deap_cnn_3secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --results_dir=./results/deap_cnnseq2seq_3secs_HSL_bin_1D_seq2seq/
    ```
    * 3 - Train Scene2Wav
    > CNN (ep200/lr0.0001/adam), CNNSeq2Seq (ep100/lr0.001/adam) (Start:Jun13th-21:45 to Jun15th-22:27 GPU5): training_loss: 1.2476	validation_loss: 1.1798	test_loss: 1.2989
    ```
    CUDA_VISIBLE_DEVICES=5 python train.py --exp DEAP2_END2END_intAudio_3secs_CNNadam_adam --seq2seq_model_type=seq2seq_gru --epoch_limit 1000 --sample_length 48000 --frame_sizes 16 4 --dataset data_npz_deap --batch_size 128 --cnn_pretrain cnnseq/results/deap_cnn_3secs_HSL_bin_1D/res_vanilla_HSL_bin_1D_CrossEntropy_ep_200_bs_30_lr_0.0001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/deap_cnnseq2seq_3secs_HSL_bin_1D/HSL_bin_1D_res_stepPred_8_ep_100_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_trainSize_760_testSize_114_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_3secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_3secs_intAudio_pad_test.npz
    ```
    > CUDA_VISIBLE_DEVICES=5 python train.py --exp DEBUG --epoch_limit 1000 --sample_length 160000 --frame_sizes 16 4 --dataset data_npz --batch_size 128 --cnn_pretrain cnnseq/results/cnn4_10secs_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam/ --cnn_seq2seq_pretrain cnnseq/results/cnnseq2seq4_10secs_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_adam_adamCNN_trainSize_953_testSize_341_cost_audio/ --n_rnn 2 --n_samples 1 --npz_filename video_feats_HSL_10fps_10secs_intAudio_pad_train.npz --npz_filename_test video_feats_HSL_10fps_10secs_intAudio_pad_test.npz

## More detailed requirements
This code requires Python 3.5+ and PyTorch 0.1.12+ (try last three options below). Installation instructions for PyTorch are available on their [website](http://pytorch.org/).
You can install the rest of the dependencies by running `pip install -r requirements.txt`.

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

* In `venv/lib/python3.6/site-packages/moviepy/video/io/ImageSequenceClip.py`:
```
if isinstance(sequence, list) or isinstance(sequence, np.ndarray)
```

## References
* [AnnotatedMV-PreProcessing](https://github.com/gcunhase/AnnotatedMV-PreProcessing) 
* [PyTorch implementation of SampleRNN](https://github.com/deepsound-project/samplernn-pytorch)
    * Only allows GRU units
    * [My fork](https://github.com/gcunhase/samplernn-pytorch)
        * DONE: save model parameters in JSON, generate audio after training
        * TODO: LSTM units

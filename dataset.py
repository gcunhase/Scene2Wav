import utils

import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from natsort import natsorted

from os import listdir
from os.path import join
import numpy as np
import os

class FolderDataset(Dataset):
    """An abstract class representing a Dataset being uploaded from a folder."""

    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        file_names = natsorted(
            [join(path, file_name) for file_name in listdir(path)]
        )
        self.file_names = file_names[
            int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))
        ]

    def __getitem__(self, index):
        (seq, _) = load(self.file_names[index], sr=None, mono=True)
        return torch.cat([
            torch.LongTensor(self.overlap_len) \
                 .fill_(utils.q_zero(self.q_levels)),
            utils.linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ])

    def __len__(self):
        return len(self.file_names)

    def get_filename(self, index):
        return self.file_names[index]


class NpzDataset(Dataset):
    """An abstract class representing a Dataset being loaded from a .npz file."""

    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        data = np.load(path)
        audio = data['audio']
        samples = len(audio)
        self.audio_samples = audio[
            int(ratio_min * samples) : int(ratio_max * samples)
        ]
        hsl_data = data['HSL_data']
        samples = len(hsl_data)
        self.hsl_data_samples = hsl_data[
                             int(ratio_min * samples): int(ratio_max * samples)]
        print('Audio samples: {}, hsl data: {}'.format(np.shape(self.audio_samples),
                                                       np.shape(self.hsl_data_samples)))

    def __getitem__(self, index):
        seq = self.audio_samples[index]
        hsl_data = self.hsl_data_samples[index]
        return torch.cat([
            torch.LongTensor(self.overlap_len) \
                 .fill_(utils.q_zero(self.q_levels)),
            utils.linear_quantize(
                torch.from_numpy(seq), self.q_levels
            )
        ]), seq, hsl_data

    def __len__(self):
        return len(self.audio_samples)

    def get_audio_sample(self, index):
        return self.audio_samples[index]


class DataLoader(DataLoaderBase):
    """
    Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
    """

    def __init__(self, dataset, batch_size, seq_len, overlap_len,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        ls = super().__len__()
        for i, batch in enumerate(super().__iter__()):
            batch_audio = batch[0]
            batch_audio_seq = batch[1]
            batch_hsl = batch[2]
            (batch_size, n_samples) = batch_audio.size()
            batch_size_hsl = batch_hsl.size()
            batch_size_audio = batch_audio.size()

            # Divide audio into groups of "audio_n_prediction"
            import math
            audio_n_prediction = 8
            batch_audio_48000 = batch_audio_seq  # batch_audio.res[:48000]  # batch_audio
            s = np.shape(batch_audio_48000)
            y_audio_dim = int(math.ceil(s[1] / audio_n_prediction))
            audio_train = np.zeros([s[0], audio_n_prediction, y_audio_dim])
            for i in range(0, np.shape(audio_train)[0]):
                k = 0
                j_final = s[1]  # -audio_n_prediction-1
                for j in range(0, j_final, audio_n_prediction):
                    splice_audio = batch_audio_48000[i, j:min(s[1], j + audio_n_prediction)]
                    audio_train[i, :, k] = splice_audio
                    k += 1

            print("input: {}, {}, hsl: {}, audio: {}/{}, audio_train: {}".
                  format(batch_size, n_samples, batch_size_hsl, batch_size_audio,
                         np.shape(batch_audio_seq), np.shape(audio_train)))

            #l = batch.__len__()
            #for b in range(0, batch.__len__()):
            #    print(batch[b].data)
            #    print(batch[b].data.get_filename(b))

            reset = True

            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len
                sequences = batch_audio[:, from_index : to_index]  # (batch_size, 1088)
                input_sequences = sequences[:, : -1]  # (batch_size, 1087)
                target_sequences = sequences[:, self.overlap_len :].contiguous()
                # import numpy as np
                # print("in: {}, tar: {}".format(np.shape(input_sequences), np.shape(target_sequences)))
                # print("batch hsl: {}, batch hsl: {}, input_seq: {}, reset: {}, target_seq: {}"
                #       .format(batch_hsl.size(), batch_audio.size(), np.shape(input_sequences), reset, np.shape(target_sequences)))
                yield (batch_hsl, audio_train, input_sequences, reset, target_sequences)

                reset = False

    def __len__(self):
        # raise NotImplementedError()
        return super().__len__()

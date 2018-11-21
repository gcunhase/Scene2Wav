import matplotlib
matplotlib.use('Agg')

from model import Generator, GeneratorCNNSeq2Sample

import torch
from torch.autograd import Variable
from torch.utils.trainer.plugins.plugin import Plugin
from torch.utils.trainer.plugins.monitor import Monitor
from torch.utils.trainer.plugins import LossMonitor

from librosa.output import write_wav
from matplotlib import pyplot

from glob import glob
import os
import pickle
import time
import numpy as np
from cnnseq import utils
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip


class TrainingLossMonitor(LossMonitor):

    stat_name = 'training_loss'


class ValidationPlugin(Plugin):

    def __init__(self, val_dataset, test_dataset):
        super().__init__([(1, 'epoch')])
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def register(self, trainer):
        self.trainer = trainer
        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['log_epoch_fields'] = ['{last:.4f}']
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['log_epoch_fields'] = ['{last:.4f}']

    def epoch(self, idx):
        self.trainer.model.eval()

        val_stats = self.trainer.stats.setdefault('validation_loss', {})
        val_stats['last'] = self._evaluate(self.val_dataset)
        test_stats = self.trainer.stats.setdefault('test_loss', {})
        test_stats['last'] = self._evaluate(self.test_dataset)

        self.trainer.model.train()

    def _evaluate(self, dataset):
        loss_sum = 0
        n_examples = 0
        for data in dataset:
            batch_hsl = data[0]
            batch_audio = data[1]
            batch_emotion = data[2]
            batch_text = data[3]
            batch_inputs = data[4: -1]
            batch_target = data[-1]
            batch_size = batch_target.size()[0]

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input, volatile=True)
                    if self.trainer.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target, volatile=True)
            if self.trainer.cuda:
                batch_target = batch_target.cuda()

            # TODO: CNN-Seq here
            # batch_output = self.trainer.model(*batch_inputs, hidden=hidden_from_CNNSeq)
            for e, (b, a) in enumerate(zip(batch_hsl, batch_audio)):
                b = np.expand_dims(b, 0)
                a = np.expand_dims(a, 0)
                h = self.trainer.model_cnnseq2sample(b, a)
                if e == 0:
                    batch_hidden = h
                else:
                    batch_hidden = torch.cat((batch_hidden, h), 1)
                    # print(np.shape(batch_output))
            batch_output = self.trainer.model(*batch_inputs, hidden=batch_hidden)
            # batch_output = self.trainer.model(*batch_inputs)
            loss_sum += self.trainer.criterion(batch_output, batch_target) \
                                    .data[0] * batch_size

            n_examples += batch_size

        n_examples = max(1, n_examples)
        return loss_sum / n_examples


class AbsoluteTimeMonitor(Monitor):

    stat_name = 'time'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 's')
        kwargs.setdefault('precision', 0)
        kwargs.setdefault('running_average', False)
        kwargs.setdefault('epoch_average', False)
        super(AbsoluteTimeMonitor, self).__init__(*args, **kwargs)
        self.start_time = None

    def _get_value(self, *args):
        if self.start_time is None:
            self.start_time = time.time()
        return time.time() - self.start_time


class SaverPlugin(Plugin):

    last_pattern = 'ep{}-it{}'
    best_pattern = 'best-ep{}-it{}'
    last_pattern_cnnseq2sample = 'cnnseq2sample-ep{}-it{}'
    best_pattern_cnnseq2sample = 'cnnseq2sample-best-ep{}-it{}'

    def __init__(self, checkpoints_path, keep_old_checkpoints):
        super().__init__([(1, 'epoch')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
            self._clear(self.last_pattern_cnnseq2sample.format('*', '*'))
        # Save SampleRNN model
        torch.save(
            self.trainer.model.state_dict(),
            os.path.join(
                self.checkpoints_path,
                self.last_pattern.format(epoch_index, self.trainer.iterations)
            )
        )
        # Save CNNSeq2SampleRNN full model
        torch.save(
            self.trainer.model_cnnseq2sample.state_dict(),
            os.path.join(
                self.checkpoints_path,
                self.last_pattern_cnnseq2sample.format(epoch_index, self.trainer.iterations)
            )
        )

        cur_val_loss = self.trainer.stats['validation_loss']['last']
        if cur_val_loss < self._best_val_loss:
            self._clear(self.best_pattern.format('*', '*'))
            self._clear(self.best_pattern_cnnseq2sample.format('*', '*'))
            torch.save(
                self.trainer.model.state_dict(),
                os.path.join(
                    self.checkpoints_path,
                    self.best_pattern.format(
                        epoch_index, self.trainer.iterations
                    )
                )
            )
            torch.save(
                self.trainer.model_cnnseq2sample.state_dict(),
                os.path.join(
                    self.checkpoints_path,
                    self.best_pattern_cnnseq2sample.format(
                        epoch_index, self.trainer.iterations
                    )
                )
            )
            self._best_val_loss = cur_val_loss

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)


class GeneratorPlugin(Plugin):

    pattern = 'ep{}-s{}.wav'

    def __init__(self, samples_path, n_samples, sample_length, sample_rate):
        super().__init__([(1, 'epoch')])
        self.samples_path = samples_path
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.sample_rate = sample_rate

    def register(self, trainer):
        self.generate = Generator(trainer.model.model, trainer.cuda)

    # New register to accept model and cuda setting (which tells whether we should use GPU or not)
    # -> audio generation after training
    def register_generate(self, model, cuda):
        self.generate = Generator(model, cuda)

    def epoch(self, epoch_index, initial_seq=None, hidden=None):
        samples = self.generate(self.n_samples, self.sample_length, initial_seq=initial_seq, hidden=hidden) \
                      .cpu().float().numpy()
        for i in range(self.n_samples):
            write_wav(
                os.path.join(
                    self.samples_path, self.pattern.format(epoch_index, i + 1)
                ),
                samples[i, :], sr=self.sample_rate, norm=True
            )


class GeneratorCNNSeq2SamplePlugin(Plugin):

    pattern = 'ep{}-s{}-em{}.wav'
    pattern_target_audio = 'ep{}-s{}-em{}_targetAudio.wav'
    pattern_video = 'ep{}-s{}-em{}.avi'
    pattern_video_target_audio = 'ep{}-s{}-em{}_targetAudio.avi'

    def __init__(self, samples_path, n_samples, sample_length, sample_rate):
        super().__init__([(1, 'epoch')])
        self.samples_path = samples_path
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.sample_rate = sample_rate

    # New register to accept model and cuda setting (which tells whether we should use GPU or not)
    # -> audio generation after training
    def register_generate(self, model, model_cnnseq2sample, cuda):
        generator = Generator(model, cuda)
        self.generate_cnnseq2sample = GeneratorCNNSeq2Sample(generator, model_cnnseq2sample, cuda)

    def epoch(self, test_data_loader, epoch_index):
        # samples = self.generate_cnnseq2sample(test_data_loader, self.n_samples, self.sample_length) \
        #               .cpu().float().numpy()
        samples, input, target_audio, emotion = self.generate_cnnseq2sample(test_data_loader, self.n_samples, self.sample_length)
        print("Shape: samples {}, input {}, target_audio {}, sr {}, emotion {}".
              format(np.shape(samples), np.shape(input), np.shape(target_audio), self.sample_rate, emotion))
        for i in range(self.n_samples):
            filename_sample = os.path.join(self.samples_path,
                                           self.pattern.format(epoch_index, i + 1, emotion[i]))
            write_wav(filename_sample, samples[i, :], sr=self.sample_rate, norm=True)

            filename_target_audio = os.path.join(self.samples_path, self.pattern_target_audio.format(epoch_index, i + 1, emotion[i]))
            write_wav(filename_target_audio, target_audio[i], sr=self.sample_rate, norm=True)

            # Save input as video with audio as samples
            print(np.shape(np.array(input[i]).squeeze()))
            input_reshaped = np.array(input[i]).squeeze()

            from skimage import color
            frame_arr = []
            for frame in input_reshaped:
                frame = np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2)  # np.transpose(frame)  # 100, 100, 3
                frame = utils.normalize(frame, min=0, max=1)
                frame_rgb = color.hsv2rgb(frame)
                frame_arr.append(frame_rgb * 255)
            print("frame_arr: {}, min: {}, max: {}".
                  format(np.shape(frame_arr), np.amin(frame_arr), np.amax(frame_arr)))
            clip = ImageSequenceClip(np.array(frame_arr), fps=10)  # 3-second clip, .tolist()
            # audio_sample = np.expand_dims(samples[i], 0)  # samples[i, :]
            # clip.set_audio(AudioArrayClip(audio_sample, fps=16000))
            filename = os.path.join(self.samples_path, self.pattern_video.format(epoch_index, i + 1, emotion[i]))
            clip.write_videofile(filename, fps=10, codec='png', audio_fps=self.sample_rate, audio=filename_sample)  # export as video

            # audio_sample = np.expand_dims(target_audio[i], 0)  # target_audio[i, :]
            # clip.set_audio(AudioArrayClip(audio_sample, fps=16000))
            filename = os.path.join(self.samples_path, self.pattern_video_target_audio.format(epoch_index, i + 1, emotion[i]))
            clip.write_videofile(filename, fps=10, codec='png', audio_fps=self.sample_rate, audio=filename_target_audio)  # export as video


class StatsPlugin(Plugin):

    data_file_name = 'stats.pkl'
    plot_pattern = '{}.svg'

    def __init__(self, results_path, iteration_fields, epoch_fields, plots):
        super().__init__([(1, 'iteration'), (1, 'epoch')])
        self.results_path = results_path

        self.iteration_fields = self._fields_to_pairs(iteration_fields)
        self.epoch_fields = self._fields_to_pairs(epoch_fields)
        self.plots = plots
        self.data = {
            'iterations': {
                field: []
                for field in self.iteration_fields + [('iteration', 'last')]
            },
            'epochs': {
                field: []
                for field in self.epoch_fields + [('iteration', 'last')]
            }
        }

    def register(self, trainer):
        self.trainer = trainer

    def iteration(self, *args):
        for (field, stat) in self.iteration_fields:
            self.data['iterations'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['iterations']['iteration', 'last'].append(
            self.trainer.iterations
        )

    def epoch(self, epoch_index):
        for (field, stat) in self.epoch_fields:
            self.data['epochs'][field, stat].append(
                self.trainer.stats[field][stat]
            )

        self.data['epochs']['iteration', 'last'].append(
            self.trainer.iterations
        )

        data_file_path = os.path.join(self.results_path, self.data_file_name)
        with open(data_file_path, 'wb') as f:
            pickle.dump(self.data, f)

        for (name, info) in self.plots.items():
            x_field = self._field_to_pair(info['x'])

            try:
                y_fields = info['ys']
            except KeyError:
                y_fields = [info['y']]

            labels = list(map(
                lambda x: ' '.join(x) if type(x) is tuple else x,
                y_fields
            ))
            y_fields = self._fields_to_pairs(y_fields)

            try:
                formats = info['formats']
            except KeyError:
                formats = [''] * len(y_fields)

            pyplot.gcf().clear()

            for (y_field, format, label) in zip(y_fields, formats, labels):
                if y_field in self.iteration_fields:
                    part_name = 'iterations'
                else:
                    part_name = 'epochs'

                xs = self.data[part_name][x_field]
                ys = self.data[part_name][y_field]

                pyplot.plot(xs, ys, format, label=label)

            if 'log_y' in info and info['log_y']:
                pyplot.yscale('log')

            pyplot.legend()
            pyplot.savefig(
                os.path.join(self.results_path, self.plot_pattern.format(name))
            )

    @staticmethod
    def _field_to_pair(field):
        if type(field) is tuple:
            return field
        else:
            return (field, 'last')

    @classmethod
    def _fields_to_pairs(cls, fields):
        return list(map(cls._field_to_pair, fields))


class CometPlugin(Plugin):

    def __init__(self, experiment, fields):
        super().__init__([(1, 'epoch')])

        self.experiment = experiment
        self.fields = [
            field if type(field) is tuple else (field, 'last')
            for field in fields
        ]

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        for (field, stat) in self.fields:
            self.experiment.log_metric(field, self.trainer.stats[field][stat])
        self.experiment.log_epoch_end(epoch_index)

import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init, Linear

import numpy as np
from cnnseq.utils_models import load_json
from cnnseq.CNNSeq2Seq2 import load_cnnseq2seq, get_hidden_state
from cnnseq.CNNSeq2Seq2_main import feats_tensor_input, feats_tensor_audio


class CNNSeq2SampleRNN(torch.nn.Module):
    def __init__(self):
        super(CNNSeq2SampleRNN, self).__init__()

        # Load pre-trained CNN-Seq2Seq
        self.cnnseq2seq_model, self.cnnseq2seq_params = load_cnnseq2seq()
        self.hidden_size = self.cnnseq2seq_params['hidden_size']
        self.num_layers = self.cnnseq2seq_params['num_layers']

        self.batch_size = 1  # self.samplernn_model.model.batch_size

        #self.fc = Linear(self.num_layers*1*self.hidden_size,
        #                 self.num_layers*self.samplernn_model.batch_size*self.samplernn_model.dim)

        #self.fc = Linear(self.num_layers * 1 * self.hidden_size,
        #                 self.num_layers * self.hidden_size * 1024)  # 2, 128, 1024

        self.fc = Linear(self.num_layers * self.batch_size * self.hidden_size,
                         self.num_layers * self.batch_size * 1024)  # 2, 128, 1024

    def forward(self, x, y):
        # Assume batch_size = 1
        # print("batch_hsl: {}, batch_audio: {}".format(np.shape(x), np.shape(y)))
        batch_hsl_tensor = feats_tensor_input(x, data_type='HSL')
        batch_audio_tensor = feats_tensor_audio(y)
        hidden_enc_arr = get_hidden_state(self.cnnseq2seq_model, batch_hsl_tensor, batch_audio_tensor, self.cnnseq2seq_params)
        hidden_from_CNNSeq = hidden_enc_arr[0]
        hidden_from_CNNSeq = hidden_from_CNNSeq
        # Considers n_rnn = 2 (self.num_layers in CNNSeq2Seq)
        # hidden_from_CNNSeq_proj = self.fc(hidden_from_CNNSeq)
        hidden_from_CNNSeq_0_proj_cpu = hidden_from_CNNSeq[0]  # torch.tensor().device(torch.device('cpu'))
        hidden_0_flatten = hidden_from_CNNSeq_0_proj_cpu.view(-1)
        hidden_from_CNNSeq_0_proj = self.fc(hidden_0_flatten)
        hidden_from_CNNSeq_0_proj = hidden_from_CNNSeq_0_proj.view(self.num_layers, 1, 1024)
        hidden_from_CNNSeq_1_proj_cpu = hidden_from_CNNSeq[1]  # torch.tensor().device(torch.device('cpu'))
        hidden_1_flatten = hidden_from_CNNSeq_1_proj_cpu.view(-1)
        hidden_from_CNNSeq_1_proj = self.fc(hidden_1_flatten)
        hidden_from_CNNSeq_1_proj = hidden_from_CNNSeq_1_proj.view(self.num_layers, 1, 1024)
        # hidden_from_CNNSeq_1_proj = self.fc(torch.FloatTensor(hidden_from_CNNSeq[1]).detach().cpu())
        # hidden_from_CNNSeq_1_proj = hidden_from_CNNSeq_1_proj.view(1, 1024)
        hidden_from_CNNSeq_tensor = []
        hidden_from_CNNSeq_tensor.append(hidden_from_CNNSeq_0_proj)
        hidden_from_CNNSeq_tensor.append(hidden_from_CNNSeq_1_proj)
        # hidden_from_CNNSeq_tensor = torch.cat([torch.LongTensor(hidden_from_CNNSeq_0_proj), hidden_from_CNNSeq_1_proj])
        # hidden_cnn = torch.LongTensor(self.model.n_rnn, self.model.batch_size, self.model.dim).fill_(0)
        return hidden_from_CNNSeq_0_proj

    @property
    def lookback(self):
        return self.samplernn_model.frame_level_rnns[-1].n_frame_samples


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels,
                 weight_norm, batch_size):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        self.batch_size = batch_size
        self.n_rnn = n_rnn

        # Add CNN and RNN encoder -> (2, 128, 1024)
        # hidden0 = torch.LongTensor((n_rnn, batch_size, self.dim)).fill_(utils.q_zero(self.q_levels))
        # hidden0 = torch.LongTensor(n_rnn, batch_size, self.dim).fill_(0)
        # self.hidden_cnn = torch.LongTensor(n_rnn, batch_size, self.dim).fill_(0)

        ns_frame_samples = map(int, np.cumprod(frame_sizes))
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, learn_h0, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ])

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim,
                 learn_h0, weight_norm):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            self.register_buffer('h0', torch.autograd.Variable(h0))

        self.input_expand = torch.nn.Conv1d(
            in_channels=n_frame_samples,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.input_expand.weight)
        init.constant_(self.input_expand.bias, 0)
        if weight_norm:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        for i in range(n_rnn):
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal]
            )
            init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.upsampling = nn.LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )
        init.uniform_(
            self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        )
        init.constant_(self.upsampling.bias, 0)
        if weight_norm:
            self.upsampling.conv_t = torch.nn.utils.weight_norm(
                self.upsampling.conv_t
            )

    def forward(self, prev_samples, upper_tier_conditioning, hidden):
        (batch_size, _, _) = prev_samples.size()

        input = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)
        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()

        # RNN (in this case GRU running)
        #hidden = hidden.detach()
        #hidden = hidden.cpu().long()
        (output, hidden) = self.rnn(input, hidden)

        output = self.upsampling(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return (output, hidden)


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.input = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=frame_size,
            bias=False
        )
        init.kaiming_uniform_(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)

        self.hidden = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.hidden.weight)
        init.constant_(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)

        self.output = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        nn.lecun_uniform(self.output.weight)
        init.constant_(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()

        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        return F.log_softmax(x.view(-1, self.q_levels)) \
                .view(batch_size, -1, self.q_levels)


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    # Make it conditional on a specific hidden state
    def reset_hidden_states(self, hidden=None):
        # self.hidden_states = {rnn: hidden for rnn in self.model.frame_level_rnns}
        # hidden_cnn = torch.LongTensor(self.model.n_rnn, self.model.batch_size, self.model.dim).fill_(0)
        self.hidden_states = {rnn: hidden for rnn in self.model.frame_level_rnns}
        # print("hidden states shape: {}".format(np.shape(self.hidden_states)))

    # self.hidden_states <- condition
    def run_rnn(self, rnn, prev_samples, upper_tier_conditioning):
        (output, new_hidden) = rnn(
            prev_samples, upper_tier_conditioning, self.hidden_states[rnn]
        )
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset, hidden=None):
        if reset:
            self.reset_hidden_states(hidden=hidden)

        (batch_size, _) = input_sequences.size()

        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * utils.linear_dequantize(
                input_sequences[:, from_index : to_index],
                self.model.q_levels
            )
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )

            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, upper_tier_conditioning
            )

        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences \
            [:, self.model.lookback - bottom_frame_size :]

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len, initial_seq=None, hidden=None, verbose=False):
        # generation doesn't work with CUDNN for some reason
        torch.backends.cudnn.enabled = False

        # CNN gives hidden state
        self.reset_hidden_states(hidden=hidden)

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = torch.LongTensor(n_seqs, self.model.lookback + seq_len) \
            .fill_(utils.q_zero(self.model.q_levels))
        if initial_seq is None:
            initial_i = self.model.lookback
            final_i = initial_i + seq_len
        else:  # CONDITIONAL
            sequences[:, 0:np.shape(initial_seq)[1]] = initial_seq
            initial_i = np.shape(initial_seq)[1] - self.model.lookback
            # initial_i = np.shape(initial_seq)[1] + self.model.lookback
            final_i = self.model.lookback + seq_len
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        for i in range(initial_i, final_i):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                prev_samples = torch.autograd.Variable(
                    2 * utils.linear_dequantize(
                        sequences[:, i - rnn.n_frame_samples : i],
                        self.model.q_levels
                    ).unsqueeze(1),
                    volatile=True
                )
                # print("Tier {}: prev_samples from {} to {}, shape {}: {}".format(tier_index, i - rnn.n_frame_samples, i, np.shape(prev_samples), prev_samples))
                if self.cuda:
                    prev_samples = prev_samples.cuda()

                l = len(self.model.frame_level_rnns) - 1
                if tier_index == l:
                    if verbose:
                        print("No upper tier conditioning")
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                           .unsqueeze(1)
                    if verbose:
                        print("Frame index {}, upper_tier_conditioning shape {}".format(frame_index, np.shape(upper_tier_conditioning)))

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples, upper_tier_conditioning
                )
                if verbose:
                    print("Tier {} frame level outputs shape {}".format(tier_index, np.shape(frame_level_outputs[tier_index])))

            # print(sequences[:, i - bottom_frame_size : i])
            prev_samples = torch.autograd.Variable(
                sequences[:, i - bottom_frame_size : i],
                volatile=True
            )
            # print("Tier {}: prev_samples from {} to {}, shape {}: {}".format(tier_index, i - bottom_frame_size, i, np.shape(prev_samples), prev_samples))
            if self.cuda:
                prev_samples = prev_samples.cuda()
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                                      .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(prev_samples, upper_tier_conditioning)
            sample_dist = sample_dist.squeeze(1).exp_().data
            if verbose:
                print("Sample dist {}".format(np.shape(sample_dist)))
                print("Before: {}".format(sequences[:, i]))
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)
            if verbose:
                print("After {}".format(sequences[:, i]))

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback :]


class GeneratorCNNSeq2Sample:

    def __init__(self, generator, model_cnnseq2sample, cuda=False):
        self.generator = generator
        self.cuda = cuda
        self.model_cnnseq2sample = model_cnnseq2sample

    def __call__(self, test_data_loader, n_seqs, seq_len):
        for e, data in enumerate(test_data_loader):
            batch_hsl = data[0]
            batch_audio = data[1]
            batch_emotion = data[2]
            batch_text = data[3]
            batch_inputs = data[4: -1]
            batch_target = data[-1]
            break

        # CNN-Seq2Sample here
        input, target_audio, emotion = [], [], []
        for e, (b, a, em) in enumerate(zip(batch_hsl, batch_audio, batch_emotion)):
            if e >= n_seqs:
                break
            b = np.expand_dims(b, 0)  # b.unsqueeze(0)
            a = np.expand_dims(a, 0)  # a.unsqueeze(0)
            #  print("b: {}, a: {}, i: {}".format(np.shape(b), np.shape(a), np.shape(i)))
            h = self.model_cnnseq2sample(b, a)
            if e == 0:
                batch_hidden = h
            else:
                batch_hidden = torch.cat((batch_hidden, h), 1)  # concat on position 1
                #  print(np.shape(batch_output))
            # batch_output = model(*batch_inputs, hidden=batch_hidden)
            input.append(b)
            target_audio.append(np.array(np.reshape(a, [-1, seq_len])).squeeze())
            emotion.append(em)
            # TODO: self.generator should be in the for loop
        samples = self.generator(n_seqs, seq_len, hidden=batch_hidden).cpu().float().numpy()
        return samples, input, target_audio, emotion

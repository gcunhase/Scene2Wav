import torch
from torch.autograd import Variable
from cnnseq.utils_models import load_json
from cnnseq.CNNSeq2Seq2 import load_cnnseq2seq, get_hidden_state
from cnnseq.CNNSeq2Seq2_main import feats_tensor_input, feats_tensor_audio
import heapq
import numpy as np
from model import CNNSeq2SampleRNN


# Based on torch.utils.trainer.Trainer code.
# Allows multiple inputs to the model, not all need to be Tensors.
class Trainer(object):

    def __init__(self, model, model_cnnseq2sample, criterion, optimizer, dataset, cuda=False):
        self.model = model
        self.model_cnnseq2sample = model_cnnseq2sample
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.cuda = cuda
        self.iterations = 0
        self.epochs = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for self.epochs in range(self.epochs + 1, self.epochs + epochs + 1):
            self.train()
            self.call_plugins('epoch', self.epochs)

    def train(self):
        for (self.iterations, data) in \
                enumerate(self.dataset, self.iterations + 1):
            batch_hsl = data[0]
            batch_audio = data[1]
            batch_inputs = data[2: -1]
            batch_target = data[-1]
            self.call_plugins(
                'batch', self.iterations, batch_inputs, batch_target
            )

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input)
                    if self.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target)
            if self.cuda:
                batch_target = batch_target.cuda()

            plugin_data = [None, None]

            def closure():
                # TODO: CNN-Seq here, for with batch size?
                # cnnseq_model = CNNSeq2SampleRNN(self.model).cuda()
                reset = batch_inputs[1]
                for e, (b, a, i) in enumerate(zip(batch_hsl, batch_audio, batch_inputs[0])):
                    b = np.expand_dims(b, 0)  # b.unsqueeze(0)
                    a = np.expand_dims(a, 0)  # a.unsqueeze(0)
                    i = i.unsqueeze(0)
                    # print("b: {}, a: {}, i: {}".format(np.shape(b), np.shape(a), np.shape(i)))
                    o = self.model_cnnseq2sample(b, a, i, reset)
                    if e == 0:
                        batch_output = o
                    else:
                        batch_output = torch.cat((batch_output, o), 0)
                    # print(np.shape(batch_output))
                # print("hsl: {}, audio: {}, inputs: {}".format(np.shape(batch_hsl), np.shape(batch_audio), np.shape(batch_inputs[0])))
                # batch_output = self.model(batch_hsl, batch_audio, batch_inputs)  # , reset=batch_inputs[1]
                # batch_output = self.model(*batch_inputs)

                loss = self.criterion(batch_output, batch_target)
                loss.backward()

                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins(
                'iteration', self.iterations, batch_inputs, batch_target,
                *plugin_data
            )
            self.call_plugins('update', self.iterations, self.model)

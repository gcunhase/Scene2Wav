
# t-SNE
from sklearn.manifold import TSNE

# PCA
from sklearn.decomposition import PCA
from plotnine import *  # from ggplot import *

# Others
import unittest
from cnnseq.CNNSeq2Seq2 import load_cnnseq2seq, get_h0
from cnnseq.CNNSeq2Seq2_main import feats_tensor_input, feats_tensor_audio
import numpy as np
from cnnseq.utils import project_dir_name, ensure_dir
import os
import math
import pandas as pd


__author__ = "Gwena Cunha"

"""
Class to plot visual features, target and obtained audio along with their emotional label
TODO:
    HSL input
    Visual features (hidden state h0 for layers 1 and 2 of model), shape: (2, 1, 128)
    Target audio
    MAYBE: Generated audio
"""

CLASS = {0: 'Negative', 1: 'Positive'}

class DataAnalysis:
    def __init__(self, data_filename, cnn_pth, cnn_seq2seq_pth, results_dir=''):
        self.results_dir = results_dir
        ensure_dir(self.results_dir)
        self.data = self.load_data(data_filename)
        self.cnnseq2seq_model, self.cnnseq2seq_params = self.load_model(cnn_pth, cnn_seq2seq_pth)

    def load_data(self, data_filename):
        """
        Data contains 4 fields: audio, HSL_data, emotion and text
        :param data_filename: filename with data
        :return: data
        """
        data = np.load(data_filename)
        return data

    def load_model(self, cnn_pth, cnn_seq2seq_pth):
        """
        Load model from CNN and CNN-Seq2Seq pre-trained weights
        :param cnn_pth: filename to CNN pre-trained model
        :param cnn_seq2seq_pth: filename to CNN-Seq2Seq pre-trained model
        :return: CNN-Seq2Seq model
        """
        # print("Load CNN-Seq2Seq model for hidden visual features")
        cnnseq2seq_model, cnnseq2seq_params = load_cnnseq2seq(cnn_pth, cnn_seq2seq_pth)
        return cnnseq2seq_model, cnnseq2seq_params

    def hsl_input(self):
        hsl_data = self.data['HSL_data']
        hsl_data = np.reshape(hsl_data, [np.shape(hsl_data)[0], -1])
        return hsl_data

    def visual_feats(self):
        print("Visual features")
        HSL_data = self.data['HSL_data']
        visual_input_tensors = feats_tensor_input(HSL_data, data_type='HSL')

        audio = self.data['audio']
        # Reshape 48,000 -> 8*6,000
        audio_n_prediction = self.cnnseq2seq_params['audio_n_prediction']
        y_audio_dim = int(math.ceil(np.shape(audio)[1] / audio_n_prediction))
        audio = np.reshape(audio, [-1, audio_n_prediction, y_audio_dim])
        audio_input_tensors = feats_tensor_audio(audio)

        hidden_info = get_h0(self.cnnseq2seq_model, visual_input_tensors, audio_input_tensors, self.cnnseq2seq_params)
        h_dim1, h_dim2 = [], []
        for h in hidden_info:
            h_0 = h[0].squeeze().cpu().detach().numpy()  # view(-1, np.shape(hidden_info[0]))
            h_1 = h[1].squeeze().cpu().detach().numpy()
            h_dim1.append(h_0)
            h_dim2.append(h_1)

        return h_dim1, h_dim2

    def save_data_in_csv(self, X, y, filename='test.csv'):
        df = pd.DataFrame(X)
        df.to_csv(filename, index=False)

        csv_input = pd.read_csv(filename)
        csv_input['emotion'] = y
        csv_input.to_csv(filename, index=False)

    def tsne_plot(self, X, y=None, n_components=5, learning_rate=200.0, n_iter=300, perplexity=40, type='feats'):
        print("Tsne plot - comp {}, lr {}, iter {}, perp {}, type {}".format(n_components, learning_rate, n_iter, perplexity, type))
        if y is None:
            y = self.data['emotion']
        print("X: {}, y: {}".format(np.shape(X), np.shape(y)))
        X = np.array(X)
        df = pd.DataFrame(X)
        df['label'] = y
        # self.save_data_in_csv(X, y, filename=project_dir_name()+'data_analysis/test.csv')
        df['label_name'] = [CLASS[y_tmp] for y_tmp in y]

        pca_n = PCA(n_components=n_components, whiten=True)
        pca_result_n = pca_n.fit_transform(X, y)
        print('Cumulative explained variation for 50 principal components: {}'.
              format(np.sum(pca_n.explained_variance_ratio_)))

        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
        tsne_pca_results = tsne.fit_transform(pca_result_n)
        df['x-tsne-pca'] = tsne_pca_results[:, 0]
        df['y-tsne-pca'] = tsne_pca_results[:, 1]

        chart = ggplot(df, aes(x='x-tsne-pca', y='y-tsne-pca', color='label_name')) \
                + geom_point(size=0.5, alpha=0.5) \
                + ggtitle("tSNE dimensions colored by class (PCA)")
        chart.save(self.results_dir + 'lr{}_com{}_perp{}_iter{}_{}.png'.format(learning_rate, n_components, perplexity,
                                                                               n_iter, type))
        chart = ggplot(df, aes(x='x-tsne-pca', y='y-tsne-pca', color='label_name')) \
                + geom_point(size=0.5, alpha=0.5) \
                + ggtitle("tSNE dimensions colored by class (PCA)") \
                + facet_wrap('~label')
        chart.save(self.results_dir + 'lr{}_com{}_perp{}_iter{}_{}_separate.png'.format(learning_rate, n_components,
                                                                                        perplexity, n_iter, type))


'''
class TestDataAnalysis(unittest.TestCase):
    def runTest(self):
        print("Run test")
        data_filename = project_dir_name() + 'datasets/data_npz/video_feats_HSL_10fps_pad_test.npz'
        cnn_pth = project_dir_name() + 'cnn2_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf'
        cnn_seq2seq_pth = 'cnnseq2seq2_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio'
        self.dataAnalysis = DataAnalysis(data_filename, cnn_pth, cnn_seq2seq_pth)

    def test_visual_feats(self):
        self.dataAnalysis.visual_feats()
'''

if __name__ == '__main__':
    # unittest.main()
    print("Run test")
    results_dir = project_dir_name() + 'data_analysis/results/'
    data_filename = project_dir_name() + 'datasets/data_npz/video_feats_HSL_10fps_pad_test.npz'
    cnn_pth = 'cnnseq/cnn2_res_vanilla_HSL_bin_1D_CrossEntropy_ep_40_bs_30_lr_0.001_we_0.0001_adam_95.36perf/'
    cnn_seq2seq_pth = 'cnnseq/cnnseq2seq2_HSL_bin_1D_res_stepPred_8_ep_20_bs_30_relu_layers_2_size_128_lr_0.001_we_1e-05_asgd_trainSize_3177_testSize_1137_cost_audio/'
    dataAnalysis = DataAnalysis(data_filename, cnn_pth, cnn_seq2seq_pth, results_dir=results_dir)
    h_dim1, h_dim2 = dataAnalysis.visual_feats()
    #dataAnalysis.tsne_plot(h_dim1, n_components=30, learning_rate=200.0, n_iter=300, perplexity=40, type='feats')
    #dataAnalysis.tsne_plot(h_dim1, n_components=30, learning_rate=200.0, n_iter=3000, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=25, learning_rate=200.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=25, learning_rate=200.0, n_iter=3000, perplexity=40, type='feats')


    '''
    dataAnalysis.tsne_plot(h_dim1, n_components=5, learning_rate=100.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=10, learning_rate=100.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=15, learning_rate=100.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=20, learning_rate=100.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=5, learning_rate=200.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=10, learning_rate=200.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=15, learning_rate=200.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=20, learning_rate=200.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=5, learning_rate=500.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=10, learning_rate=500.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=15, learning_rate=500.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=20, learning_rate=500.0, n_iter=300, perplexity=40, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=5, learning_rate=200.0, n_iter=300, perplexity=50, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=10, learning_rate=200.0, n_iter=300, perplexity=60, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=15, learning_rate=200.0, n_iter=300, perplexity=80, type='feats')
    dataAnalysis.tsne_plot(h_dim1, n_components=20, learning_rate=200.0, n_iter=300, perplexity=100, type='feats')
    '''

    hsl_in = dataAnalysis.hsl_input()
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=100.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=10, learning_rate=100.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=15, learning_rate=100.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=20, learning_rate=100.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=200.0, n_iter=300, perplexity=40, type='HSL')
    # dataAnalysis.tsne_plot(hsl_in, n_components=10, learning_rate=200.0, n_iter=300, perplexity=40, type='HSL')
    # dataAnalysis.tsne_plot(hsl_in, n_components=80, learning_rate=200.0, n_iter=300, perplexity=20, type='HSL')
    # dataAnalysis.tsne_plot(hsl_in, n_components=40, learning_rate=200.0, n_iter=300, perplexity=40, type='HSL')
    # dataAnalysis.tsne_plot(hsl_in, n_components=30, learning_rate=200.0, n_iter=300, perplexity=40, type='HSL')
    # dataAnalysis.tsne_plot(hsl_in, n_components=25, learning_rate=200.0, n_iter=3000, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=15, learning_rate=200.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=20, learning_rate=200.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=500.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=10, learning_rate=500.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=15, learning_rate=500.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=20, learning_rate=500.0, n_iter=300, perplexity=40, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=200.0, n_iter=300, perplexity=5, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=200.0, n_iter=300, perplexity=10, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=200.0, n_iter=300, perplexity=15, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=5, learning_rate=200.0, n_iter=300, perplexity=50, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=10, learning_rate=200.0, n_iter=300, perplexity=60, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=15, learning_rate=200.0, n_iter=300, perplexity=80, type='HSL')
    #dataAnalysis.tsne_plot(hsl_in, n_components=20, learning_rate=200.0, n_iter=300, perplexity=100, type='HSL')

import numpy as np
from modules import utils
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random

# Tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Univariate
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# t-SNE
from sklearn.manifold import TSNE

# PCA
from sklearn.decomposition import PCA
from plotnine import *  # from ggplot import *

# UMAP
import umap
import seaborn as sns
import datashader as ds
import requests
import os
import datashader as ds
import datashader.utils as ds_utils
import datashader.transfer_functions as tf


MARKERS = ["o", "D"]
ROOT_DIR = utils.project_dir_name()
RESULTS_DIR = ROOT_DIR + 'modules_feature_selection/res_umap_feat_selection/'
utils.ensure_dir(RESULTS_DIR)
CLASS = {0: 'Negative', 1: 'Positive'}
CLASS2 = {'Negative': 0, 'Positive': 1}


def plot_radviz(filename, show=True, fig_filename=None):
    """ A final multivariate visualization technique pandas has is radviz which puts each feature as a point on a
       2D plane, and then simulates having each sample attached to those points through a spring weighted by the
       relative value for that feature

    :param filename: name of .csv file containing data
    :param show: indicates if plot is gonna be shown
    :param fig_filename: name of figure to save
    :return:
    """
    iris = pd.read_csv(filename)

    fig = plt.figure()

    from modules.pandas_plotting_misc import radviz

    radviz(iris, class_column="emotion", color=['red', 'blue'], hue_order=CLASS2.keys(), markers=MARKERS)

    plt.tight_layout()
    if show:
        plt.show()
    if fig_filename is not None:
        fig.savefig(fig_filename)


def save_data_in_csv(X, y, filename='test.csv'):
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)

    csv_input = pd.read_csv(filename)
    csv_input['emotion'] = y
    csv_input.to_csv(filename, index=False)


def feat_reduction_tree(X, y):
    print("\nTree-based feature selection")
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    array_feat_imp = clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print("X_shape: {}, X_new_shape: {}".format(X.shape, X_new.shape))
    # print("feature_importances: {}".format(array_feat_imp))

    # Save in .csv
    data_csv_filename = RESULTS_DIR + "tree.csv"
    save_data_in_csv(X_new, y, filename=data_csv_filename)
    plot_radviz(data_csv_filename, show=False, fig_filename=RESULTS_DIR + 'tree.png')

    return X_new


def feat_reduction_pca_tsne(df, n_components=5, learning_rate=200.0, n_iter=300, type=None):
    if type is not None:
        ext = '_{}'.format(type)
    else:
        ext = ''
    results_dir = RESULTS_DIR + 'pca{}_tsne_iter{}{}/'.format(n_components, n_iter, ext)
    utils.ensure_dir(results_dir)

    pca_n = PCA(n_components=n_components, whiten=True)
    pca_result_n = pca_n.fit_transform(X, y)
    print('Cumulative explained variation for 50 principal components: {}'.
          format(np.sum(pca_n.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=n_iter, learning_rate=learning_rate)
    tsne_pca_results = tsne.fit_transform(pca_result_n)

    df_tsne = df.copy()
    df_tsne['x-tsne-pca'] = tsne_pca_results[:, 0]
    df_tsne['y-tsne-pca'] = tsne_pca_results[:, 1]

    chart = ggplot(df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='label_name')) \
            + geom_point(size=0.5, alpha=0.3) \
            + ggtitle("tSNE dimensions colored by class (PCA)")
    chart.save(results_dir + 'lr{}.png'.format(learning_rate))
    chart = ggplot(df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='label_name')) \
            + geom_point(size=0.5, alpha=0.3) \
            + ggtitle("tSNE dimensions colored by class (PCA)") \
            + facet_wrap('~label')
    chart.save(results_dir + 'lr{}_separate.png'.format(learning_rate))

    # Saves features in pickle
    data_pca_tsne = {'pca': pca_n, 'pca_fit': pca_result_n, 'n_components': n_components,
                     'tsne': df_tsne, 'n_iter': n_iter, 'lr': learning_rate,
                     'tsne_res': tsne_pca_results, 'pca_res': np.transpose(pca_result_n),
                     'label': df_tsne['label'], 'label_name': df_tsne['label_name']}

    saved_data_path = results_dir + 'lr{}.pkl'.format(learning_rate)
    with open(saved_data_path, 'w') as f:
        pickle.dump(data_pca_tsne, f)

    return results_dir, saved_data_path


# TODO
def feat_reduction_umap(type=None, filename='data.csv', random_state=42):

    print("UMAP with random_state {}".format(random_state))

    if type is not None:
        ext = '_{}'.format(type)
    else:
        ext = ''

    filename = RESULTS_DIR + filename
    source_df = pd.read_csv(filename)

    data = source_df.iloc[:, :224].values.astype(np.float32)
    target = source_df['emotion'].values

    pal = [
        '#0000FF',
        '#FF0000'  # red
        # '#FFFF00'  # yellow
    ]
    color_key = {str(d): c for d, c in enumerate(pal)}

    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(data)

    df = pd.DataFrame(embedding, columns=('x', 'y'))
    df['class'] = pd.Series([str(int(x)) for x in target], dtype="category")
    # for k, v in df.iteritems():
    #     print("k: {}, v: {}".format(k, v))

    # cvs = ds.Canvas(plot_width=400, plot_height=400)
    cvs = ds.Canvas()
    agg = cvs.points(df, 'x', 'y', ds.count_cat('class'))
    img = tf.shade(agg, color_key=color_key, how='eq_hist')

    ds_utils.export_image(img, filename='cognimuse_{}'.format(random_state), export_path=RESULTS_DIR, background='white')

    image = plt.imread(RESULTS_DIR + 'cognimuse_{}.png'.format(random_state))
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(image)
    plt.setp(ax, xticks=[], yticks=[])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("COGNIMUS data data embedded\n"
              "into two dimensions by UMAP\n"
              "visualised with Datashader",
              fontsize=12)

    plt.savefig(RESULTS_DIR + 'cognimuse_{}_labels.png'.format(random_state))



def save_train_test_data(results_dir, saved_data_path, train_perc=0.9, shuffle=False, verbose=False):
    """ Dataset info

        150 samples (50 per class) with 4 attributes
        Gets train_num per class then shuffle

    :param results_dir: directory with data information
    :param saved_data_path: path to pickle with saved data after PCA + t-SNE
    :param train_perc: percentage of data to be considered in the train dataset
    :param shuffle: boolean to shuffle data
    :param verbose: verbose printing for debugging
    :return:
    """

    with open(saved_data_path, 'r') as f:
        data_res = pickle.load(f)

    label = data_res['label']
    label_name = data_res['label_name']
    tsne_res = data_res['tsne_res']
    pca_res = data_res['pca_res']

    # attributes = attributes[0:100, :]
    # label = label[0:100]

    len_att = len(label)
    train_size = int(round(train_perc * len_att))
    test_size = len_att - train_size

    tsne_res = np.transpose(tsne_res)

    label_train = label[0:train_size]
    label_name_train = label_name[0:train_size]
    tsne_res_train = tsne_res[:, 0:train_size]
    pca_res_train = pca_res[:, 0:train_size]

    label_test = label[train_size:len_att]
    label_name_test = label_name[train_size:len_att]
    tsne_res_test = tsne_res[:, train_size:len_att]
    pca_res_test = pca_res[:, train_size:len_att]

    if verbose:
        print("Train - label: {}, label_name: {}, tsne: {}, pca: {}".format(np.shape(label_train),
                                                                            np.shape(label_name_train),
                                                                            np.shape(tsne_res_train),
                                                                            np.shape(pca_res_train)))
        print("Test - label: {}, label_name: {}, tsne: {}, pca: {}".format(np.shape(label_test),
                                                                           np.shape(label_name_test),
                                                                           np.shape(tsne_res_test),
                                                                           np.shape(pca_res_test)))

    if shuffle:
        c1 = list(zip(label_train, label_name_train, tsne_res_train, pca_res_train))
        random.shuffle(c1)
        label_train, label_name_train, tsne_res_train, pca_res_train = zip(*c1)

        c2 = list(zip(label_test, label_name_test, tsne_res_test, pca_res_test))
        random.shuffle(c2)
        label_test, label_name_test, tsne_res_test, pca_res_test = zip(*c2)

    dict = {'label': label_train, 'label_name': label_name_train, 'tsne_res': tsne_res_train, 'pca_res': pca_res_train}
    saved_res_path = saved_data_path.split('.pkl')[0] + '_train.pkl'
    with open(saved_res_path, 'w') as f:
        pickle.dump(dict, f)

    dict = {'label': label_test, 'label_name': label_name_test, 'tsne_res': tsne_res_test, 'pca_res': pca_res_test}
    saved_res_path = saved_data_path.split('.pkl')[0] + '_test.pkl'
    with open(saved_res_path, 'w') as f:
        pickle.dump(dict, f)


if __name__ == '__main__':
    # --------------- Feature selection ALL DATA ---------------
    # Data name
    data_filename = 'shuffled_video_feats_and_labels_BMI_CHI_CRA_DEP_FNE_GLA_LOR.npz'
    print(ROOT_DIR)
    data_dir = ROOT_DIR + 'data/'
    input_file = data_dir + data_filename

    # Load data
    # subset_num = 10
    data = np.load(input_file)
    y = data['labels']  # [0:subset_num]
    y_name = []
    for y_t in y:
        y_t = int(y_t)
        y_name.append(CLASS[y_t])
    # print("y_name shape: {}, values: {}".format(np.shape(y_name), y_name))
    X = data['video_feats']  # [0:subset_num]

    print("data shape: {}, label: {}".format(np.shape(X), np.shape(y)))

    # Save in .csv
    save_data_in_csv(X, y, filename=RESULTS_DIR + 'data.csv')
    # plot_radviz(RESULTS_DIR + 'data.csv', show=False, fig_filename=RESULTS_DIR + 'data.png')

    # Tree-based feature selection
    # X_tree = feat_reduction_tree(X, y)

    # Univariate feature selection
    # print("\nUnivariate feature selection")
    # k = 4
    # X_new = SelectKBest(chi2, k=k).fit_transform(X, y)
    # print("X_shape: {}, X_new_shape: {}".format(X.shape, X_new.shape))
    # save_data_in_csv(X_new, y, filename=RESULTS_DIR + 'uni_{}.csv'.format(k))
    # plot_radviz(RESULTS_DIR + 'uni_{}.csv'.format(k), show=False, fig_filename=RESULTS_DIR + 'uni_{}.png'.format(k))

    n_components = 10  # 6
    '''
    # PCA
    n_components = 10 # 6
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X, y)

    df = pd.DataFrame(X)
    df['label_name'] = y_name
    df['label'] = y
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    print("PCA shapes: {}, {}".format(np.shape(df['pca-one']), np.shape(df['pca-two'])))

    chart = ggplot(df.loc[:, :], aes(x='pca-one', y='pca-two', color='label_name')) \
            + geom_point(size=0.5, alpha=0.3) \
            + ggtitle("First and Second Principal Components colored by class")
    chart.save(RESULTS_DIR + 'pca.png')
    chart = ggplot(df.loc[:, :], aes(x='pca-one', y='pca-two', color='label_name')) \
            + geom_point(size=0.5, alpha=0.3) \
            + ggtitle("First and Second Principal Components colored by class") \
            + facet_wrap('~label')
    chart.save(RESULTS_DIR + 'pca_separate.png')
    '''

    '''
    # t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X, y)

    df_tsne = df.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]

    chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label_name')) \
            + geom_point(size=0.5, alpha=0.3) \
            + ggtitle("tSNE dimensions colored by class")
    chart.save(RESULTS_DIR + 'tsne.png')
    chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label_name')) \
            + geom_point(size=0.5, alpha=0.3) \
            + ggtitle("tSNE dimensions colored by class") \
            + facet_wrap('~label')
    chart.save(RESULTS_DIR + 'tsne_separate.png')
    '''

    '''
    # PCA + t-SNE
    df = pd.DataFrame(X)
    df['label_name'] = y_name
    df['label'] = y
    # results_dir, saved_data_path = feat_reduction_pca_tsne(df, n_components=n_components, learning_rate=200, n_iter=300)
    # Separate into train and test
    results_dir = RESULTS_DIR + 'pca10_tsne_iter300/'
    saved_data_path = results_dir + 'lr200.pkl'
    save_train_test_data(results_dir, saved_data_path, verbose=True)

    results_dir = RESULTS_DIR + 'pca10_tsne_iter300/'
    saved_data_path = results_dir + 'lr300.pkl'
    save_train_test_data(results_dir, saved_data_path, verbose=True)

    results_dir = RESULTS_DIR + 'pca10_tsne_iter300/'
    saved_data_path = results_dir + 'lr500.pkl'
    save_train_test_data(results_dir, saved_data_path, verbose=True)

    #results_dir, saved_data_path = feat_reduction_pca_tsne(df, n_components=n_components, learning_rate=300, n_iter=300)
    #save_train_test_data(results_dir, saved_data_path, verbose=True)

    #results_dir, saved_data_path = feat_reduction_pca_tsne(df, n_components=n_components, learning_rate=500, n_iter=300)
    #save_train_test_data(results_dir, saved_data_path, verbose=True)
    '''

    # UMAP
    # feat_reduction_umap(filename='data.csv', random_state=6)
    # feat_reduction_umap(filename='data.csv', random_state=10)
    # feat_reduction_umap(filename='data.csv', random_state=32)
    feat_reduction_umap(filename='data.csv', random_state=42)
    # feat_reduction_umap(filename='data.csv', random_state=100)
    # feat_reduction_umap(filename='data.csv', random_state=150)
    # feat_reduction_umap(filename='data.csv', random_state=200)
    # feat_reduction_umap(filename='data.csv', random_state=300)
    # feat_reduction_umap(filename='data.csv', random_state=1000)

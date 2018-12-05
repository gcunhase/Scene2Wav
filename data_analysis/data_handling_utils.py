import matplotlib.pyplot as plt
import pandas as pd

MARKERS = ["o", "D"]
CLASS = {0: 'Negative', 1: 'Positive'}
CLASS2 = {'Negative': 0, 'Positive': 1}


def normalize_data(X, type='rescale'):
    # Normalizing data
    if type == 'rescale':
        # Rescale data (between 0 and 1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_norm = scaler.fit_transform(X)
    else:
        # Standardize data (0 mean, 1 stdev)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X_norm = scaler.transform(X)
    return X_norm


def save_data_in_csv(X, y, filename='test.csv', label_type='names'):
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)

    csv_input = pd.read_csv(filename)
    if label_type == 'names':
        label = []
        for y_t in y:
            label.append(CLASS[y_t])
    else:
        label = y

    csv_input['emotion'] = label
    csv_input.to_csv(filename, index=False)
    return label


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

    radviz(iris, class_column="emotion", color=['red', 'blue'], hue_order=CLASS2.keys(), markers=MARKERS, s=4)

    plt.tight_layout()
    if show:
        plt.show()
    if fig_filename is not None:
        fig.savefig(fig_filename)


def get_data_from_mat():
    print("Get data from .mat file")


if __name__ == '__main__':
    print("Test")

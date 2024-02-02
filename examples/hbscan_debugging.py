import time

import numpy as np
import seaborn as sns
import sklearn.datasets as data
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from hdbscan import HDBSCAN

plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}


def make_var_density_blobs(n_samples=750, centers=[[0,0]], cluster_std=[0.5], random_state=0):
    samples_per_blob = n_samples // len(centers)
    blobs = [make_blobs(n_samples=samples_per_blob, centers=[c], cluster_std=cluster_std[i])[0]
             for i, c in enumerate(centers)]
    labels = [i * np.ones(samples_per_blob) for i in range(len(centers))]
    return np.vstack(blobs), np.hstack(labels)

if __name__ == "__main__":
    ##############################################################################
    # Generate sample data
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # densities = [0.2, 0.35, 0.5]
    # X, labels_true = make_var_density_blobs(n_samples=750, centers=centers, cluster_std=densities,
    #                             random_state=0)

    # X = StandardScaler().fit_transform(X)

    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
    test_data = np.vstack([moons, blobs])
    plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
    plt.savefig('initial_data.png')
    plt.close()

    ##############################################################################
    hdb_t1 = time.time()
    hdb = HDBSCAN(min_cluster_size=10, max_cluster_eps=0.1).fit(test_data)
    hdb_labels = hdb.labels_
    hdb_elapsed_time = time.time() - hdb_t1


    ##############################################################################
    palette = sns.color_palette()
    cluster_colors = [sns.desaturate(palette[col], sat)
                    if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                    zip(hdb.labels_, hdb.probabilities_)]
    plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
    plt.savefig('cluster_labels.png')
    plt.close()

    print('BREAK POINT')
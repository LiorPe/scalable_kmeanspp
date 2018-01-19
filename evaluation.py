from sklearn.metrics import calinski_harabaz_score, silhouette_score, homogeneity_completeness_v_measure, \
    adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, mutual_info_score, \
    normalized_mutual_info_score

from KMeans import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import os

headers = ['n_clusters', 'max_iterations', 'seeds_algo', 't', 'l']


def plot(X, clusters, all_settings, output_path):
    X = np.asanyarray(X)
    Y = PCA(n_components=2).fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=clusters, cmap=plt.cm.Spectral, s=4)
    plt.title('Seeding algorithm: {}'.format(all_settings['seeds_algo']))
    file_name = '$'.join('{}={}'.format(k, v) for k, v in all_settings.items())
    plt.savefig(os.path.join(output_path, '{}.jpg'.format(file_name)))


def evaluate(X, labels=None,
             settings=[{'seeds_algo': 'kmeans++', 'n_clusters': 10, 'max_iterations': None, 't': 5, 'l': 10}]):
    output_path = 'evaluation_output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    X = [np.asarray(x) for x in X]
    # all_seeds_algo = []
    # all_seeds_algo.append('forgy')
    # all_seeds_algo.append('random')
    # all_seeds_algo.append('kmeans++')
    # all_seeds_algo.append('scalable-kmeans++')
    results_df = pd.DataFrame(columns=headers)
    for set in settings:
        result = {}
        print(set)
        kmeans = KMeans()
        kmeans.set_params(**set)
        clusters = kmeans.fit(X)
        plot(X, clusters, set, output_path)
        result['calinski_harabaz_score'] = calinski_harabaz_score(X, clusters)
        result['silhouette_score'] = silhouette_score(X, clusters)
        result['total_iterations'] = kmeans.total_itterations
        result['seeding_duration[s]'] = kmeans.seeds_init_duration

        if labels != None:
            result['homogeneity'], result['completeness'], result['v_measure'] = homogeneity_completeness_v_measure(
                labels, clusters)
            result['adjusted_mutual_info_score'] = adjusted_mutual_info_score(labels, clusters)
            result['adjusted_rand_score'] = adjusted_rand_score(labels, clusters)
            # result['fowlkes_mallows_score'] = fowlkes_mallows_score(labels,clusters)
            result['mutual_info_score'] = mutual_info_score(labels, clusters)
            result['normalized_mutual_info_score'] = normalized_mutual_info_score(labels, clusters)
        results_df = results_df.append(dict(**set, **result), ignore_index=True)
    results_df.to_csv(os.path.join(output_path, 'results.csv'))

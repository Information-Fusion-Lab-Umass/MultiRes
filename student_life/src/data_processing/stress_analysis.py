import sys, os
import pickle
import importlib
import numpy as np
from math import pi
import sklearn
from sklearn import cluster as CL
from sklearn.decomposition import PCA

import src.definitions as definitions
import pandas as pd
 

def load_csv(csv_path):
    return pd.load_csv(csv_path)


def cluster(estimator, data):
    reduced_data = PCA(n_components=2).fit_transform(data)
    print(estimator)
    print()

    estimator.fit(X)
    centroids = estimator.cluster_centers_
    labels = estimator.labels_
    
    return centroids, label

def stress_weight_mean(stress_dist, N=5):
    out = np.sum([stress_dist['stress_level_stress_{}'.format(i)]*(i+1) for i in range(N)], axis=0)
    total = np.sum([stress_dist['stress_level_stress_{}'.format(i)] for i in range(N)], axis=0)
    return out/total



if __name__ == '__main__':
    print("tst")
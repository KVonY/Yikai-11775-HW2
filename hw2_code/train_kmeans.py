#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MiniBatchKMeans
import cPickle
import sys
import pandas as pd

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(sys.argv)
        print "Usage: {0} surf_csv_file cluster_num output_file".format(sys.argv[0])
        print "surf_csv_file -- path to the surf csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    surf_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # selection = pd.read_csv('select.mfcc.csv', sep=';', dtype='float')
    # mfcc_vec = np.genfromtxt(mfcc_csv_file, delimiter=";")
    # surf_vec = pd.read_csv(surf_csv_file, sep=';', dtype='float')
    surf_vec = np.loadtxt(surf_csv_file, delimiter=";")

    # model = KMeans(cluster_num)
    model = MiniBatchKMeans(n_clusters=cluster_num)
    model.fit(surf_vec)
    save_file = open(output_file, 'wb')
    cPickle.dump(model, save_file)

    print "K-means trained successfully!"

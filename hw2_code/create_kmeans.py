#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import collections
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1) 

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model, 'rb'))
    files = open(file_list)
    output_path = "kmeans/feature.vec"
    output = open(output_path, 'w')
    for f in files:
        video_name = f.strip()
        surf_path = "surf/{}.surf.npy".format(video_name)
        if os.path.exists(surf_path) != True:
            print "No SURF features"
            z = np.zeros(cluster_num)
            z.fill(1.0/cluster_num)
            zeros = map(str, z)
            zeros_feature = ';'.join(zeros)
            print zeros_feature
            output.write(video_name + ' ' + zeros_feature + '\n')
            continue
        arr = np.load(surf_path)
        idx = 0
        while idx < len(arr):
            if arr[idx] is None:
                idx += 1
                continue
            surf_vec = arr[idx]
            break
        for i in range(idx + 1, len(arr)):
            if arr[i] is None:
                continue
            surf_vec = np.vstack((surf_vec, arr[i]))
        labels = kmeans.predict(surf_vec) # vector of labels
        # bag-of-word vector representation
        label_count = collections.Counter(labels) # dictionary
        bow = np.zeros(cluster_num)
        for i in range(cluster_num):
            bow[i] = label_count[i]
        # normalize
        if np.sum(bow) == 0:
            bow_vec = np.zeros(cluster_num)
            bow_vec.fill(1.0/cluster_num)
            print "WRONG bag-of-word vector representation\n"
        else:
            bow_vec = bow/float(np.sum(bow))
        # output
        feature = ';'.join([str(i) for i in bow_vec])
        output.write(video_name + ' ' + feature + '\n')
        print "{} done.".format(video_name)
    output.close()
    files.close()

    print "K-means features generated successfully!"

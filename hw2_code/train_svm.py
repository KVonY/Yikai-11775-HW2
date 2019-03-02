#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import chi2_kernel

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        print "USE -- for map/kaggle use"
        exit(1)
    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    USE = sys.argv[5]  # map/kaggle
    output = open(output_file, 'wb')
    feature_path = feat_dir + "/feature.vec"


    all_video = []
    if USE == 'map':
        y = []
        # y: labels
        label_file = open('/home/ubuntu/11775-hws/all_trn.lst', 'r')
        for i in label_file:
            video_name, label = i.strip().split(' ')
            all_video.append(video_name)
            if label == event_name:
                y.append(1)
            else:
                y.append(0)
        # x: features
        x = list(np.zeros(len(y)))
        feature_file = open(feature_path, 'r')
        for j in feature_file:
            video_name, feature_vec = j.strip().split(' ')
            try:
                idx = all_video.index(video_name)
                feature = feature_vec.split(';')
                feature = map(float, feature)
                x[idx] = feature
            except ValueError:
                continue
    elif USE == 'kaggle':
        y = []
        # y: labels
        label_file_train = open('/home/ubuntu/11775-hws/all_trn.lst', 'r')
        for i in label_file_train:
            video_name, label = i.strip().split(' ')
            all_video.append(video_name)
            if label == event_name:
                y.append(1)
            else:
                y.append(0)
        lable_file_val = open('/home/ubuntu/11775-hws/all_val.lst', 'r')
        for i in lable_file_val:
            video_name, label = i.strip().split(' ')
            all_video.append(video_name)
            if label == event_name:
                y.append(1)
            else:
                y.append(0)
        # x: features
        x = list(np.zeros(len(y)))
        feature_file = open(feature_path, 'r')
        for i in feature_file:
            video_name, feature_vec = i.strip().split(' ')
            try:
                idx = all_video.index(video_name)
                feature = feature_vec.split(';')
                feature = map(float, feature)
                x[idx] = feature
            except ValueError:
                continue
    # linear, poly, rbf, cosine_similarity, laplacian_kernel, chi2_kernel
    model = SVC(kernel=chi2_kernel, gamma='scale')
    model.fit(x, y)
    cPickle.dump(model, output)

    print 'SVM trained successfully for event %s!' % (event_name)

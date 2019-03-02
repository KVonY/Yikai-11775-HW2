#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import chi2_kernel

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        print "USE -- for map/kaggle use"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    USE = sys.argv[5]  # map/kaggle
    feature_path = feat_dir + '/feature.vec'

    model = cPickle.load(open(model_file, 'rb'))
    all_video = []
    if USE == 'kaggle':
        video_file = open("list/test.video", 'r')
    elif USE == 'map':
        video_file = open("list/val.video", 'r')

    for i in video_file:
        all_video.append(i.strip())
    # print(all_video)
    # x: features
    x = list(np.zeros(len(all_video)))
    feature_file = open(feature_path, 'r')
    for i in feature_file:
        video_name, feature_vec = i.strip().split(' ')
        # print(video_name)
        # print(video_name in all_video)
        try:
            idx = all_video.index(video_name)
            feature = feature_vec.split(';')
            # print(feature)
            feature = map(float, feature)
            x[idx] = feature
        except ValueError:
            # print("ValueError")
            continue
    # predict
    # print(x)
    prediction = open(output_file, 'w')
    for f in x:
        f_reshape = np.array(f).reshape(1, -1)
        # print(f_reshape)
        pred = model.decision_function(f_reshape)
        print(pred)
        prediction.write(str(pred[0]) + '\n')
    prediction.close()

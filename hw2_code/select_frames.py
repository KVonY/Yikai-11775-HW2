#!/bin/python
# Randomly select 

import numpy as np
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} file_list select_ratio output_file".format(sys.argv[0])
        print "file_list -- the list of video names"
        print "select_ratio -- the ratio of frames to be randomly selected from each audio file"
        print "output_file -- path to save the selected frames (feature vectors)"
        exit(1)

    file_list = sys.argv[1]; output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list,"r")
    # fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    np.random.seed(18877)
    count = 0
    for line in fread.readlines():
        # mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        surf_path = "surf/" + line.replace('\n','') + ".surf.npy"
        print "doing " + surf_path
        if os.path.exists(surf_path) == False:
            continue
        # array = numpy.genfromtxt(surf_path, delimiter=";")
        arr = np.load(surf_path)
        idx = 0
        while idx < len(arr):
            if arr[idx] is None:
                idx += 1
                continue
            array = arr[idx]
            break
        for i in range(idx + 1, len(arr)):
            if arr[i] is None:
                continue
            array = np.vstack((array, arr[i]))
        np.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]
        result_arr = []
        for n in xrange(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')
        count += 1
        np.save(output_file)
        print "done " + str(count)
    # fwrite.close()


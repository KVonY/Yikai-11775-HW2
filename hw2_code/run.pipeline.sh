#!/bin/bash

# pip install pyyaml


# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -k true -y filepath

# Reading of all arguments:
while getopts p:f:m:k:y: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=/home/ubuntu/11775-hws-old/hw1_code/11775_videos/video  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../hw1_code/list/val > list/val.video
    awk '{print $1}' ../hw1_code/list/test.video > list/test.video
    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/all.video"); do
        ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    python surf_feat_extraction.py -i list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
	python cnn_feat_extraction.py list/all.video config.yaml


fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    python select_frames.py list/train.video 0.2 select.surf.csv || exit 1;
    cluster_num=500
    python train_kmeans.py select.surf.csv $cluster_num kmeans.${cluster_num}.model || exit 1;


    # 2. TODO: Create kmeans representation for SURF features
    python create_kmeans.py kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;

	echo "#####################################"
    echo "#   CNN FEATURE REPRESENTATION      #"
    echo "#####################################"

	# 1. TODO: Train kmeans to obtain clusters for CNN features
    python create_cnn.py list/all.video || exit 1;

    # 2. TODO: Create kmeans representation for CNN features

fi

if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    mkdir -p surf_pred
    # iterate over the events
    feat_dim_surf=500
    use=map
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      # now train a svm model
      python train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.model $use;
      # apply the svm model to *ALL* the testing videos;
      # output the score of each testing video to a file ${event}_pred
      python test_svm.py surf_pred/svm.$event.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst $use;
      # compute the average precision by calling the mAP package
      ap list/${event}_val_label surf_pred/${event}_surf.lst
    done

#    feat_dim_surf=500
#    # 1. TODO: Train SVM with OVR using only videos in training set.
#    python train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.model map || exit 1;
#    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
#    python test_svm.py surf_pred/svm.$event.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst map || exit 1;
#	# 3. TODO: Train SVM with OVR using videos in training and validation set.
#    python train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.model kaggle || exit 1;
#	# 4. TODO: Test SVM with test set saving scores for submission
#    python test_svm.py surf_pred/svm.$event.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst kaggle || exit 1;

    echo "#######################################"
    echo "# MED with CNN Features: MAP results  #"
    echo "#######################################"

    mkdir -p cnn_pred
    # iterate over the events
    feat_dim_cnn=2048
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      # now train a svm model
      python train_svm.py $event "cnnfeat/" $feat_dim_cnn cnn_pred/svm.$event.model $use;
      # apply the svm model to *ALL* the testing videos;
      # output the score of each testing video to a file ${event}_pred
      python test_svm.py cnn_pred/svm.$event.model "cnnfeat/" $feat_dim_cnn cnn_pred/${event}_cnn.lst $use;
      # compute the average precision by calling the mAP package
      ap list/${event}_val_label cnn_pred/${event}_cnn.lst
    done
    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

fi


if [ "$KAGGLE" = true ] ; then

    echo "##########################################"
    echo "# MED with SURF Features: KAGGLE results #"
    echo "##########################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    mkdir -p surf_pred
    # iterate over the events
    feat_dim_surf=500
    use=kaggle
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      # now train a svm model
      python train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.model $use;
      # apply the svm model to *ALL* the testing videos;
      # output the score of each testing video to a file ${event}_pred
      python test_svm.py surf_pred/svm.$event.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst $use;
      # compute the average precision by calling the mAP package
      ap list/${event}_val_label surf_pred/${event}_surf.lst
    done


    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

    # 4. TODO: Test SVM with test set saving scores for submission


    echo "##########################################"
    echo "# MED with CNN Features: KAGGLE results  #"
    echo "##########################################"

    mkdir -p cnn_pred
    # iterate over the events
    feat_dim_cnn=2048
    for event in P001 P002 P003; do
      echo "=========  Event $event  ========="
      # now train a svm model
      python train_svm.py $event "cnnfeat/" $feat_dim_cnn cnn_pred/svm.$event.model $use;
      # apply the svm model to *ALL* the testing videos;
      # output the score of each testing video to a file ${event}_pred
      python test_svm.py cnn_pred/svm.$event.model "cnnfeat/" $feat_dim_cnn cnn_pred/${event}_cnn.lst $use;
      # compute the average precision by calling the mAP package
      ap list/${event}_val_label cnn_pred/${event}_cnn.lst
    done
    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission
	python kaggle.py

fi

#!/usr/bin/env bash
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

python kaggle.py
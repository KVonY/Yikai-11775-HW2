echo "#######################################"
echo "# MED with SURF Features: MAP results #"
echo "#######################################"

# Paths to different tools;
map_path=/home/ubuntu/tools/mAP
export PATH=$map_path:$PATH

mkdir -p surf_pred
# iterate over the events
feat_dim_surf=500
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.model map;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python test_svm.py surf_pred/svm.$event.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst map;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label surf_pred/${event}_surf.lst
done
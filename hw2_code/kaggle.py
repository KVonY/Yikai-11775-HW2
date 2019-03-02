import numpy as np

P001_asr_file = open("asr_pred/P001_asr.lst")
P002_asr_file = open("asr_pred/P002_asr.lst")
P003_asr_file = open("asr_pred/P003_asr.lst")
P001_mfcc_file = open("mfcc_pred/P001_mfcc.lst")
P002_mfcc_file = open("mfcc_pred/P002_mfcc.lst")
P003_mfcc_file = open("mfcc_pred/P003_mfcc.lst")
test_video_file = open("list/test.video")
# test video
test_video = []
for v in test_video_file:
    tmp = v.strip()
    test_video.append(tmp)
# mfcc
mfcc_label = []
mfcc = []
P001_mfcc = []
P002_mfcc = []
P003_mfcc = []
output_mfcc_file = open("mfcc_kaggle_prediction.csv", 'w')
for a in P001_mfcc_file:
    tmp = float(a.strip())
    P001_mfcc.append(tmp)
for b in P002_mfcc_file:
    tmp = float(b.strip())
    P002_mfcc.append(tmp)
for c in P003_mfcc_file:
    tmp = float(c.strip())
    P003_mfcc.append(tmp)
mfcc = np.array([P001_mfcc, P002_mfcc, P003_mfcc])
mfcc = mfcc.T
# print mfcc
output_mfcc_file.write("VideoID,Label\n")
for i in range(len(mfcc)):
    tmp_video = test_video[i]
    tmp = tmp_video + ',' + str(np.argmax(mfcc[i]) + 1) + '\n'
    mfcc_label.append(tmp)
    output_mfcc_file.write(tmp)
output_mfcc_file.close()
# asr
asr_label = []
asr = []
P001_asr = []
P002_asr = []
P003_asr = []
output_asr_file = open("asr_kaggle_prediction.csv", 'w')
for a in P001_asr_file:
    tmp = float(a.strip())
    P001_asr.append(tmp)
for b in P002_asr_file:
    tmp = float(b.strip())
    P002_asr.append(tmp)
for c in P003_asr_file:
    tmp = float(c.strip())
    P003_asr.append(tmp)
asr = np.array([P001_asr, P002_asr, P003_asr])
asr = asr.T
# print asr
output_asr_file.write("VideoID,Label\n")
for i in range(len(asr)):
    tmp_video = test_video[i]
    tmp = tmp_video + ',' + str(np.argmax(asr[i]) + 1) + '\n'
    asr_label.append(tmp)
    output_asr_file.write(tmp)
output_asr_file.close()
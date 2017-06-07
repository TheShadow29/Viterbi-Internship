import os
# import shutil
import subprocess
tdir = '/home/agharwal/kitchen/clips/feats/m_fc7/'
dest_path = '/media/arka_s/fc23520b-5ce4-4c6e-be90-09d667b2a245/kitchen/clips/feats/m_fc7/'
# dir_list = os.listdir(tdir)

# subprocess.call(["ls", "-l"])
for i in range(13,14):
    # subprocess.call(["gcp","-r","../cnn_caffe_places365/places365_temp.py","."])
    src = tdir + 's' + str(i) + '*'
    # src = tdir + 's13-d08-cam-002_1_14'
    subprocess.call(["gcp", "-r", src, dest_path])

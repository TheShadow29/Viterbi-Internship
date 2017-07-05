from subprocess import call
import os


# img_tdir = '../../data/protest_data/only_text/beach_mdf_019/'
img_tdir = '../../data/protest_data/only_text/protest_img_017/'
img_list = []
for img_f in os.listdir(img_tdir):
    if img_f[-4:] == '.png' or img_f[-4:] == '.jpg':
        img_list.append(img_f)

app_file = '/home/arka_s/internship_files/image_segmentation/opencv-wrapper-egbis/build/main1'

seg_dir = img_tdir + 'seg/'

if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)

for img_n in img_list:
    call([app_file, img_tdir + img_n, seg_dir + img_n])

# call(["/home/arka_s/internship_files/image_segmentation/opencv-wrapper-egbis/build/main1", "../../data/protest_data/only_text/beach_mdf_019/0.png", "./a2.png"])

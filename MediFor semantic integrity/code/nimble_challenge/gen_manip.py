import skimage.io as skio
import skimage.transform as skt
import skimage.util as skutil
import matplotlib.pyplot as plt
import os
# import shutil
import random
import numpy as np
import sys

if sys.argv[1] == '1':
    base_tdir = '/arka_data/NC2017_Dev1_Beta4/world/'
    out_tdir = '/mnt/disk1/ark_data/code_manip/dev1/seed_2/'
elif sys.argv[1] == '3':
    base_tdir = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/'
    out_tdir = '/mnt/disk1/ark_data/code_manip/dev3/seed_2/'

all_imgs_file_list = os.listdir(base_tdir)
all_imgs = []

for f in all_imgs_file_list:
    if f[-4:] == '.jpg' or f[-4:] == '.png':
        all_imgs.append(f)

total_imgs = len(all_imgs)
random.seed(2)
for itern in range(1000):
    try:
        ind_base = int(random.random() * total_imgs)
        img_base = skio.imread(base_tdir + all_imgs[ind_base])
        # base_alpha = False
        if img_base.shape == (2,):
            img_base = img_base[0]
            # if img_base.shape[2] == 4:
            # img_base = img_base[:, :, :3]
            # base_alpha = True
        while True:
            ind_probe = int(random.random() * total_imgs)
            if (ind_base != ind_probe):
                break
        img_probe = skio.imread(base_tdir + all_imgs[ind_probe])
        if img_probe.shape == (2,):
            img_probe = img_probe[0]
        # probe_alpha = False
        # if img_probe.shape[2] == 4:
        # img_probe = img_probe[:, :, :3]
        # probe_alpha = True
        base_h, base_w, base_ch = img_base.shape
        # mask_width =
        # img_probe2 = skimage.transform.resize(img_probe, (base_w/2, base_h/2), mode='reflect')
        img_probe2 = skt.resize(img_probe, (base_h/2, base_w/2))
        angle = random.random() * 360
        img_probe2 = skt.rotate(img_probe2, angle, resize=True)
        # img_probe2 = skt.resize(img_probe, (base_h/2, base_w/2))
        img_probe2 = skutil.img_as_ubyte(img_probe2)
        # new_img = np.zeros(img_base.shape)
        new_img = np.copy(img_base)
        probe_h, probe_w, probe_ch = img_probe2.shape
        iter_chooser = 0
        while True:
            iter_chooser += 1
            if iter_chooser > 100:
                print ('Too many iterations ', iter_chooser)
                x1 = 0
                y1 = 0
                break
            x1 = int(random.random() * base_w)
            y1 = int(random.random() * base_h)
            if x1 + probe_w < base_w and y1 + probe_h < base_h:
                break
        # new_img[y1:y1 + probe_h, x1:x1 + probe_w] -= img_base[y1:y1 + probe_h, x1:x1 + probe_w]

        for xi in range(x1, x1 + probe_w):
            for yi in range(y1, y1 + probe_h):
                if not np.array_equal(img_probe2[yi - y1][xi - x1], np.zeros(3,)):
                    # new_img[yi][xi] -= img_base[yi][xi]
                    new_img[yi][xi][:3] = img_probe2[yi - y1][xi - x1][:3]
                    # print (img_probe2[yi - y1][xi - x1])
                    # print ('lol')
        fig = plt.figure()
        plt.imshow(new_img)
        img_fname = out_tdir + 'base_' + str(all_imgs[ind_base][:-4]) + \
                               '_probe_' + str(all_imgs[ind_probe][:-4]) + '.png'
        # plt.savefig(img_fname, bbox_inches='tight', pad_inches=0)
        skio.imsave(img_fname, new_img)
        plt.close()
        print ('Itern ', itern, 'done')
    except Exception as e:
        print ('Itern ', itern, 'not completed')
        continue
    # plt.show()
    # x1 = random.random()*base_w
    # skio.imshow(new_img)
    # skio.imshow(img_probe2)
    # skio.imshow(new_img)
    # skio.show()

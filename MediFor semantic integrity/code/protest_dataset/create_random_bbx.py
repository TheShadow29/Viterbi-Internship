# import random
import os
import numpy as np
import skimage.io as skio
import pickle


if __name__ == '__main__':
    # img_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    # img_tdir = '/mnt/disk1/ark_data/code_manip/dev3/seed_1/'
    # img_tdir = '/mnt/disk1/ark_data/code_manip/protest/'
    img_tdir = '/arka_data/places_data/copied_data/'
    dict_random_bbx = dict()
    img_list = list()
    for fname in os.listdir(img_tdir):
        if fname[-4:] == '.png' or fname[-4:] == '.jpg':
            img_list.append(fname)

    for itern, img_fname in enumerate(img_list):
        img = skio.imread(img_tdir + img_fname)
        r1 = img.shape[0]
        c1 = img.shape[1]
        bbx_list = list()
        # counter = 0
        while True:
            bb_x1 = np.random.randint(c1)
            bb_y1 = np.random.randint(r1)
            bb_x2 = np.random.randint(c1)
            bb_y2 = np.random.randint(r1)
            bb_x1, bb_x2 = min(bb_x1, bb_x2), max(bb_x1, bb_x2)
            bb_y1, bb_y2 = min(bb_y1, bb_y2), max(bb_y1, bb_y2)
            if bb_x2 < c1 and bb_y2 < r1:
                bbx_list.append([bb_x1, bb_y1, bb_x2, bb_y2])
            if len(bbx_list) == 100:
                break
        dict_random_bbx[img_fname] = np.array(bbx_list)
        print ('Itern ', itern)
    pickle.dump(dict_random_bbx, open('./random_bbx.pkl', 'w'))

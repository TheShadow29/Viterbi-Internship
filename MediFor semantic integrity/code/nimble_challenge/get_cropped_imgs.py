import os
import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':
    probe_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    modf_npy_file = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/mod_fine_labels.npy'
    bbox_dict_npy = np.load(modf_npy_file)
    bbox_dict = bbox_dict_npy.item()
    mdf_imgs = []
    dest_dir = '../../data/protest_data/cropped/'
    for sdir in os.listdir(probe_tdir):
        if sdir[-4:] == '.jpg' or sdir[-4:] == '.png':
            mdf_imgs.append(sdir)

    for f in mdf_imgs:
        img = io.imread(probe_tdir + f)
        pts = bbox_dict[f[:-4]]
        img1 = img[pts[1]:pts[3], pts[0]:pts[2]]
        # fig, ax = plt.subplots(1)
        # io.imsave(dest_dir + 'direct_cropped/' + f[:-4] + '_cropped' + f[-4:], img1)
        # img1_eq = skimage.exposure.equalize_hist(img1)
        img1_eq = np.zeros(img1.shape)
        if len(img1.shape) == 3:
            for i in range(3):
                img1_eq[:, :, i] = skimage.exposure.equalize_hist(img1[:, :, i])
        else:
            img1_eq = skimage.exposure.equalize_hist(img1)
            # img1_eq_scaled = img1_eq * 256
        # img1_eq_scaled_uint = img1_eq_scaled.astype('uint8')
        img1_eq_scaled = skimage.exposure.rescale_intensity(img1_eq, out_range='uint8')
        img1_eq_scaled_uint = img1_eq_scaled.astype('uint8')
        io.imsave(dest_dir + 'hist_equalized/' + f[:-4] + '_cropped_heq' + f[-4:], img1_eq_scaled_uint)
        # ax.imshow(img1_eq)
        # plt.hist(img1_eq_scaled.flatten(), 100)
        # rect = mpatches.Rectangle((pts[0], pts[1]), pts[2] - pts[0],
        #                           pts[3] - pts[1], edgecolor='red', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

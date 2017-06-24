import sys
sys.path.append('../nimble_challenge/')
import two_imgs_eff
import os
import pdb

if __name__ == '__main__':
    pr_img_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    ac_img_tdir = '../../data/protest_data/cropped/direct_cropped/'

    pr_img_list = list()
    for fname in os.listdir(pr_img_tdir):
        if fname[-4:] == '.jpg' or fname[-4:] == '.png':
            pr_img_list.append(fname)

    for pr_img in pr_img_list:
        pr_num = pr_img.split('_')[-1]
        ac_img = 'protest_img_' + pr_num
        a1 = two_imgs_eff.are_the_two_imgs_same(pr_img_tdir + pr_img, ac_img_tdir + ac_img)
        pdb.set_trace()

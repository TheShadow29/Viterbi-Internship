# import caffe
# import matplotlib.pyplot as plt
import pickle
import two_imgs_eff
import numpy as np
import os
# import sys
import shutil
# from get_all_feature_vecs17 import info_storer, info_storer_all
# from parse_all_data_nimble17 import nimble_references, nimble_prov_reference, prov_ref_file
from get_all_feature_vecs_protest import info_storer
import pdb
import disp_img


if __name__ == '__main__':
    # probe_file = open('../../data/protest_data/Modified_Images_ProtestL.pkl', 'rb')
    # probe_file = open('../../data/protest_data/Modified_Images_ProtestL_bbox.pkl', 'rb')
    probe_file = open('../../data/protest_data/Modified_Images_ProtestL_hist_eq.pkl')
    probe_all_info = pickle.load(probe_file)
    # world_file = open('../../data/protest_data/Pruned_Protest_YFCCImages.pkl', 'rb')
    world_file = open('../../data/protest_data/Pruned_Protest_YFCCImages_hist_eq.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    world_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Pruned_Protest_YFCCImages/'
    probe_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    num_corr = 0
    total_num = 0
    k = 5
    for itern, p_dat in enumerate(probe_all_info.data[:3]):
        # pdb.set_trace()
        print ("iter no. ", itern)
        fv_probe = p_dat.fv3
        pfid = p_dat.fid
        gt_label = int(pfid.split('.')[0].split('_')[-1])
        all_corr = np.array([])
        for w in world_all_info.data:
            fv_world = w.fv3
            corr = two_imgs_eff.cmp_fv(fv_probe, fv_world, 'ncc')['pear_ncc']
            all_corr = np.append(all_corr, corr)
        top_ids = all_corr.argsort()[::-1]
        top_k_ids = top_ids[:k]
        top_corr = all_corr[top_k_ids]
        print(top_corr)
        
        l1 = []
        guess = False
        guess_id = 10
        l1.append(probe_tdir + pfid)
        print ('GT label is ', gt_label)
        for it, id1 in enumerate(top_k_ids):
            w1 = world_all_info.data[id1].fid
            l1.append(world_tdir + w1)
            w_label = int(w1.split('.')[0].split('_')[-1])
            print ('W label is ', w_label)
            if w_label == gt_label:
                guess = True
                guess_id = it
        if guess:
            num_corr += 1
            print ('Correct guess in top ' + it)
        else:
            print ('No correct result in top 5')
        disp_img.show_img_protest(l1, top_corr)
        total_num += 1

    print (num_corr, total_num)

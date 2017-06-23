# import caffe
# import matplotlib.pyplot as plt
import sys
sys.path.append('../nimble_challenge/')
sys.path.append('/home/arka_s/Caffe/caffe/python/')
import pickle
import two_imgs_eff
import numpy as np
import os
# import sys
import shutil
import prov_ref_node_creator as prnc
# from get_all_feature_vecs17 import info_storer, info_storer_all
# from get_all_fv_code_gen import info_s-torer, info_storer_all
from txt_data_storer import info_storer, info_storer_all
from parse_all_data_nimble17 import nimble_references, nimble_prov_reference, prov_ref_file
import pdb
import disp_img
import imshow_coll
from txt_data_storer import info_storer, info_storer_all



if __name__ == '__main__':
    # probe_file = open('../../data/nimble17_data/alexnet365_dev3_seed_2.pkl', 'rb')
    probe_file = open('../../data/protest_data/alexnet365_Modified_Images_ProtestL_\
txt_boxes.pkl', 'rb')
    # probe_file = open('../../data/protest_data/bvlc_alexnet_Modified_Images_ProtestL_\
# txt_boxes.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)
    # world_file = open('../../data/nimble17_data/alexnet365_NC2017_Dev3_Beta1_world.pkl')
    world_file = open('../../data/protest_data/alexnet365_Pruned_Protest_\
YFCCImages_txt_boxes.pkl', 'rb')
    # world_file = open('../../data/protest_data/bvlc_alexnet_Pruned_Protest_\
# YFCCImages_txt_boxes.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    # test_dir = '/mnt/disk1/ark_data/code_manip/dev3/seed_2/'
    test_dir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    test_imgs = []
    for f in os.listdir(test_dir):
        if f[-4:] == '.jpg' or f[-4:] == '.png':
            test_imgs.append(f)

    # wtdir = '/arka_data/NC2017_Dev1_Beta4/world/'
    # wtdir = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/'
    wtdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Pruned_Protest_YFCCImages/'

    probe_dict = dict()
    world_dict = dict()
    for ind, pdat in enumerate(probe_all_info.data):
        probe_dict[pdat.fid] = ind
    for ind, wdat in enumerate(world_all_info.data):
        world_dict[wdat.fid] = ind

    num_corr = 0
    total_num = 0
    goods = 0
    bad_nums = []

    for itern, pfid in enumerate(test_imgs[:]):
        # pdb.set_trace()
        # base_gt_num = pfid.split.split('_')[-1]
        try:
            base_gt_num = pfid.split('_')[-1]
            base_gt_file = 'protest_img_' + base_gt_num
            p_info = probe_all_info.data[probe_dict[pfid]]
            corr_arr = list()
            m_cor_all_w = 0
            wid_match = ''
            for w_info in world_all_info.data:
                for fvp in p_info.fv_list:
                    for fvw in w_info.fv_list:
                        corr = two_imgs_eff.cmp_fv(fvp, fvw, 'ncc')['pear_ncc']
                        if corr > m_cor_all_w:
                            m_r_all_w = corr
                            wid_match = w_info.fid
                            # pdb.set_trace()
            if wid_match == base_gt_file:
                num_corr += 1
            total_num += 1
        except KeyError as e:
            pass

    print (num_corr, total_num)
    print bad_nums
    print goods

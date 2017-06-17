# import caffe
# import matplotlib.pyplot as plt
import pickle
import two_imgs_eff
import numpy as np
import os
# import sys
import shutil
import prov_ref_node_creator as prnc
# from get_all_feature_vecs17 import info_storer, info_storer_all
from get_all_fv_code_gen import info_storer, info_storer_all
from parse_all_data_nimble17 import nimble_references, nimble_prov_reference, prov_ref_file
import pdb
import disp_img
import imshow_coll


if __name__ == '__main__':
    # Start with dev1 and then onto dev3
    # Start by testing on the seed 1 and then go to seed 2(may take some time for generation)
    # probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/bvlc_alexnet_probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/dev3/alexnet365_probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/alexnet365_dev1_seed_1.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/alexnet365_dev3_seed_1.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/bvlc_alexnet_dev1_seed_1.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/bvlc_alexnet_dev3_seed_1.pkl', 'rb')
    probe_file = open('../../data/nimble17_data/alexnet365_dev3_seed_2.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)
    # world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/bvlc_alexnet_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/alexnet365_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/alexnet365_bigger_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_bigger_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/alexnet365_NC2017_Dev1_Beta4_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/bvlc_alexnet_NC2017_Dev1_Beta4_world.pkl')
    world_file = open('../../data/nimble17_data/alexnet365_NC2017_Dev3_Beta1_world.pkl')
    # world_file = open('../../data/nimble17_data/bvlc_alexnet_NC2017_Dev3_Beta1_world.pkl', 'rb')
    # world_file = open('..')
    world_all_info = pickle.load(world_file)

    num_corr = 0
    total_num = 0
    goods = 0
    bad_nums = []

    # test_dir = '/mnt/disk1/ark_data/code_manip/dev1/seed_1/'
    # test_dir = '/mnt/disk1/ark_data/code_manip/dev3/seed_1/'
    test_dir = '/mnt/disk1/ark_data/code_manip/dev3/seed_2/'
    test_imgs = []
    for f in os.listdir(test_dir):
        if f[-4:] == '.jpg' or f[-4:] == '.png':
            test_imgs.append(f)
    # for itern, p_node in enumerate(prov_data.nodes[:]):
    # wtdir = '/arka_data/NC2017_Dev1_Beta4/world/'
    wtdir = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/'
    for itern, pfid in enumerate(test_imgs):
        # pdb.set_trace()
        print ("iter no. ", itern)
        # p_info_idx = dict_probe[pfid]
        probe_info = probe_all_info.data[probe_all_info.data_id_dict[pfid]]
        fv_probe = probe_info.fv3
        temp_array = np.array([])
        id_array = np.array([])
        # k = len(p_node.wfids)
        k = 10
        for idx, w in enumerate(world_all_info.data):
            if pfid not in w.fid:
                # pdb.set_trace()
                # lol += 1
                corr = two_imgs_eff.cmp_fv(fv_probe, w.fv3, 'ncc')['pear_ncc']
                temp_array = np.append(temp_array, corr)
                id_array = np.append(id_array, idx)
            # else:
            # lol += 1
        top_k = temp_array.argsort()[-k:]
        top_k = list(reversed(top_k))
        print (temp_array[top_k])
        ids = id_array[top_k].astype(int)
        # pdb.set_trace()
        base_gt = pfid.split('.')[0].split('_')[1]
        try:
            wbase_gt_ind = world_all_info.data_id_dict[base_gt + '.png']
        except KeyError as e:
            wbase_gt_ind = world_all_info.data_id_dict[base_gt + '.jpg']
        wbase_gt = world_all_info.data[wbase_gt_ind]
        corr_gt = two_imgs_eff.cmp_fv(fv_probe, wbase_gt.fv3, 'ncc')['pear_ncc']
        print ('Ground truth base corr ', corr_gt)
        for i in range(k):
            # if p_dat.base_browser_file_name in world_all_info.data[top_k[i]].fid:
            guess = False
            # if world_all_info.data[ids[i]].fid[:-4] in p_node.wfids:
            # if world_all_info.data[ids[i]].fid.split('.')[0].split('/')[-1] in p_node.wfids:
            wi = world_all_info.data[ids[i]]
            bool1 = wi.fid.split('.')[0].split('/')[-1] == base_gt
            corr1 = two_imgs_eff.cmp_fv(wi.fv3, wbase_gt.fv3, 'ncc')['pear_ncc']
            bool2 = corr1 > 0.95
            if bool1 or bool2:
            # if bool1:
                num_corr += 1
                guess = True
                if (guess):
                    if (temp_array[top_k[i]] > 0.95):
                        goods += 1
                # print
                break
            # if (guess):
        # print
        print ('Itern is ' + str(guess))
        # guess = False
        if not guess:
        # if True:

            bad_nums.append(itern)
            print 'Pfid is ' + str(pfid)
            tmp_w_array = [world_all_info.data[ids[i]] for i in range(k)]
            print 'Wfids predicted are '
            # wfids_to_print = ['world/' + str(w.fid) for w in tmp_w_array]
            # pdb.set_trace()
            wfids_to_print = [str(w.fid) for w in tmp_w_array]
            # print [str(w.fid) for w in tmp_w_array]
            print wfids_to_print
            # l1 = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/probe/' + str(pfid)
            l1 = []
            l1.append(test_dir + str(pfid))
            # l1 = 'world/' + str(p_node.fid)
            # pdb.set_trace()
            # l1 = [l1] + [wtdir + base_gt]
            l1.append(wtdir + base_gt)
            wfids_to_print2 = [wtdir + w.fid for w in tmp_w_array]
            l1 = l1 + wfids_to_print2
            # l1.append(wfids_to_print)
            # disp_img.show_img(l1)
            # imshow_coll.imshow_collection_new(l1)
            # pdb.set_trace()

        total_num += 1

    print (num_corr, total_num)
    print bad_nums
    print goods

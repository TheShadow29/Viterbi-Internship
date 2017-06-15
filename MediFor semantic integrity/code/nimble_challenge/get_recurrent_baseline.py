import pickle
import two_imgs_eff
import numpy as np
import os
# import sys
import shutil
import prov_ref_node_creator as prnc
from get_all_feature_vecs17 import info_storer, info_storer_all
from parse_all_data_nimble17 import nimble_references, nimble_prov_reference, prov_ref_file
import pdb
import disp_img
import imshow_coll

if __name__ == '__main__':
    # probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/bvlc_alexnet_probe.pkl', 'rb')
    probe_file = open('../../data/nimble17_data/dev3/alexnet365_probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_probe.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)
    # world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/bvlc_alexnet_world.pkl', 'rb')
    world_file = open('../../data/nimble17_data/dev3/alexnet365_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/alexnet365_bigger_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_bigger_world.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    pref_file = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
                'reference/provenance/NC2017_Dev3-provenance-ref-node.csv'
    prov_data = prnc.prov_nodes(pref_file)
    prov_data.populate_data()

    dict_probe = dict()
    dict_world = dict()
    for i, p in enumerate(probe_all_info.data):
        dict_probe[p.fid[:-4]] = i
    for i, w in enumerate(world_all_info.data):
        dict_world[w.fid[:-4]] = i

    num_corr = 0
    total_num = 0
    goods = 0
    bad_nums = []
    wtdir = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/'

    for itern, p_node in enumerate(prov_data.nodes[:]):
        # pdb.set_trace()
        # print ("iter no. ", itern)
        pfid = p_node.fid
        p_info_idx = dict_probe[pfid]
        probe_info = probe_all_info.data[p_info_idx]
        fv_probe = probe_info.fv3


        # k = len(p_node.wfids)
        k = 1
        curr_wfids = [pfid]
        n1 = len(p_node.wfids)
        plus_one = False
        if pfid not in p_node.wfids:
            plus_one = True
            n1 += 1
        # pdb.set_trace()
        # while len(curr_wfids) < len(p_node.wfids):
        while len(curr_wfids) < n1:
            temp_array = np.array([])
            id_array = np.array([])
            # imshow_coll.imshow_collection_new([wtdir + w123 for w123 in curr_wfids])
            # pdb.set_trace()
            for idx, w in enumerate(world_all_info.data):
                # if pfid not in w.fid:
                max_cor = 0
                if w.fid[:-4] not in curr_wfids:
                    for ws in curr_wfids[:1]:
                        wid = dict_world[ws]
                        w_1 = world_all_info.data[wid]
                        fv123 = w_1.fv3
                        corr_i = two_imgs_eff.cmp_fv(fv123, w.fv3, 'ncc')['pear_ncc']
                        if corr_i > max_cor:
                            max_cor = corr_i
                    temp_array = np.append(temp_array, max_cor)
                    id_array = np.append(id_array, idx)
            top_k = temp_array.argsort()[-k:]
            top_k = list(reversed(top_k))
            # print (temp_array[top_k])
            ids = id_array[top_k].astype(int)
            for i in range(k):
                # print (world_all_info.data[ids[i]].fid.split('.')[0].split('/')[-1])
                curr_wfids.append(world_all_info.data[ids[i]].fid.split('.')[0].split('/')[-1])
                
        # print (p_node.wfids)
        # print(curr_wfids)
        # imshow_coll.imshow_collection_new([wtdir + w123 for w123 in curr_wfids])
        # if (len(p_node.wfids) == len(set(p_node.wfids) & set(curr_wfids[1:]))):
        guess = False
        if not plus_one:
            if set(p_node.wfids) == set(curr_wfids):
                num_corr += 1
                guess = True
        else:
            if set(p_node.wfids) == set(curr_wfids[1:]):
                num_corr += 1
                guess = True
        print ('Itern ', itern ,' ' + str(guess))
    #     if True:
    #         bad_nums.append(itern)
    #         print 'Pfid is ' + str(p_node.fid)
    #         tmp_w_array = [world_all_info.data[ids[i]] for i in range(k)]
    #         print 'Wfids predicted are '
    #         # wfids_to_print = ['world/' + str(w.fid) for w in tmp_w_array]
    #         wfids_to_print = ['/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/' + str(w.fid) for w in tmp_w_array]
    #         # wfids_to_print = [str(w.fid) for w in tmp_w_array]
    #         # print [str(w.fid) for w in tmp_w_array]
    #         print wfids_to_print
    #         l1 = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/probe/' + str(p_node.fid)
    #         # l1 = 'world/' + str(p_node.fid)
    #         l1 = [l1] + wfids_to_print
    #         # l1.append(wfids_to_print)
    #         disp_img.show_img(l1)
    #         # imshow_coll.imshow_collection_new(l1)
    #         pdb.set_trace()
        
        total_num += 1

    print (num_corr, total_num)
    # print bad_nums
    # print goods

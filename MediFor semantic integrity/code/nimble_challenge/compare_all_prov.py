# import caffe
# import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    # probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/bvlc_alexnet_probe.pkl', 'rb')
    probe_file = open('../../data/nimble17_data/dev3/alexnet365_probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_probe.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)
    # world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/bvlc_alexnet_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/alexnet365_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_world.pkl', 'rb')
    world_file = open('../../data/nimble17_data/dev3/alexnet365_bigger_world.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    # prov_data = nimble_references(prov_ref_file)
    # prov_data.populate_data()
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
    lol = 0
    # k = 10
    for itern, p_node in enumerate(prov_data.nodes[:]):
        # pdb.set_trace()
        print ("iter no. ", itern)
        pfid = p_node.fid
        p_info_idx = dict_probe[pfid]
        probe_info = probe_all_info.data[p_info_idx]
        fv_probe = probe_info.fv3
        temp_array = np.array([])
        id_array = np.array([])
        # k = len(p_node.wfids)
        k = 1
        for idx, w in enumerate(world_all_info.data):
            if pfid not in w.fid:
                # pdb.set_trace()
                # lol += 1
                corr = two_imgs_eff.cmp_fv(fv_probe, w.fv3, 'ncc')['pear_ncc']
                temp_array = np.append(temp_array, corr)
                id_array = np.append(id_array, idx)
            else:
                lol += 1
        top_k = temp_array.argsort()[-k:]
        top_k = list(reversed(top_k))
        print (temp_array[top_k])
        # print (id_array[top_k])
        # w_base = world_all_info.data[dict_world[p_dat.base_browser_file_name]]
        # act_cor = two_imgs_eff.cmp_fv(fv_probe, w_base.fv3, 'ncc')['pear_ncc']
        # print ("probe file ", p_dat.provenance_probe_file_id)
        # print ("expected base file", p_dat.base_browser_file_name, "expected corr: ", act_cor)
        # dest_top_dir = '../../data/nimble17_data/tmp_folder/'
        # dest_path = dest_top_dir + str(itern)
        # if not os.path.isdir(dest_path):
        # os.mkdir(dest_path)
        # src_p = '/arka_data/NC2017_Dev1_Beta4/probe/' + p_dat.provenance_probe_file_id + '.jpg'
        # shutil.copy2(src_p, dest_path + '/probe_img.jpg')
        ids = id_array[top_k].astype(int)
        # for ind, i in enumerate(ids):
        #     src = '/arka_data/NC2017_Dev1_Beta4/world/' + world_all_info.data[i].fid
        #     if (src[-4:] == '.jpg'):
        #         shutil.copy2(src, dest_path + '/w' + str(ind) + '.jpg')
        #     else:
        #         shutil.copy2(src, dest_path + '/w' + str(ind) + '.png')
        #     print (world_all_info.data[i].fid)
        # src_base = '/arka_data/NC2017_Dev1_Beta4/world/' + p_dat.base_browser_file_name
        # if (src_base[-4:] == '.jpg'):
        #     shutil.copy2(src_base, dest_path + '/w_exp.jpg')
        # else:
        #     shutil.copy2(src_base, dest_path + '/w_exp.png')
        
        for i in range(k):
            # if p_dat.base_browser_file_name in world_all_info.data[top_k[i]].fid:
            guess = False
            # if world_all_info.data[ids[i]].fid[:-4] in p_node.wfids:
            if world_all_info.data[ids[i]].fid.split('.')[0].split('/')[-1] in p_node.wfids:
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
        if not guess:
            bad_nums.append(itern)
            print 'Pfid is ' + str(p_node.fid)
            tmp_w_array = [world_all_info.data[ids[i]] for i in range(k)]
            print 'Wfids predicted are '
            wfids_to_print = ['world/' + str(w.fid) for w in tmp_w_array]
            # print [str(w.fid) for w in tmp_w_array]
            print wfids_to_print
            l1 = 'probe/' + str(p_node.fid)
            l1 = [l1] + wfids_to_print
            # l1.append(wfids_to_print)
            # disp_img.show_img(l1)
            # pdb.set_trace()
        
        total_num += 1

    print (num_corr, total_num)
    print bad_nums
    print goods

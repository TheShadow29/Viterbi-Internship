# import caffe
# import matplotlib.pyplot as plt
import pickle
import two_imgs_eff
import numpy as np
import os
# import sys
import shutil
from get_all_feature_vecs17 import info_storer, info_storer_all
from parse_all_data_nimble17 import nimble_references, nimble_prov_reference, prov_ref_file
import pdb
if __name__ == '__main__':
    probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)
    world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    prov_data = nimble_references(prov_ref_file)
    prov_data.populate_data()

    dict_probe = dict()
    dict_world = dict()
    for i, p in enumerate(probe_all_info.data):
        dict_probe[p.fid[:-4]] = i
    for i, w in enumerate(world_all_info.data):
        dict_world[w.fid] = i

    num_corr = 0
    total_num = 0
    for itern, p_dat in enumerate(prov_data.data[:]):
        # pdb.set_trace()
        print ("iter no. ", itern)
        pfid = p_dat.provenance_probe_file_id
        p_info_idx = dict_probe[pfid]
        probe_info = probe_all_info.data[p_info_idx]
        fv_probe = probe_info.fv3
        temp_array = np.array([])
        id_array = np.array([])
        for idx, w in enumerate(world_all_info.data):
            if pfid not in w.fid:
                corr = two_imgs_eff.cmp_fv(fv_probe, w.fv3, 'ncc')['pear_ncc']
                temp_array = np.append(temp_array, corr)
                id_array = np.append(id_array, idx)
        k = 5
        top_k = temp_array.argsort()[-k:]
        top_k = list(reversed(top_k))
        print (temp_array[top_k])
        # print (id_array[top_k])
        w_base = world_all_info.data[dict_world[p_dat.base_browser_file_name]]
        act_cor = two_imgs_eff.cmp_fv(fv_probe, w_base.fv3, 'ncc')['pear_ncc']
        print ("probe file ", p_dat.provenance_probe_file_id)
        print ("expected base file", p_dat.base_browser_file_name, "expected corr: ", act_cor)
        dest_top_dir = '../../data/nimble17_data/tmp_folder/'
        dest_path = dest_top_dir + str(itern)
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
        src_p = '/arka_data/NC2017_Dev1_Beta4/probe/' + p_dat.provenance_probe_file_id + '.jpg'
        shutil.copy2(src_p, dest_path + '/probe_img.jpg')
        ids = id_array[top_k].astype(int)
        for ind, i in enumerate(ids):
            src = '/arka_data/NC2017_Dev1_Beta4/world/' + world_all_info.data[i].fid
            if (src[-4:] == '.jpg'):
                shutil.copy2(src, dest_path + '/w' + str(ind) + '.jpg')
            else:
                shutil.copy2(src, dest_path + '/w' + str(ind) + '.png')
            print (world_all_info.data[i].fid)
        src_base = '/arka_data/NC2017_Dev1_Beta4/world/' + p_dat.base_browser_file_name
        if (src_base[-4:] == '.jpg'):
            shutil.copy2(src_base, dest_path + '/w_exp.jpg')
        else:
            shutil.copy2(src_base, dest_path + '/w_exp.png')
            
        for i in range(k):
            if p_dat.base_browser_file_name in world_all_info.data[top_k[i]].fid:
                num_corr += 1
                # print
                break
        total_num += 1

    print (num_corr, total_num)

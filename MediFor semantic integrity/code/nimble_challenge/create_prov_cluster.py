import prov_ref_node_creator as prnc
import networkx as nx
import pandas as pd
import pdb
import os
import two_imgs_eff
import pickle
from get_all_feature_vecs17 import info_storer, info_storer_all
import matplotlib.pyplot as plt
# import disp_img
import imshow_coll
import numpy as np
import imshow_coll


class graph_node:
    def __init__(self, fid):
        self.fid = fid
        self.top_5_min = 0

    def get_top5_min(self, G):
        nn5 = np.zeros(1)
        for g in G[self.fid].values():
            nn5 = np.append(nn5, g['weight'])


if __name__ == '__main__':
    # prov_file = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
                # 'reference/provenance/NC2017_Dev3-provenance-ref-node.csv'
    # prov_data = prnc.prov_nodes(prov_file)
    # prov_data.populate_data()

    # probe_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_probe.pkl', 'rb')
    probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)

    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_world.pkl', 'rb')
    world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    dict_probe = dict()
    dict_world = dict()
    for i, p in enumerate(probe_all_info.data):
        dict_probe[p.fid[:-4]] = i
    for i, w in enumerate(world_all_info.data):
        dict_world[w.fid[:-4]] = i

    prov_file_loc = '/arka_data/NC2017_Dev1_Beta4/reference/' + \
                    'provenance/NC2017_Dev1-provenance-ref.csv'
    prov_file = open(prov_file_loc, 'rb')
    reader = pd.read_csv(prov_file, sep='|')
    # G = nx.Graph()
    list_probe_files = []
    for ind, row in reader.iterrows():
        list_probe_files.append(row['ProvenanceProbeFileName'])
    list_world_files = []
    wtdir = '/arka_data/NC2017_Dev1_Beta4/world/'

    for wfile in os.listdir(wtdir):
        if wfile[-4:] == '.jpg' or wfile[-4:] == '.png':
            list_world_files.append(wfile)
    some_problem_files = 0
    clusters = [[]]
    for itern, pfile in enumerate(list_probe_files[:3]):
        pfid = pfile.split('.')[0].split('/')[-1]
        # pid = dict_probe[pfid]
        pid = dict_world[pfid]
        # probe_info = probe_all_info.data[pid]
        probe_info = world_all_info.data[pid]
        fv_probe = probe_info.fv3
        pnode_id = pfid + pfile[-4:]
        # G.add_node(pnode_id)
        curr_pfid_all_corr = dict()
        list_world_files1 = list_world_files
        curr_wids = []
        curr_wfids = []
        while True:
            for wfile in list_world_files1[:]:
                wfid = wfile.split('.')[0]
                try:
                    wid = dict_world[wfid]
                    world_info = world_all_info.data[wid]
                    fv_world = world_info.fv3
                    if pfid != wfid:
                        wnode_id = wfid + wfile[-4:]
                        # G.add_node(wnode_id)
                        mcor1 = 0
                        for w1 in curr_wids:
                            corr1 = two_imgs_eff.cmp_fv(world_all_info.data[w1].fv3, fv_world, 'ncc')['pear_ncc']
                            if corr1 > mcor1:
                                mcor1 = corr1
                        corr2 = two_imgs_eff.cmp_fv(fv_probe, fv_world, 'ncc')['pear_ncc']
                        corr = max(mcor1, corr2)
                        curr_pfid_all_corr[wid] = corr
                        # if corr > 0.9:
                        # G.add_edge(pnode_id, wnode_id, weight=corr)
                except Exception as e:
                    some_problem_files += 1
            max_cor_id = max(curr_pfid_all_corr.iterkeys(), key=(lambda k: curr_pfid_all_corr[k]))
            max_corr = curr_pfid_all_corr[max_cor_id]

            if max_corr < 0.8:
                break
            else:
                curr_wids.append(max_cor_id)
                curr_wfids.append(world_all_info.data[max_cor_id].fid)
                list_world_files1.remove(world_all_info.data[max_cor_id].fid)
        cl_list_to_append = [wtdir + w for w in curr_wfids]
        cl_list_to_append.append(wtdir + pfid)
        clusters.append(cl_list_to_append)
        print ('Itern ', itern)
    fig_save_dir = '../../data/nimble17_data/dev1/que_method/'
    for ind, cl in enumerate(clusters):
        fig = imshow_coll.imshow_collection_new(cl, show=False)
        fig.savefig(fig_save_dir + str(ind) + '.png')
        print ('Ind', ind)

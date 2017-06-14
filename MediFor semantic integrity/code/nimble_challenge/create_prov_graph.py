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
    G = nx.Graph()
    list_probe_files = []
    for ind, row in reader.iterrows():
        list_probe_files.append(row['ProvenanceProbeFileName'])
    list_world_files = []
    wtdir = '/arka_data/NC2017_Dev1_Beta4/world/'

    for wfile in os.listdir(wtdir):
        if wfile[-4:] == '.jpg' or wfile[-4:] == '.png':
            list_world_files.append(wfile)
    some_problem_files = 0
    for itern, pfile in enumerate(list_probe_files[:]):
        pfid = pfile.split('.')[0].split('/')[-1]
        # pid = dict_probe[pfid]
        pid = dict_world[pfid]
        # probe_info = probe_all_info.data[pid]
        probe_info = world_all_info.data[pid]
        fv_probe = probe_info.fv3
        pnode_id = pfid + pfile[-4:]
        G.add_node(pnode_id)
        for wfile in list_world_files[:]:
            wfid = wfile.split('.')[0]
            try:
                wid = dict_world[wfid]
                world_info = world_all_info.data[wid]
                fv_world = world_info.fv3
                if pfid != wfid:
                    wnode_id = wfid + wfile[-4:]
                    G.add_node(wnode_id)
                    corr = two_imgs_eff.cmp_fv(fv_probe, fv_world, 'ncc')['pear_ncc']
                    if corr > 0.9:
                        G.add_edge(pnode_id, wnode_id, weight=corr)
            except Exception as e:
                some_problem_files += 1
                pass
        print ('Itern ', itern)
    fig_save_dir = '../../data/nimble17_data/dev1/'
    for ind, c in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
        c1 = list(c)
        l1 = [wtdir + w for w in c1]
        fig = imshow_coll.imshow_collection_new(l1, show=False)
        fig.savefig(fig_save_dir + str(ind) + '.png')
        print ('Ind ', ind)
        # fig.show()
        # pdb.set_trace()
    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
    # pos = nx.spring_layout(G)   # positions for all nodes

    # # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=700)
    # # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5,
    #                        edge_color='b', style='dashed')

    # # labels
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # plt.axis('off')
    # plt.savefig("weighted_graph.png") # save as png
    # plt.show() # displa

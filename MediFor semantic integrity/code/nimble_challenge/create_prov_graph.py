import prov_ref_node_creator as prnc
import networkx as nx
import pandas as pd
import pdb
import os
import two_imgs_eff
import pickle
# from get_all_feature_vecs17 import info_storer, info_storer_all
from get_all_fv_code_gen import info_storer, info_storer_all
import matplotlib.pyplot as plt
# import disp_img
import imshow_coll
import numpy as np
from itertools import chain
import pickle


def contracted_nodes(G, u, v, self_loops=True):
    """Returns the graph that results from contracting `u` and `v`.
    Node contraction identifies the two nodes as a single node incident to any
    edge that was incident to the original two nodes.
    Parameters
    ----------
    G : NetworkX graph
       The graph whose nodes will be contracted.
    u, v : nodes
       Must be nodes in `G`.
    self_loops : Boolean
       If this is True, any edges joining `u` and `v` in `G` become
       self-loops on the new node in the returned graph.
    Returns
    -------
    Networkx graph
       A new graph object of the same type as `G` (leaving `G` unmodified)
       with `u` and `v` identified in a single node. The right node `v`
       will be merged into the node `u`, so only `u` will appear in the
       returned graph.
    Examples
    --------
    Contracting two nonadjacent nodes of the cycle graph on four nodes `C_4`
    yields the path graph (ignoring parallel edges)::
        >>> import networkx as nx
        >>> G = nx.cycle_graph(4)
        >>> M = nx.contracted_nodes(G, 1, 3)
        >>> P3 = nx.path_graph(3)
        >>> nx.is_isomorphic(M, P3)
        True
    See also
    --------
    contracted_edge
    quotient_graph
    Notes
    -----
    This function is also available as `identified_nodes`.
    """

    H = G.copy()
    if H.is_directed():
        in_edges = ((w, u, d) for w, x, d in G.in_edges(v, data=True)
                    if self_loops or w != u)
        out_edges = ((u, w, d) for x, w, d in G.out_edges(v, data=True)
                     if self_loops or w != u)
        new_edges = chain(in_edges, out_edges)
    else:
        # new_edges = ((u, w, d) for x, w, d in G.edges(v, data=True)
        #              if self_loops or w != u)
        # new_edges = ((u, w, d) for x, w, d in G.edges(v, data=True)
        # if self_loops or w != u)
        new_edges = list()
        nodes_u = [w for x, w in G.edges(u)]
        for x, w, d in G.edges(v, data=True):
            if w != u:
                if w not in nodes_u:
                    new_edges.append((u, w, d))
                else:
                    max_d = max(d['weight'], G[u][w]['weight'])
                    new_edges.append((u, w, max_d))
    v_data = H.node[v]
    pdb.set_trace()
    H.remove_node(v)
    H.add_edges_from(new_edges)
    if 'contraction' in H.node[u]:
        H.node[u]['contraction'][v] = v_data
    else:
        H.node[u]['contraction'] = {v: v_data}
    return H


# class graph_node:
#     def __init__(self, fid):
#         self.fid = fid
#         self.top_5_min = 0

#     # def get_top5_min(self, G):
#     #     nn5 = np.zeros(1)
#     #     for g in G[self.fid].values():
#     #         nn5 = np.append(nn5, g['weight'])
#     def get_thresh(self, G):
#         all_corr = G.edges(self.fid, )

if __name__ == '__main__':
    # prov_file = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
                # 'reference/provenance/NC2017_Dev3-provenance-ref-node.csv'
    # prov_data = prnc.prov_nodes(prov_file)
    # prov_data.populate_data()

    # probe_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    # probe_file = open('../../data/nimble17_data/alexnet365_NC2017_Dev1_Beta4_world.pkl', 'rb')
    # probe_all_info = pickle.load(probe_file)

    # world_file = open('../../data/nimble17_data/dev3/bvlc_alexnet_world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    # world_file = open('../../data/nimble17_data/alexnet365_NC2017_Dev1_Beta4_world.pkl', 'rb')
    # net_name = 'alex'
    world_file = open('../../data/nimble17_data/alexnet365_NC2017_Dev3_Beta1_world.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    # dict_probe = dict()
    # dict_world = dict()
    # for i, p in enumerate(probe_all_info.data):
    #     dict_probe[p.fid[:-4]] = i
    # for i, w in enumerate(world_all_info.data):
    #     dict_world[w.fid[:-4]] = i

    # prov_file_loc = '/arka_data/NC2017_Dev1_Beta4/reference/' + \
    # 'provenance/NC2017_Dev1-provenance-ref.csv'

    prov_file_loc = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
                    'reference/provenance/NC2017_Dev3-provenance-ref.csv'
    prov_file = open(prov_file_loc, 'rb')
    reader = pd.read_csv(prov_file, sep='|')
    G = nx.Graph()
    list_probe_files = []
    for ind, row in reader.iterrows():
        list_probe_files.append(row['ProvenanceProbeFileName'])
    # list_world_files = []
    # wtdir = '/arka_data/NC2017_Dev1_Beta4/world/'
    wtdir = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/'
    out_file_name = '_'.join(wtdir.split('/')[-3:-1])

    # for wfile in os.listdir(wtdir):
    #     if wfile[-4:] == '.jpg' or wfile[-4:] == '.png':
    #         list_world_files.append(wfile)

    some_problem_files = 0

    for itern, pfile in enumerate(list_probe_files[:]):
        # pfid = pfile.split('.')[0].split('/')[-1]
        # pid = dict_probe[pfid]
        # pid = dict_world[pfid]
        pfid = pfile.split('/')[-1]
        pid = world_all_info.data_id_dict[pfid]
        # probe_info = probe_all_info.data[pid]
        probe_info = world_all_info.data[pid]
        fv_probe = probe_info.fv3
        pnode_id = pfid
        G.add_node(pnode_id)
        # for wfile in list_world_files[:]:
        #     wfid = wfile.split('.')[0]
        #     try:
        #         wid = dict_world[wfid]
        #         world_info = world_all_info.data[wid]
        #         fv_world = world_info.fv3
        #         if pfid != wfid:
        #             wnode_id = wfid + wfile[-4:]
        #             G.add_node(wnode_id)
        #             corr = two_imgs_eff.cmp_fv(fv_probe, fv_world, 'ncc')['pear_ncc']
        #             # if corr > 0.9:
        #             G.add_edge(pnode_id, wnode_id, weight=corr)
        #     except Exception as e:
        #         some_problem_files += 1
        #         pass
        for w in world_all_info.data:
            if w.fid != pfid:
                corr = two_imgs_eff.cmp_fv(fv_probe, w.fv3, 'ncc')['pear_ncc']
                G.add_edge(pnode_id, w.fid, weight=corr)

        print ('Itern ', itern)

    g = open('../../data/nimble17_data/' + out_file_name + '_graph_all_corr.pkl', 'w')
    pickle.dump(G, g)
    g.close()
    # Now I have the compelte graph that would be required.
    # Now I should contract similar nodes
    # clusters = [[]]
    # for node in G.nodes():
    #     neighbors = G[node]
    #     max_v = 0
    #     max_k = ''
    #     for k, v in neighbors:
    #         if v > max_v:
    #             max_v = v
    #             max_k = k
        # if (max_v > thresh):
            # G = contracted_nodes(G, node, k)


    # fig_save_dir = '../../data/nimble17_data/dev1/max_grapher/'
    # for ind, c in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
    # c1 = list(c)
    # l1 = [wtdir + w for w in c1]
    # fig = imshow_coll.imshow_collection_new(l1, show=False)
    # fig.savefig(fig_save_dir + str(ind) + '.png')
    # print ('Ind ', ind)
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

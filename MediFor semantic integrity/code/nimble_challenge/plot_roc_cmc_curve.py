import matplotlib.pyplot as plt
import pickle
import pandas as pd
# import networkx as nx
import prov_ref_node_creator as prnc
import numpy as np

def plot_roc(G, list_probe_files, thresholds, pr_nodes):
    tp_list_for_diff_thresh = []
    fp_list_for_diff_thresh = []
    for t in thresholds:
        tp = 0
        fp = 0
        for ind, pfile in enumerate(list_probe_files):
            pfid = pfile.split('/')[-1]
            node_ind = pr_nodes.nodes_dict[pfid[:-4]]
            pnode = pr_nodes.nodes[node_ind]
            # for e in G.edges(pfid, data=True):
            edge_list = [(u,v,d) for (u,v,d) in G.edges(pfid,data=True) if d['weight'] > t]
            bool1 = False
            for u,v,d in edge_list:
                v1 = v[:-4]
                if v1 in pnode.wfids and not bool1:
                    tp += 1
                    bool1 = True
                else:
                    fp += 1
        tp_list_for_diff_thresh.append(tp)
        fp_list_for_diff_thresh.append(fp)
                    
    plt.figure()
    # plt.plot(fp_list_for_diff_thresh, tp_list_for_diff_thresh)
    plt.scatter(fp_list_for_diff_thresh, tp_list_for_diff_thresh)
    plt.show()

if __name__ == '__main__':
    graph_file = open('../../data/nimble17_data/NC2017_Dev1_Beta4_world_graph_all_corr.pkl', 'rb')
    G = pickle.load(graph_file)
    prov_file_loc = '/arka_data/NC2017_Dev1_Beta4/reference/' + \
                    'provenance/NC2017_Dev1-provenance-ref.csv'
    prov_ref_node_file = '/arka_data/NC2017_Dev1_Beta4/reference/provenance/' + \
                         'NC2017_Dev1-provenance-ref-node.csv'
    # prov_file_loc = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
    # 'reference/provenance/NC2017_Dev3-provenance-ref.csv'
    prov_file = open(prov_file_loc, 'rb')
    reader = pd.read_csv(prov_file, sep='|')
    list_probe_files = []
    for ind, row in reader.iterrows():
        list_probe_files.append(row['ProvenanceProbeFileName'])
    # thresholds = [0.95, 0.9, 0.8, 0.5]
    thresholds = np.arange(0, 1, 0.01)
    # print(len(G.edges(data=True)))
    pr_nodes = prnc.prov_nodes(prov_ref_node_file)
    pr_nodes.populate_data()
    plot_roc(G, list_probe_files, thresholds, pr_nodes)

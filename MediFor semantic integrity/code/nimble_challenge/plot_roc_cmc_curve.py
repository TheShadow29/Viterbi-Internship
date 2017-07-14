import matplotlib.pyplot as plt
import pickle
import pandas as pd
# import networkx as nx
import prov_ref_node_creator as prnc
import numpy as np
import pdb
# i


def plot_roc(G, list_probe_files, thresholds, pr_nodes, pr_it=False, pr_ind=False):
    tp_list_for_diff_thresh = []
    fp_list_for_diff_thresh = []
    for itern, t in enumerate(thresholds):
        tp = 0
        fp = 0
        for ind, pfile in enumerate(list_probe_files):
            pfid = pfile.split('/')[-1]
            node_ind = pr_nodes.nodes_dict[pfid[:-4]]
            pnode = pr_nodes.nodes[node_ind]
            # for e in G.edges(pfid, data=True):
            edge_list = [(u, v, d) for (u, v, d) in G.edges(pfid, data=True) if d['weight'] > t]
            # pdb.set_trace()
            bool1 = False
            for u, v, d in edge_list:
                v1 = v[:-4]
                if v1 in pnode.wfids and not bool1:
                    tp += 1
                    bool1 = True
                else:
                    fp += 1

            if pr_ind:
                print ('Ind itern', ind)
        tp_list_for_diff_thresh.append(tp)
        fp_list_for_diff_thresh.append(fp)
        if pr_it:
            print ('T itern ', itern)

    plt.figure()
    # plt.plot(fp_list_for_diff_thresh, tp_list_for_diff_thresh)
    plt.scatter(fp_list_for_diff_thresh, tp_list_for_diff_thresh)
    plt.title('ROC curve with any ref node')
    plt.show()


def plot_roc1(G, dict_probe_world_files, thresholds, pr_nodes, pr_it=False, pr_ind=False):
    tp_list_for_diff_thresh = []
    fp_list_for_diff_thresh = []
    for itern, t in enumerate(thresholds):
        tp = 0
        fp = 0
        # pdb.set_trace()
        for ind, pfile in enumerate(dict_probe_world_files.keys()):
            # pdb.set_trace()
            gt_base_file = dict_probe_world_files[pfile]
            # pfid = pfile.split('/')[-1]
            pfid = pfile
            node_ind = pr_nodes.nodes_dict[pfid[:-4]]
            pnode = pr_nodes.nodes[node_ind]
            # for e in G.edges(pfid, data=True):
            # pdb.set_trace()
            edge_list = [(u, v, d) for (u, v, d) in G.edges(pfid, data=True) if d['weight'] > t]
            # pdb.set_trace()
            for u, v, d in edge_list:
                v1 = v[:-4]
                # pdb.set_trace()
                if v == gt_base_file:
                    tp += 1
                elif v1 in pnode.wfids:
                    # tp += 1
                    pass
                else:
                    fp += 1
            if pr_ind:
                print ('Ind ', ind)
        tp_list_for_diff_thresh.append(tp)
        fp_list_for_diff_thresh.append(fp)
        print ('Itern ', itern)
    plt.figure()
    # plt.plot(fp_list_for_diff_thresh, tp_list_for_diff_thresh)
    plt.scatter(fp_list_for_diff_thresh, tp_list_for_diff_thresh)
    plt.title('ROC curve discarding ref-nodes')
    plt.show()
    return fp_list_for_diff_thresh, tp_list_for_diff_thresh

# def plot_roc_rec_baseline(fname):
#     f = open(fname, 'rb')
#     l1 = f.readlines()
#     l11 = l1[:2156]
#     l12 = l1[2156:-2]
#     int_l = []
#     tot_l = []
#     for li in l11:
#          int_l.append(int(li.split(',')[0].split('[')[-1].split(']')[0]))
#     for li in l12:
#          tot_l.append(int(li.split(',')[0].split('[')[-1].split(']')[0]))

#     int_np = np.array(int_l)
#     tot_np = np.array(tot_l)
#     fp_list = tot_np - int_np


def plot_cmc(G, dict_probe_world_files, ranks, pr_nodes):
    tpir_list_for_diff_ranks = np.zeros(len(ranks))
    for ind, pfile in enumerate(dict_probe_world_files.keys()):
        gt_base_file = dict_probe_world_files[pfile]
        pfid = pfile.split('/')[-1]
        node_ind = pr_nodes.nodes_dict[pfid[:-4]]
        pnode = pr_nodes.nodes[node_ind]
        # -dat['weight'] to get in descending order
        edge_list_sorted = sorted(G.edges(pfid, data=True), key=lambda (src, dest, dat) : -dat['weight'])
        for ind1, r in enumerate(ranks):
            # pdb.set_trace()
            edge_list_upto_rank_r = edge_list_sorted[:r]
            wid_edges = [v for (u, v, d) in edge_list_upto_rank_r]
            # pdb.set_trace()
            # if gt_base_file in edge_list_upto_rank_r:
            if gt_base_file in wid_edges:
                tpir_list_for_diff_ranks[ind1] += 1
            # else:
            #     for w in wid_edges:
            #         if G[w][gt_base_file] > 0.95:
            #             tpir_list_for_diff_ranks[ind1] += 1
    plt.figure()
    plt.scatter(ranks, tpir_list_for_diff_ranks)
    plt.title('ROC curve discarding ref-nodes')
    plt.show()
    return tpir_list_for_diff_ranks
    # return fp_list_for_diff_thresh, tp_list_for_diff_thresh


if __name__ == '__main__':
    # graph_file = open('../../data/nimble17_data/NC2017_Dev1_Beta4_world_graph_all_corr.pkl', 'rb')
    graph_file = open('../../data/nimble17_data/NC2017_Dev3_Beta1_world_graph_all_corr.pkl', 'rb')
    G = pickle.load(graph_file)
    print ('-------- Graph File Loaded --------')
    # prov_file_loc = '/arka_data/NC2017_Dev1_Beta4/reference/' + \
    #                 'provenance/NC2017_Dev1-provenance-ref.csv'
    # prov_ref_node_file = '/arka_data/NC2017_Dev1_Beta4/reference/provenance/' + \
    #                      'NC2017_Dev1-provenance-ref-node.csv'
    prov_file_loc = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
                    'reference/provenance/NC2017_Dev3-provenance-ref.csv'
    prov_ref_node_file = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/' + \
                         'reference/provenance/NC2017_Dev3-provenance-ref-node.csv'
    # 'reference/provenance/NC2017_Dev3-provenance-ref.csv'
    prov_file = open(prov_file_loc, 'rb')
    reader = pd.read_csv(prov_file, sep='|')
    list_probe_files = []
    dict_probe_world_files = dict()
    for ind, row in reader.iterrows():
        list_probe_files.append(row['ProvenanceProbeFileName'])
        dict_probe_world_files[row['ProvenanceProbeFileName'].split('/')[-1]] = row['BaseBrowserFileName']
    # thresholds = [0.95, 0.9, 0.8, 0.5]
    thresholds = np.arange(0, 1, 0.01)
    ranks = range(1, 30)
    # print(len(G.edges(data=True)))
    pr_nodes = prnc.prov_nodes(prov_ref_node_file)
    pr_nodes.populate_data()
    plot_roc(G, list_probe_files, thresholds, pr_nodes)
    # fpl, tpl = plot_roc1(G, dict_probe_world_files, thresholds, pr_nodes)
    # fpl, tpl = plot_roc2(G, dict_probe_world_files, thresholds, pr_nodes)
    # plot_cmc(G, dict_probe_world_files, ranks, pr_nodes)

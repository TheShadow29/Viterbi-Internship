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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


if __name__ == '__main__':
    probe_file = open('../../data/nimble17_data/probe.pkl', 'rb')
    probe_all_info = pickle.load(probe_file)
    world_file = open('../../data/nimble17_data/world.pkl', 'rb')
    world_all_info = pickle.load(world_file)

    # prov_data = nimble_references(prov_ref_file)
    # prov_data.populate_data()
    prov_data = prnc.prov_nodes(prnc.pref_file)
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
    all_data_dict = dict()
    # all_data = np.array([np.ones(365,)])
    all_dict = dict()
    count = 0
    for itern, p_node in enumerate(prov_data.nodes[:]):
        print ("iter no. ", itern)
        pfid = p_node.fid
        p_info_idx = dict_probe[pfid]
        probe_info = probe_all_info.data[p_info_idx]
        fv_probe = probe_info.fv3
        # temp_array = np.array([])
        # id_array = np.array([])
        # k = len(p_node.wfids)
        # k = 1
        # all_data = np.array([[fv_probe]])

        # all_data = np.append(all_data, [fv_probe], axis=0)
        all_data_dict[pfid] = fv_probe
        all_dict[pfid] = count
        # pdb.set_trace()
        count += 1
        # total_num += 1
        for wfid in p_node.wfids:
            try:
                w_node_id = dict_world[wfid]
                w_info = world_all_info.data[w_node_id]
                # all_data = np.append(all_data, [w_info.fv3], axis=0)
                all_dict[wfid] = count
                all_data_dict[wfid] = w_info.fv3
                # pdb.set_trace()
                count += 1
            except Exception as e:
                print ("World node not found: ", wfid)
                pass
            total_num += 1
    all_data = np.array([np.ones(365,)])
    for k, v in all_data_dict.items():
        all_data = np.append(all_data, [v], axis=0)
    all_data = all_data[1:, ...]
    keys = all_data_dict.keys()
    # all_data = np.transpose(all_data)
    # print (num_corr, total_num)
    # print bad_nums
    # print goods
    # X = np.array([[1, 2], [1, 4], [1, 0],
    #               [4, 2], [4, 4], [4, 0]])
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    # reduced_data = PCA(n_components=2).fit_transform(all_data)
    # reduced_data = X
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    kmeans = KMeans(init='k-means++', n_clusters=65, n_init=1000)
    # kmeans.fit(reduced_data)
    kmeans.fit(all_data)
    src_tdir = '/arka_data/NC2017_Dev1_Beta4/world/'
    dest_tdir = '../../data/nimble17_data/cluster_folder/'
    files_in_src = os.listdir(src_tdir)
    for i, l in enumerate(kmeans.labels_):
        dest_path = dest_tdir + str(l)
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
        key = keys[i]
        for fname in files_in_src:
            if key in fname and (fname[-4:] == '.jpg' or fname[-4:] == '.png'):
                break
        
        src = src_tdir + fname
        shutil.copy2(src, dest_path)
        print ('Files done: ', i)
    # h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each

    # x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    # y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # # Obtain labels for each point in mesh. Use last trained model.
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.cm.Paired,
    #            aspect='auto', origin='lower')

    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    # plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
    #           'Centroids are marked with white cross')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

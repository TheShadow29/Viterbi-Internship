import os
import pdb
import multiprocessing as mp
import time
import caffe
import numpy as np
import skimage
import pickle


def slice_img(img, slice_id):
    assert img.shape[2] == 3
    if (slice_id == 0):  # Height/2
        im_new1 = img[:img.shape[0]/2, :, :]
        im_new2 = img[img.shape[0]/2:, :, :]
    elif (slice_id == 1):  # width/2
        im_new1 = img[:, :img.shape[1]/2, :]
        im_new2 = img[:, img.shape[1]/2:, :]
    return im_new1, im_new2


def worker_task(img_top_dir, transformer, slice_id, bbox_dict='none'):
    # global count
    # Convention followed :
    # before preprocess : img
    # after preprocess : im
    # if (bbox_dict == 'none'):
    backoff = 0.1

    while True:
        if q1.qsize() < 100:
            img_file_name = q2.get()
            # img1_path, img2_path = img_paths(img_folder_num, img_top_dir)
            img1 = caffe.io.load_image(img_top_dir + img_file_name)
            if (bbox_dict != 'none' and bbox_dict != 'hist_eq'):
                pts = bbox_dict[img_file_name[:-4]]
                img1 = img1[pts[1]:pts[3], pts[0]:pts[2]]
                # skimage.io.imshow(img1)
            elif (bbox_dict == 'hist_eq'):
                img1 = skimage.exposure.equalize_hist(img1)

            img1_s1, img1_s2 = slice_img(img1, slice_id)
            # img2_s1, img2_s2 = slice_img(img2,slice_id)
            im1 = transformer.preprocess('data', img1)
            im1_s1 = transformer.preprocess('data', img1_s1)
            im1_s2 = transformer.preprocess('data', img1_s2)
            # im2_s1 = transformer.preprocess('data', img2_s1)
            # im2_s2 = transformer.preprocess('data', img2_s2)

            # q1.put(((im1_s1,im2_s1),(im1_s2,im2_s2),img_folder_num))
            q1.put((im1_s1, im1_s2, im1, img_file_name))
            # q1.put(())
            # except Exception as e:
            # print 'img_folder_num ' + str(img_folder_num)
            # raise e

            # pdb.set_trace()
            # raise e
            # pass
        else:
                backoff *= 2
                # time.sleep(backoff)


class info_storer:
    def __init__(self, _fid, _layer, _fv1, _fv2, _fv3):
        # self.fol = _fol
        self.fid = _fid
        self.layer = _layer
        self.fv1 = _fv1
        self.fv2 = _fv2
        self.fv3 = _fv3


class info_storer_all:
    def __init__(self, _fol):
        self.folder_name = _fol
        self.data = []

    def add_one_info(self, info_storer):
        self.data.append(info_storer)


if __name__ == '__main__':
    start_time = time.time()
    # img_path_name = lambda x : '../../data/nimble17_data/NC2016_' + str(x) + '.jpg'
    # img_path_orig = lambda x,y : '/arka_data/NC2016_Test0613/' + str(x) + '/NC2016_' + '%04d' %y + '.jpg'
    # img_path_1 = lambda x : '../../data/nimble_data/manipulated/' + str(x)
    # caffe_model_dir = '../../data/caffe_model/alexnet365/'
    caffe_model_dir = '/home/arka_s/Caffe/caffe/models/bvlc_alexnet/'
    descriptor_path = caffe_model_dir + 'deploy.prototxt'
    # descriptor_path = caffe_model_dir + 'deploy_alexnet_places365.prototxt'
    weights_path = caffe_model_dir + 'bvlc_alexnet.caffemodel'
    # weights_path = caffe_model_dir + 'alexnet_places365.caffemodel'
    net_name = caffe_model_dir.split('/')[-2]
    ilsvrc_mean_path = '/home/arka_s/Caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    net = caffe.Net(descriptor_path, weights_path, caffe.TEST)

    # transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ilsvrc_mean_path).mean(1).mean(1))

    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    # batch_size = 1
    slice_id = 0
    # changing batch size from 2 to 4, because there are 4 slices.
    net.blobs['data'].reshape(3, 3, 227, 227)
    q1 = mp.Queue()
    q2 = mp.Queue()
    # img_top_dir = '../../data/nimble_data/manipulated/'
    # img_top_dir = '../../data/nimble17_data/manipulated/' #
    # img_top_dir = '../../data/nimble17_data/spliced/' #
    # img_top_dir = '../../data/nimble17_data/provenance/'
    # img_top_dir = '/arka_data/NC2017_Dev1_Beta4/world/'
    # img_top_dir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Pruned_Protest_YFCCImages/'
    # img_top_dir = '../../data/protest_data/cropped/direct_cropped/'
    img_top_dir = '/mnt/disk1/ark_data/NC2017_Dev3_Beta1/NC2017_Dev3_Beta1/world/'
    # img_top_dir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    folder_name = img_top_dir.split('/')[-2]
    # folder_name = 'Modified_Images_ProtestL'
    # res = ''
    dict_npy_file = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/mod_fine_labels.npy'
    bbox_dict_npy = np.load(dict_npy_file)
    bbox_dict = bbox_dict_npy.item()
    bbox_dict = 'none'
    store_all = info_storer_all(folder_name)
    # img_file_names = os.listdir(img_top_dir)
    skimage.io.use_plugin('matplotlib')
    # for subdir in img_dirs:
    #     if img_paths(subdir, img_top_dir) == -1:
    #         img_dirs.remove(subdir)
    img_file_names = []
    for files in os.listdir(img_top_dir):
        if files[-4:] == '.jpg' or files[-4:] == '.png':
            img_file_names.append(files)
    num_process = mp.cpu_count() - 1
    # count = [0]
    for i in img_file_names:
        q2.put(i)
        # pdb.set_trace()
    # pdb.set_trace()
    workers = [mp.Process(target=worker_task, args=(img_top_dir, transformer, slice_id, bbox_dict)) for i in range(num_process)]
    for w in workers:
        w.start()
        total_num = 0
    while total_num < len(img_file_names):
        # while True:
        # for didx in range(batch_size):
        try:
            # im_tuple1 : im1_s1, im2_s1
            # im_tuple2 : im1_s2, im2_s2
            # im_tuple1,im_tuple2,im_f_n = q1.get()
            im_s1, im_s2, im1, im_file_name = q1.get()
            net.blobs['data'].data[0, :, :, :] = im_s1
            net.blobs['data'].data[1, :, :, :] = im_s2
            net.blobs['data'].data[2, :, :, :] = im1
            # net.blobs['data'].data[3,:,:,:] = im_tuple2[1]
            #
            out = net.forward()
            # layer = 'prob'
            # layer = 'fc7'
            layer = 'fc8'

            fv1 = net.blobs[layer].data[0].flatten()
            fv2 = net.blobs[layer].data[1].flatten()
            fv3 = net.blobs[layer].data[2].flatten()
            in1 = info_storer(im_file_name, layer, fv1, fv2, fv3)
            store_all.add_one_info(in1)
        except Exception as e:
            raise e
            # print (e)
        total_num += 1
        # print ('Total Num Completed: ' + str(total_num) + ' img_dir_num ' + str(im_f_n) +' ' + str(to_pr[3]['pear_ncc']) + ' ' +
        # str(to_pr2[3]['pear_ncc']))
        print ('Total Num Completed: ' + str(total_num))

    # g = open('../../data/nimble17_data/results/pb_comp_'+folder_name + layer + '_slice' +str(slice_id) + '.txt','w')
    # g.write(res)
    # g.close()

    # g = open('../../data/protest_data/' + folder_name + '_hist_eq.pkl', 'w')
    g = open('../../data/nimble17_data/' + net_name + '_' + folder_name + '.pkl', 'w')
    info_storer_all.__module__ = "get_all_feature_vecs_protest"
    pickle.dump(store_all, g)
    g.close()
    print("--- %s seconds ---" % (time.time() - start_time))

import numpy as np
import matplotlib.pyplot as plt
import caffe
import pickle
import sys
# from multiprocessing import Process
import multiprocessing as mp
# Pros = []
import time
def all_test_imgs(descriptor_path, weights_path, test_img_top_dir_path,labels_file, ilsvrc_mean_path,val_ground_truth):
    net = caffe.Net(descriptor_path, weights_path, caffe.TEST)

    #transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ilsvrc_mean_path).mean(1).mean(1))

    transformer.set_transpose('data',(2,0,1))
    transformer.set_channel_swap('data',(2,1,0))
    transformer.set_raw_scale('data', 255.0)

    g = open(val_ground_truth,'rb')
    num_corr_pred = 0
    total_num = 0;
    for line_val_file in g:

        # img_str = 'Places365_test_' + "%08d"%2 +'.jpg'
        str_list = line_val_file.split(" ")
        img_str = str_list[0]
        gt_label = int(str_list[1].split('\n')[0])
        im = caffe.io.load_image(test_img_top_dir_path + img_str)
        net.blobs['data'].reshape(1,3,227,227)
        net.blobs['data'].data[:,:,:] = transformer.preprocess('data',im)

        out = net.forward()
        # if (total_num % 100 == 0):
        print ('Iter ' + str(total_num))

        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        if gt_label in top_k:
            num_corr_pred += 1
        total_num += 1


        
    # with open(labels_file, 'rb') as f:
    #     labels = pickle.load(f)
    #     top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #     for i,k in enumerate(top_k):
    #         print i, labels[k]
    print ("Num corr pred: " + str(num_corr_pred))
    print ("Total num: " + str(total_num))
    return num_corr_pred, total_num

    

if __name__ == '__main__':
    start_time = time.time()
    caffe_model_dir = '../../data/caffe_model/alexnet365/'
    descriptor_path = caffe_model_dir + 'deploy_alexnet_places365.prototxt'
    weights_path = caffe_model_dir + 'alexnet_places365.caffemodel'
    # test_img_top_dir_path  = '/arka_data/places_data/test_256/'
    test_img_top_dir_path = '/arka_data/places_data/val_256/'
    val_ground_truth = '/arka_data/places_data/filelist_places365-standard/places365_val_' + sys.argv[1] + '.txt'
    labels_file = '../../data/labels.pkl'
    ilsvrc_mean_path = '/home/arka_s/Caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    print(all_test_imgs(descriptor_path,weights_path,test_img_top_dir_path,labels_file,ilsvrc_mean_path,val_ground_truth))
    print("--- %s seconds ---" % (time.time() - start_time))
    # for i in range(10):
    #     print ("Thread " + str(i) + " started")
    #     p = mp.Process(target = all_test_imgs, args = (descriptor_path,weights_path,test_img_top_dir_path,labels_file,ilsvrc_mean_path,val_ground_truth + str(i) + '.txt'))
    #     Pros.append(p)
    #     p.start()

    # for t in Pros:
    #     t.join()

    

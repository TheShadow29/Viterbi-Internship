import two_imgs_matcher
import os
import caffe
import skimage.io as skio
import numpy as np


if __name__ == '__main__':
    # caffe_model_dir = '../../data/caffe_model/alexnet365/'
    # descriptor_path = caffe_model_dir + 'deploy_alexnet_places365.prototxt'
    # weights_path = caffe_model_dir + 'alexnet_places365.caffemodel'

    caffe_model_dir = '/home/arka_s/Caffe/caffe/models/bvlc_alexnet/'
    descriptor_path = caffe_model_dir + 'deploy.prototxt'
    weights_path = caffe_model_dir + 'bvlc_alexnet.caffemodel'

    ilsvrc_mean_path = '/home/arka_s/Caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    net = caffe.Net(descriptor_path, weights_path, caffe.TEST)
    # skimage.io.use_plugin('matplotlib')
    skio.use_plugin('matplotlib')
    # transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ilsvrc_mean_path).mean(1).mean(1))

    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(2, 3, 227, 227)
    layer = 'fc8'
    img1_tdir = '../../data/protest_data/only_text/beach_mdf_019/'
    img2_tdir = '../../data/protest_data/only_text/protest_img_019/'
    mcor = 0
    max_id = [0, 0]
    itern = 0
    for i in os.listdir(img1_tdir)[:]:
        for j in os.listdir(img2_tdir)[:]:
            itern += 1
            a1 = two_imgs_matcher.are_the_two_imgs_same(img1_tdir + i,
                                                        img2_tdir + j, transformer, net, layer)
            corr = a1[5]['pear_ncc']

            if corr > mcor:
                mcor = corr
                max_id[0] = i
                max_id[1] = j

            print ('Itern ', itern)
    # a1 = two_imgs_matcher.are_the_two_imgs_same(img1_tdir + '0.png',
    # img2_tdir + '1.png', transformer, net)

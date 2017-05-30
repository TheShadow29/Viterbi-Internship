import numpy as np
import matplotlib.pyplot as plt
import caffe
import pickle
import sys

def are_the_two_imgs_same(img1_path, img2_path):
    fv1 = get_feature_vector(img1_path)
    fv2 = get_feature_vector(img2_path)
    # Metrics to try:
    ##Absolute Distance between the two fv
    ##Angle between the two fv
    are_same0 = cmp_fv(fv1,fv2,metric='ssd')#sum of squared distance
    are_same1 = cmp_fv(fv1,fv2,metric='sad')#sum of absolute distance
    are_same2 = cmp_fv(fv1,fv2,metric='ip')#inner product
    ##Only Inner Product ?? => Only Valid for FCN??
    
    
    return are_same0, are_same1, are_same2

def cmp_fv(fv1,fv2,metric='ssd'):
    if metric == 'sad':
        return {
            'metric' : metric,
            'sad' : np.sum(np.absolute(np.subtract(fv1,fv2))),
            # '|fv1|' : np.sum(np.absolute(fv1),axis=0),
            # '|fv2|' : np.sum(np.absolute(fv2),axis=0)
    }
    elif metric == 'ssd':
        return {
            'metric' : metric,
            'ssd' : np.sqrt(np.sum(np.square(np.subtract(fv1,fv2))))
        }
    elif metric == 'ip':
        return {
            'metric' : metric,
            # 'inner_prod' : np.dot(fv1,fv2)
            'inner_prod' : np.inner(fv1,fv2)
        }
    else:
        return -1
    

def get_feature_vector(img_path):
    #first try with direct probabilities
    feature_vector = get_direct_prob(img_path)
    #then try without the softmax probabilities
    return feature_vector

def get_direct_prob(img_path):
    caffe_model_dir = '../../data/caffe_model/alexnet365/'
    descriptor_path = caffe_model_dir + 'deploy_alexnet_places365.prototxt'
    weights_path = caffe_model_dir + 'alexnet_places365.caffemodel'
    ilsvrc_mean_path = '/home/arka_s/Caffe/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    net = caffe.Net(descriptor_path, weights_path, caffe.TEST)

    #transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ilsvrc_mean_path).mean(1).mean(1))

    transformer.set_transpose('data',(2,0,1))
    transformer.set_channel_swap('data',(2,1,0))
    transformer.set_raw_scale('data', 255.0)

    im = caffe.io.load_image(img_path)
    net.blobs['data'].reshape(1,3,227,227)
    net.blobs['data'].data[:,:,:] = transformer.preprocess('data',im)

    out = net.forward()

    fv = net.blobs['prob'].data[0].flatten()
    return fv
if __name__ == '__main__':
    img_top_dir = '../cnn_caffe_places365/'
    img1_path = img_top_dir + 'Places365_test_00000001.jpg'
    img2_path = img_top_dir + 'Places365_test_00000002.jpg'
    # print (are_the_two_imgs_same(img1_path, img2_path))
    

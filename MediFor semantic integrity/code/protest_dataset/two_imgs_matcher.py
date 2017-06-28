import numpy as np
# import matplotlib.pyplot as plt
import caffe
# import pickle
# import sys
import pdb
# import scipy
# import os
# import skimage
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# from scipy.stats.stats import pearsonr


def println(arg):
    print('\n'.join(arg))


def are_the_two_imgs_same(img1_path, img2_path, transformer, net, layer):
    # fv1 = get_feature_vector(img1_path)
    # fv2 = get_feature_vector(img2_path)
    # fv1, fv2 = get_feature_vector(img1_path, img2_path)
    fv1, fv2 = get_direct_prob(img1_path, img2_path, transformer, net, layer)
    # Metrics to try:
    # # Absolute Distance between the two fv
    # # Angle between the two fv
    are_same0 = cmp_fv(fv1, fv2, metric='ssd')   # sum of squared distance
    are_same1 = cmp_fv(fv1, fv2, metric='sad')   # sum of absolute distance
    are_same2 = cmp_fv(fv1, fv2, metric='ip')    # inner product
    are_same3 = cmp_fv(fv1, fv2, metric='ncc')   # normalize cross correlation
    # are_same4 = cmp_fv( fv1, fv2, metric='ct')#census transform
    # #Only Inner Product ?? => Only Valid for FCN??

    return fv1, fv2, are_same0, are_same1, are_same2, are_same3


def cmp_fv(fv1, fv2, metric='ssd'):
    # note: corr here is normalized cross correlation using pearson's coefficient
    if metric == 'sad':
        return {
            'metric': metric,
            'sad': np.sum(np.absolute(np.subtract(fv1, fv2))),
            # '|fv1|' : np.sum(np.absolute(fv1),axis=0),
            # '|fv2|' : np.sum(np.absolute(fv2),axis=0)
        }
    elif metric == 'ssd':
        return {
            'metric': metric,
            'ssd': np.sqrt(np.sum(np.square(np.subtract(fv1, fv2))))
        }
    elif metric == 'ip':
        return {
            'metric': metric,
            # 'inner_prod' : np.dot(fv1,fv2)
            'inner_prod': np.inner(fv1, fv2)
        }
    elif metric == 'ncc':
        return{
            'metric': metric,
            # 'pear_ncc' : scipy.stats.stats.pearsonr(fv1,fv2)
            'pear_ncc': np.corrcoef(fv1, fv2)[0, 1]
        }
    else:
        return -1


# def get_feature_vector(img1_path, img2_path):
#     #first try with direct probabilities
#     feature_vector = get_direct_prob(img1_path, img2_path)
#     #then try without the softmax probabilities
#     return feature_vector

def get_direct_prob(img1_path, img2_path, transformer, net, layer):
    # caffe.io.use_plugin('matplotlib')
    if (type(img1_path) == str):
        im1 = caffe.io.load_image(img1_path)
        im2 = caffe.io.load_image(img2_path)
        # pdb.set_trace()
    else:
        im1 = img1_path
        im2 = img2_path
    net.blobs['data'].data[0, :, :, :] = transformer.preprocess('data', im1)
    net.blobs['data'].data[1, :, :, :] = transformer.preprocess('data', im2)

    out = net.forward()
    # pdb.set_trace()
    fv1 = net.blobs[layer].data[0].flatten()
    fv2 = net.blobs[layer].data[1].flatten()
    # pdb.set_trace()
    return fv1, fv2

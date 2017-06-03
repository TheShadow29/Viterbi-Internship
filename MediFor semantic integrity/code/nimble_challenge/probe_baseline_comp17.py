import two_imgs_eff
import os
import pdb
import multiprocessing as mp
import time
import caffe
import numpy as np
import skimage
# def cmp_tw
def img_paths(img_folder_num, img_top_dir):
    img_dir = img_top_dir + str(img_folder_num)
    im_paths = []
    for d in os.listdir(img_dir):
        if d[-3:] == 'jpg' or d[-3:] == 'png':
            im_paths.append(img_dir + '/' + d)
            # return im_paths[0], im_paths[1]
    if len(im_paths) == 2:
        return im_paths[0], im_paths[1]
    else:
        return -1

def slice_img(img, slice_id):

    assert img.shape[2] == 3
    if (slice_id== 0):#Height/2
        im_new1 = img[:img.shape[0]/2,:,:]
        im_new2 = img[img.shape[0]/2:,:,:]
    elif (slice_id == 1):#width/2
        im_new1 = img[:,:img.shape[1]/2,:]
        im_new2 = img[:,img.shape[1]/2:,:]
    return im_new1, im_new2
def worker_task(img_top_dir,transformer,slice_id):
    # global count
    #Convention followed :
    #before preprocess : img
    #after preprocess : im
    backoff = 0.1
    while True:
        if q1.qsize() < 100:
            # time.sleep(backoff)
            count = q2.get()
            img_folder_num = img_dirs[count]
            # count[0] += 1            
            img1_path, img2_path = img_paths(img_folder_num, img_top_dir)
            img1 = caffe.io.load_image(img1_path)
            # img1 = caffe.io.resize(img1,(227,227,3))
            img2 = caffe.io.load_image(img2_path)
            # img2 = caffe.io.load_image(img2, (227,227,3))

            img1_s1, img1_s2 = slice_img(img1,slice_id)
            img2_s1, img2_s2 = slice_img(img2,slice_id)
            
            im1_s1 = transformer.preprocess('data', img1_s1)
            im1_s2 = transformer.preprocess('data', img1_s2)
            im2_s1 = transformer.preprocess('data', img2_s1)
            im2_s2 = transformer.preprocess('data', img2_s2)
            
            q1.put(((im1_s1,im2_s1),(im1_s2,im2_s2),img_folder_num))
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
            
def get_cmp(fv1,fv2):
    are_same0 = two_imgs_eff.cmp_fv( fv1, fv2, metric='ssd' )#sum of squared distance
    are_same1 = two_imgs_eff.cmp_fv( fv1, fv2, metric='sad' )#sum of absolute distance
    are_same2 = two_imgs_eff.cmp_fv( fv1, fv2, metric='ip' )#inner product
    are_same3 = two_imgs_eff.cmp_fv( fv1, fv2, metric='ncc')#normalize cross correlation

    return (are_same0, are_same1, are_same2, are_same3)

if __name__ == '__main__':
    start_time = time.time()
    # img_path_name = lambda x : '../../data/nimble17_data/NC2016_' + str(x) + '.jpg'
    # img_path_orig = lambda x,y : '/arka_data/NC2016_Test0613/' + str(x) + '/NC2016_' + '%04d' %y + '.jpg'
    # img_path_1 = lambda x : '../../data/nimble_data/manipulated/' + str(x)
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
    # batch_size = 1
    slice_id = 0
    #changing batch size from 2 to 4, because there are 4 slices.
    net.blobs['data'].reshape(4,3,227,227)
    q1 = mp.Queue()
    q2 = mp.Queue()
    # img_top_dir = '../../data/nimble_data/manipulated/'
    img_top_dir = '../../data/nimble17_data/manipulated/'
    res = ''
    img_dirs = os.listdir(img_top_dir)
    skimage.io.use_plugin('matplotlib')
    for subdir in img_dirs:
        if img_paths(subdir, img_top_dir) == -1:
            img_dirs.remove(subdir)
            
    num_process = mp.cpu_count() - 1
    # count = [0]
    for i in range(len(img_dirs)):
        q2.put(i)
        # pdb.set_trace()
    workers = [mp.Process(target = worker_task, args = (img_top_dir,transformer,slice_id)) for i in range(num_process)]
    for w in workers:
        w.start()
        total_num = 0
    while total_num < len(img_dirs):
        # while True:
        # for didx in range(batch_size):
        try:
            #im_tuple1 : im1_s1, im2_s1
            #im_tuple2 : im1_s2, im2_s2
            im_tuple1,im_tuple2,im_f_n = q1.get()
            net.blobs['data'].data[0,:,:,:] = im_tuple1[0]
            net.blobs['data'].data[1,:,:,:] = im_tuple1[1]
            net.blobs['data'].data[2,:,:,:] = im_tuple2[0]
            net.blobs['data'].data[3,:,:,:] = im_tuple2[1]
            
            out = net.forward()
            # layer = 'prob'
            # layer = 'fc7'
            layer = 'fc8'

            fv1 = net.blobs[layer].data[0].flatten()
            fv2 = net.blobs[layer].data[1].flatten()
            fv3 = net.blobs[layer].data[2].flatten()
            fv4 = net.blobs[layer].data[3].flatten()

            to_pr = get_cmp(fv1,fv2)
            to_pr2 = get_cmp(fv3,fv4)

            with open(img_top_dir + str(im_f_n) + '/eval_' + str(layer) + '_slicing' + str(slice_id) + '.txt','wb') as f:
                f.write(str(to_pr))
                f.write('\n')
                f.write(str(to_pr2))
                f.close()
                # res +=  str(to_pr)
            res += str(im_f_n) +' ' + str(to_pr[3]['pear_ncc']) + ' ' + str(to_pr2[3]['pear_ncc']) + '\n'
        except Exception as e:
            # pass
            # raise e
            # pdb.set_trace()
            print (e)
        total_num += 1
        print ('Total Num Completed: ' + str(total_num) + ' img_dir_num ' + str(im_f_n) +' ' + str(to_pr[3]['pear_ncc']) + ' ' +
               str(to_pr2[3]['pear_ncc']))
            
    g = open('../../data/nimble17_data/results/pb_comp_' + layer + '_slice' +str(slice_id) + '.txt','w')
    g.write(res)
    g.close()
    print("--- %s seconds ---" % (time.time() - start_time))

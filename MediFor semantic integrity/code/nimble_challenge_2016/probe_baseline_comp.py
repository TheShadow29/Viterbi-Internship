import two_imgs_eff
import os
import pdb
import multiprocessing as mp
import time
import caffe
import numpy as np

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
def worker_task(img_top_dir,transformer):
    # global count
    backoff = 0.1
    while True:
        if q1.qsize() < 100:
            # time.sleep(backoff)
            count = q2.get()
            img_folder_num = img_dirs[count]
            # count[0] += 1            
            img1_path, img2_path = img_paths(img_folder_num, img_top_dir)
            try:
                img1 = caffe.io.load_image(img1_path)
                img2 = caffe.io.load_image(img2_path)
                im1 = transformer.preprocess('data',img1)
                im2 = transformer.preprocess('data',img2)
                
                q1.put((im1,im2,img_folder_num))
            except Exception as e:
                pass
        else:
            backoff *= 2
            # time.sleep(backoff)
            

if __name__ == '__main__':
    start_time = time.time()
    img_path_name = lambda x : '../../data/nimble_data/NC2016_' + str(x) + '.jpg'
    img_path_orig = lambda x,y : '/arka_data/NC2016_Test0613/' + str(x) + '/NC2016_' + '%04d' %y + '.jpg'
    img_path_1 = lambda x : '../../data/nimble_data/manipulated/' + str(x)
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
    net.blobs['data'].reshape(2,3,227,227)
    q1 = mp.Queue()
    q2 = mp.Queue()
    img_top_dir = '../../data/nimble_data/manipulated/'
    res = ''
    img_dirs = os.listdir(img_top_dir)

    for subdir in img_dirs:
        if img_paths(subdir, img_top_dir) == -1:
            img_dirs.remove(subdir)
        
    num_process = mp.cpu_count() - 1
    # count = [0]
    for i in range(len(img_dirs)):
        q2.put(i)
    # pdb.set_trace()
    workers = [mp.Process(target = worker_task, args = (img_top_dir,transformer)) for i in range(num_process)]
    for w in workers:
        w.start()
    total_num = 0
    while total_num < 300:
        # for didx in range(batch_size):
        try:
            im1,im2,im_f_n = q1.get()
            net.blobs['data'].data[0,:,:,:] = im1
            net.blobs['data'].data[1,:,:,:] = im2
            out = net.forward()

            fv1 = net.blobs['prob'].data[0].flatten()
            fv2 = net.blobs['prob'].data[1].flatten()
            are_same0 = two_imgs_eff.cmp_fv( fv1, fv2, metric='ssd' )#sum of squared distance
            are_same1 = two_imgs_eff.cmp_fv( fv1, fv2, metric='sad' )#sum of absolute distance
            are_same2 = two_imgs_eff.cmp_fv( fv1, fv2, metric='ip' )#inner product
            are_same3 = two_imgs_eff.cmp_fv( fv1, fv2, metric='ncc')#normalize cross correlation
            
            to_pr = (are_same0, are_same1, are_same2, are_same3)
            with open(img_top_dir + str(im_f_n) + '/eval.txt','wb') as f:
                f.write(str(to_pr))
                f.close()
            # res +=  str(to_pr)
            res += str(im_f_n) +' ' + str(to_pr[3]['pear_ncc']) + '\n'
        except Exception as e:
            pass
        total_num += 1
        print ('Total Num Completed: ' + str(total_num) + ' img_dir_num ' + str(im_f_n) +' ' + str(to_pr[3]['pear_ncc']))
              
            
        
    
    
    # for subdir in os.listdir(img_top_dir):
    # # subdir = '601'
    #     bo = True
    #     img_dir = img_top_dir + subdir + '/'
    #     dir_paths = os.listdir(img_dir)
    #     imgs_paths = []
    #     for k in dir_paths:
    #         if k[-4:] == '.jpg' or k[-4:] == '.png':
    #             imgs_paths.append(k)
    #     if len(imgs_paths) == 2:
    #         img1_path = img_dir + imgs_paths[0]
    #         img2_path = img_dir + imgs_paths[1]
    #         try:
    #             # to_print = two_imgs.are_the_two_imgs_same(img1_path, img2_path)
    #         except Exception as e :
    #             bo = False
    #             pass
    #         # pdb.set_trace()
    #         # with open(img_dir + 'eval.txt', 'w') as f:
    #         if bo == True:
    #             f = open(img_dir + 'eval.txt','w')
    #             f.write(str(to_print))
    #             # f.write('acb')
    #             f.close()
    #             res += str(subdir) + str(to_print[3]['pear_ncc']) + '\n'
    #         i += 1
    #         print ('Iter: ' + str(i) +  ' Counter ' + str(subdir))
    g = open('../../data/nimble_data/pb_comp.txt','w')
    g.write(res)
    g.close()
    print("--- %s seconds ---" % (time.time() - start_time))

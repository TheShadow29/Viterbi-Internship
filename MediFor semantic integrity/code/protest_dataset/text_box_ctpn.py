import sys
sys.path.append('/home/arka_s/Caffe/cf/caffe/python/')
import caffe
# sys.path.append('../../CTPN_forked/src/')
# sys.path.append('../../CTPN_forked/tools/')
sys.path.append('./ctpn/src/')
import detectors
import other
import os
import skimage.io as skio
import skimage.transform as skt
# import cfg
import pdb
import matplotlib.pyplot as plt
import pickle
import numpy as np


def resize_img(img, scale, max_scale=None):
    f = float(scale)/min(img.shape[0], img.shape[1])
    if max_scale is not None and f*max(img.shape[0], img.shape[1]) > max_scale:
        f = float(max_scale)/max(img.shape[0], img.shape[1])
    # return skt.resize(im, (0, 0), fx=f, fy=f), f
    return skt.rescale(img, f), f


if __name__ == '__main__':
    # img_tdir = '/mnt/disk1/ark_data/code_manip/dev3/seed_1/'
    # img_tdir = '/mnt/disk1/ark_data/code_manip/protest/'
    # img_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Pruned_Protest_YFCCImages/'
    # img_tdir = '../../CTPN_forked/demo_images/'
    # img_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    img_tdir = '../../data/protest_data/cropped/direct_cropped/'
    folder_name = img_tdir.split('/')[-2]
    # model_tdir = '../../CTPN_forked/models/'
    deploy_file = '../../CTPN_forked/models/deploy.prototxt'
    weight_file = '../../CTPN_forked/models/ctpn_trained_model.caffemodel'
    caffe.set_device(0)
    # out_tdir = '../../data/protest_data/tl_modified/'
    out_tdir = '../../data/protest_data/tl_cropped/'
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_mean('data', np.load(ilsvrc_mean_path).mean(1).mean(1))

    # transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_channel_swap('data', (2, 1, 0))
    # transformer.set_raw_scale('data', 255.0)

    # Uncomment only for the Modified_Images_ProtestL
    bbx_file = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/mod_fine_labels.npy'
    bbx_npy = np.load(bbx_file)
    bbox_dict = bbx_npy.item()
    # is_img_modified = True
    is_img_modified = False

    # config_params = cfg.Config
    text_proposals_detector = detectors.TextProposalDetector(
        other.CaffeModel(deploy_file, weight_file))
    text_detector = detectors.TextDetector(text_proposals_detector)
    img_file_list = []
    for fls in os.listdir(img_tdir):
        if fls[-4:] == '.png' or fls[-4:] == '.jpg':
            img_file_list.append(fls)
    dict_img_text_lines = dict()
    exceptions = 0
    no_text_images = 0
    for itern, img_fname in enumerate(img_file_list):
        try:
            img1 = skio.imread(img_tdir + img_fname)
            # img1_resized, f = resize_img(img1, , config_params.MAX_SCALE)
            if is_img_modified:
                gt_box = bbox_dict[img_fname[:-4]]
                img1_resized = img1[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]]
            else:
                img1_resized = img1

            text_lines = text_detector.detect(img1_resized)
            im_with_text_lines = other.draw_boxes(img1_resized,
                                                  text_lines, caption=img_fname, wait=False)
            # pdb.set_trace()
            print (text_lines)
            dict_img_text_lines[img_fname] = text_lines
            # fig = plt.figure(0)
            # plt.imshow(im_with_text_lines)
            # plt.show()
            # pdb.set_trace()
            skio.imsave(out_tdir + folder_name + '_' + img_fname + '_tl.png', im_with_text_lines)
            if text_lines.shape[0] == 0:
                no_text_images += 1

            # pdb.set_trace()
        except ValueError as e:
            exceptions += 1
            print (e)
            pass
        print ('Itern ', itern)
        if itern % 10 == 0:
            print ('exceptions ', exceptions, ' no_text_images', no_text_images)
    g = open('../../data/protest_data/dict_' + folder_name + '_text_lines.pkl', 'w')
    pickle.dump(dict_img_text_lines, g)
    g.close()

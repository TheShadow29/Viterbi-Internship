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
    img_tdir = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/'
    # model_tdir = '../../CTPN_forked/models/'
    deploy_file = '../../CTPN_forked/models/deploy.prototxt'
    weight_file = '../../CTPN_forked/models/ctpn_trained_model.caffemodel'
    caffe.set_device(0)

    # config_params = cfg.Config
    text_proposals_detector = detectors.TextProposalDetector(
        other.CaffeModel(deploy_file, weight_file))
    text_detector = detectors.TextDetector(text_proposals_detector)
    img_file_list = []
    for fls in os.listdir(img_tdir):
        if fls[-4:] == '.png' or fls[-4:] == '.jpg':
            img_file_list.append(fls)
    for itern, img_fname in enumerate(img_file_list):
        img1 = skio.imread(img_tdir + img_fname)
        # img1_resized, f = resize_img(img1, , config_params.MAX_SCALE)
        img1_resized = img1
        text_lines = text_detector.detect(img1_resized)
        im_with_text_lines = other.draw_boxes(img1_resized,
                                              text_lines, caption=img_fname, wait=False)
        pdb.set_trace()
        print ('Itern ', itern)


        

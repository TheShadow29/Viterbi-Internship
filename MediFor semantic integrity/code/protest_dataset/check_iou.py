import numpy as np
import os
import pickle
import scipy.io as sio
import pdb

def get_area(bbox):
    '''
    Assumes bbox is in the format
    x1, y1, x2, y2
    '''
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def get_iou(bbox_pred, bbox_gt):
    '''
    Assumes the bbox is in the format:
    x1, y1, x2, y2
    (x1, y1) being bottom left corner
    for the format of matplotlib imshow
    (x2, y2) being top right corner
    '''
    
    x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred
    x1_gt, y1_gt, x2_gt, y2_gt = bbox_gt
    intersection_coordinates = (max(x1_pred, x1_gt), max(y1_pred, y1_gt),
                                min(x2_pred, x2_gt), min(y2_pred, y2_gt))
    area_overlap = get_area(intersection_coordinates)
    if area_overlap < 0:
        # pdb.set_trace()
        area_overlap = 0
    area_union = get_area(bbox_pred) + get_area(bbox_gt) - area_overlap
    return area_overlap, area_union, float(area_overlap)/float(area_union)


if __name__ == '__main__':
    img_tdir = '/arka_data/places_data/copied_data/'
    file_list = []
    for f in os.listdir(img_tdir):
        if f[-4:] == '.jpg' or f[-4:] == '.png':
            file_list.append(f)
            
    gt_dict_bbox = pickle.load(open('../../data/protest_data/code_manip_bbox_dict.pkl', 'rb'))
    pred_dict_bbox = pickle.load(open('./code_manip_bbox_dict.pkl', 'rb'))
    total_num = 0
    num_cor = 0
    for f1 in file_list[:]:
        iou_list = list()
        b1 = gt_dict_bbox[f1]
        b1[2] = b1[0] + b1[2]
        b1[3] = b1[1] + b1[3]
        b2 = pred_dict_bbox[f1]
        for i in range(len(b2[:])):
            _, _, iou = get_iou(b2[i] - 1, b1)
            iou_list.append(iou)
        # print max(iou_list)
        iou_np = np.array(iou_list)
        ids = iou_np.argsort()[::-1]
        print ids[0], b2[ids[0]], iou_np[ids[0]]
        total_num += 1
        if iou_np[ids[0]] > 0.5:
            num_cor += 1
    print (num_cor, total_num, float(num_cor)/float(total_num))

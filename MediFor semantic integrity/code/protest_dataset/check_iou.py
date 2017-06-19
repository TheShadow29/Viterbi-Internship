import numpy as np


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
    area_union = get_area(bbox_pred) + get_area(bbox_gt) - area_overlap
    return float(area_overlap)/float(area_union)

if __name__ == '__main__':
    

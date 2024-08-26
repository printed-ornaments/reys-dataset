import numpy as np
import torch

#from gmair.utils.bbox.bbox import bbox_overlaps

#from gmair.config import config as cfg

import os
import numpy as np
import cv2
import torch

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score

import bbox_overlaps
#from gmair.config import config as cfg

from ipdb import set_trace


def get_bbox_labels(z_where, z_cls, obj_prob, ground_truth_bbox, conf_thresh = 0.5, iou_thresh = 0.5):
    #image_size = cfg.input_image_shape[-1]
    image_size = 128
    batch_size = z_where.size(0)

    z_where = z_where.view(batch_size, -1, 4).detach().cpu().numpy()
    z_cls = z_cls.view(batch_size, -1, cfg.num_classes).detach().cpu().numpy()
    obj_prob = obj_prob.view(batch_size, -1, 1).detach().cpu().numpy()

    z_where[..., :2] -= z_where[..., 2:]/2
    z_where *= image_size
    
    z_pred = np.concatenate((z_where, obj_prob), axis = 2)
    
    ground_truth_bbox = ground_truth_bbox.detach().cpu().numpy()

    true_labels = np.zeros(0, dtype=np.int64)
    pred_labels = np.zeros(0, dtype=np.int64)
    
    for i in range(batch_size):
        pred_info = z_pred[i, z_pred[i,:,4]>=conf_thresh, :].astype('float64')
        pred_cls = z_cls[i, z_pred[i,:,4]>=conf_thresh, :]
        pred_cls = np.argmax(pred_cls, axis=1)
        
        gt_boxes = ground_truth_bbox[i, ground_truth_bbox[i,:,0]>=0, :].astype('float64')
        
    
        _pred = pred_info.copy()
        _gt = gt_boxes.copy()
        true_label = np.zeros(_pred.shape[0], dtype=np.int64)
    
        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        overlaps = bbox_overlaps(_pred[:, :4], _gt[:, :4])

        if _gt.shape[0] != 0:
            for h in range(_pred.shape[0]):
                gt_overlap = overlaps[h]
                max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
                if max_overlap >= iou_thresh:
                    true_label[h] = _gt[max_idx, 4]
    
        true_labels = np.concatenate((true_labels, true_label), axis = 0)
        pred_labels = np.concatenate((pred_labels, pred_cls), axis = 0)
   
    pred_labels = pred_labels[true_labels > 0]
    true_labels = true_labels[true_labels > 0] - 1
    return pred_labels, true_labels

def image_eval(pred, gt, iou_thresh, pred_info_class):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    cnt = 0
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
                cnt += 1

        pred_recall[h] = cnt
    return pred_recall


def img_pr_info(thresh_num, pred_info, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    
    r_index = -1
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        while r_index + 1 < len(pred_info) and pred_info[r_index, 4] >= thresh:
            r_index += 1
        
        pr_info[t, 0] = r_index + 1
        pr_info[t, 1] = 0 if r_index == -1 else pred_recall[r_index]
        
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_obj):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        if pr_curve[i, 1] == 0:
            _pr_curve[i, 0] = 0
            _pr_curve[i, 1] = 0
        else:
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_obj
    return _pr_curve

def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    
def mAP(z_where, score, ground_truth_bbox, truth_bbox_digit_count, conf_thresh=0.5, iou_thresh=0.5, thresh_num=1):
    image_size =128
    batch_size = score.shape[0]
    num_classes = 20
    
    assert(batch_size == z_where[0].shape[0])
    
    z_where_bbox, z_where_class = z_where
    z_where_bbox = z_where_bbox.reshape(batch_size, -1, 4)
    z_where_class = z_where_class.reshape(batch_size, -1, num_classes)
    
    score = score.reshape(batch_size, -1, 1)

    z_where_bbox[..., :2] -= z_where_bbox[..., 2:]/2
    z_where_bbox *= image_size
    
    z_pred_bbox = np.concatenate((z_where_bbox, score), axis=2)
    #ground_truth_bbox = ground_truth_bbox.cpu().numpy()

    count_obj = {class_idx: 0 for class_idx in range(num_classes)}
    pr_curve = np.zeros((num_classes, thresh_num, 2)).astype('float')
    
    for i in range(batch_size):
        pred_info_bbox = z_pred_bbox[i, z_pred_bbox[i,:,4]>=conf_thresh, :].astype('float64')
        pred_info_bbox = pred_info_bbox[pred_info_bbox[:, 4] > 0.7]  # Filtering by score
        pred_info_bbox = pred_info_bbox[np.argsort(pred_info_bbox[:, 4])[::-1]]
        pred_info_class = z_where_class[i, z_pred_bbox[i,:,4]>=conf_thresh, :].astype('float64')
        pred_info_class = pred_info_class[np.argsort(pred_info_bbox[:, 4])[::-1]]
        
        gt_boxes = ground_truth_bbox[i, ground_truth_bbox[i,:,0]>=0,:].astype('float64')

        for class_idx in range(num_classes):
            count_obj[class_idx] += len(gt_boxes[gt_boxes[:, -1] == class_idx])

            if len(gt_boxes) == 0 or len(pred_info_bbox) == 0:
                continue

            pred_recall = image_eval(pred_info_bbox, gt_boxes, iou_thresh, pred_info_class)

            _img_pr_info = img_pr_info(thresh_num, pred_info_bbox, pred_recall)

            pr_curve[class_idx] += _img_pr_info
            
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_obj)

    propose = pr_curve[:, :, 0]
    recall = pr_curve[:, :, 1]

    ap_per_class = [voc_ap(recall[class_idx], propose[class_idx]) for class_idx in range(num_classes)]
    
    mAP = np.mean(ap_per_class)
    
    return mAP


def object_count_accuracy(z_pres:torch.Tensor, truth_bbox_digit_count):

    batch_size = z_pres.size(0)
    z_pres = z_pres.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
    z_pres_count = z_pres.round().sum(dim = -2)

    count_accuracy = (truth_bbox_digit_count - z_pres_count).mean()
    return count_accuracy


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A B / A B = A B / (area(A) + area(B) - A B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    

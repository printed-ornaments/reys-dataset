import numpy as np


def calculate_iou_threshold(prediction, ground_truth_mask, threshold, t_id, conf_mat):
    # Compute the iou given a threshold
    prediction = (prediction > threshold).astype('int')
    tp = np.count_nonzero(np.logical_and(prediction, ground_truth_mask))  # True positive
    fp = np.count_nonzero(np.logical_and(prediction, 1-ground_truth_mask))  # False positive
    fn = np.count_nonzero(np.logical_and(1-prediction, ground_truth_mask))  # False negative
    tn = np.count_nonzero(np.logical_and(1-prediction, 1-ground_truth_mask))  # True negative
    iou = tp/(tp + fp + fn) * 100
    tp_rate = tp/(tp+fn)
    fp_rate = fp/(fp+tn)
    conf_mat[t_id] += np.array([[tn, fn], [fp, tp]])
    return iou, tp_rate, fp_rate, conf_mat


def calculate_iou(prediction, ground_truth_mask, conf_mat, num_threshold):
    # Compute the iou for all thresholds
    iou_list, tp_rates, fp_rates = [], [], []
    for t_id, threshold in enumerate(np.linspace(1, 0, num_threshold)):
        iou, tp_rate, fp_rate, conf_mat = calculate_iou_threshold(prediction,
                                                                  ground_truth_mask,
                                                                  threshold,
                                                                  t_id,
                                                                  conf_mat)
        iou_list.append(iou)
        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)
    return iou_list, tp_rates, fp_rates, conf_mat


def compute_auc(tp_rates, fp_rates):
    # Compute the AUROC metric
    delta_xs = np.diff(fp_rates)
    left_endpoints_y = np.array(tp_rates[:-1])
    right_endpoints_y = np.array(tp_rates[1:])
    trap_areas = 0.5 * (left_endpoints_y + right_endpoints_y) * delta_xs
    auc = trap_areas.sum()
    return auc * 100


class Metrics:
    def __init__(self, glyphs, num_threshold):
        self.num_threshold = num_threshold
        self.glyphs = glyphs
        self.all_ious = np.zeros((len(glyphs), num_threshold))
        self.conf_mat = np.zeros((num_threshold, 2, 2))
        self.aucs = []

    def update(self, pred, gt, glyph_idx):
        ious, tp_rates, fp_rates, conf_mat = calculate_iou(pred, gt, self.conf_mat, self.num_threshold)
        self.conf_mat = conf_mat
        self.aucs.append(compute_auc(tp_rates, fp_rates))
        self.all_ious[glyph_idx] = ious

    def compute_scores(self):
        all_iou_mean = np.mean(self.all_ious, axis=0)
        thresholds = np.linspace(1, 0, self.num_threshold)
        selected_threshold_idx = np.argmax(all_iou_mean)
        selected_threshold = thresholds[selected_threshold_idx]
        all_iou_at_threshold = self.all_ious[:, selected_threshold_idx]
        conf_mat_at_threshold = self.conf_mat[selected_threshold_idx]
        overall_iou = conf_mat_at_threshold[1, 1]/(conf_mat_at_threshold[1, 1] +
                                                   conf_mat_at_threshold[0, 1] +
                                                   conf_mat_at_threshold[1, 0]) * 100
        mean_iou = all_iou_at_threshold.mean()
        tp_rates = [cm[1, 1]/(cm[1, 1]+cm[0, 1]) for cm in self.conf_mat]
        fp_rates = [cm[1, 0]/(cm[1, 0]+cm[0, 0]) for cm in self.conf_mat]
        overall_auc = compute_auc(tp_rates, fp_rates)
        mean_auc = np.mean(self.aucs)
        metrics = {'threshold': selected_threshold,
                   'mean_iou': mean_iou,
                   'overall_iou': overall_iou,
                   'mean_auc': mean_auc,
                   'overall_auc': overall_auc,
                   'all_iou_at_threshold': {g: iou for g, iou in zip(self.glyphs, all_iou_at_threshold)}
                   }
        return metrics

import os
import numpy as np
import torch
import colormap
from scipy.optimize import linear_sum_assignment

def IOU(gt, pred):
    """
    Mean Intersection over Union metric 
    params:
    -------
        gt: np.array (h x w)
        pred: np.array (h x w)
    """
    if len(gt.shape) > 2:
        gt, gt_labels = colormap.as_labels(gt, return_labels=True)
    else:
        gt_labels = np.unique(gt)

    if len(pred.shape) > 2:
        pred, pred_labels = colormap.as_labels(pred, return_labels=True)
    else:
        pred_labels = np.unique(pred)
    
    labels = set(np.concatenate([gt_labels, pred_labels], axis=0))
    intersections = np.zeros(len(labels))
    unions = np.zeros(len(labels))
    for i, label in enumerate(labels):
        intersections[i] = np.sum((gt == label) & (pred == label))
        unions[i] = np.sum((gt == label) | (pred == label))

    ious = intersections / unions
    return ious


def average_precision(gt, pred, threshold=0.5):
    """
    Computes average precision at a given threshold
    AP = TP s
    """

def hungarian_assigmnent(gt, pred):
    """
    Compute an optimal assignment of predicted labels to ground truth
    labels using the hungarian algorithm. 

    Args: 
        gt (`np.ndarray`):
            Array of labels with shape (height x width)
        pred (`np.ndarray`):
            Array of predicted labels with shape (height x width)
    
    Returns:
        assignment (`np.ndarray`):
            The assignment of predicted labels to ground truth labels
        metric (`np.float64`):
            The metric evaluation for the given assignment
    """
    assert len(gt.shape) == 2 and len(pred.shape) == 2

    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)
    n_labels = max(len(gt_labels), len(pred_labels))

    cost_matrix = np.zeros((n_labels, n_labels))  
    label_matrix = np.ones((n_labels, n_labels)) * -1   # unnasigned labels will be given a -1 

    # build cost matrix
    # For each predicted label, compute the iou
    # if it was associated with a different ground truth label   
    import matplotlib.pyplot as plt
    for i, pred_label in enumerate(pred_labels):
        for j, gt_label in enumerate(gt_labels):
            pred_match = pred == pred_label
            gt_match = gt == gt_label

            intersection = np.sum(pred_match & gt_match)
            union = np.sum(pred_match | gt_match)
            score = intersection / union
            cost_matrix[i, j] = score   
            label_matrix[i, j] = gt_label
            
    row_inds, col_inds = linear_sum_assignment(cost_matrix=cost_matrix, maximize=True)
    labels = label_matrix[row_inds, col_inds].astype(np.int8)
    costs = cost_matrix[row_inds, col_inds]
    metric = np.mean(costs)
    
    return metric, labels
    
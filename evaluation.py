"""
  @Time    : 2018-11-20 06:07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : SemirNet
  @File    : test.py
  @Function: 
  
"""
import os
import numpy as np
import skimage.io
from PIL import Image


def get_mask(imgname, MASK_DIR):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    gt_folder = MASK_DIR
    gt_path = gt_folder + "/" + filestr + ".png"
    if not os.path.exists(gt_path):
        print("{} has no gt mask".format(filestr))
    mask = skimage.io.imread(gt_path)

    mask = np.where(mask == 255, 1, 0).astype(np.uint8)

    return mask


def get_predict_mask(imgname, PREDICT_MASK_DIR):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = PREDICT_MASK_DIR
    mask_path = mask_folder + "/" + filestr + ".jpg"
    if not os.path.exists(mask_path):
        print("{} has no predict mask".format(filestr))
    mask = skimage.io.imread(mask_path)

    mask = np.where(mask >= 127.5, 1, 0).astype(np.uint8)

    return mask


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def iou(predict_mask, gt_mask):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Here, n_cl = 1 as we have only one class (mirror).
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        IoU = 0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def accuracy_all(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = (TP + TN) / (N_p + N_n)

    return accuracy_


def accuracy_mirror(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = TP/N_p

    return accuracy_


def ber(predict_mask, gt_mask):
    """
    BER: balance error rate.
    :param predict_mask:
    :param gt_mask:
    :return:
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    if N_p + N_n != 640 * 512:
        raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    return ber_
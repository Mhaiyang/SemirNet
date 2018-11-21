"""
  @Time    : 2018-11-20 18:07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : SemirNet
  @File    : test.py
  @Function: 
  
"""
import os
import evaluation


gt_dir = "/home/taylor/SemirNet/data/test/mask"
# predict_dir = "/home/taylor/SemirNet/data/test/dsc_results"
predict_dir = "/home/taylor/SemirNet/ckpt/BDRAR/(BDRAR)sbu_prediction_3001"

IOU = []
ACC_all = []
ACC_mirror = []
BER = []

masklist = os.listdir(gt_dir)
for i, maskname in enumerate(masklist):
    print(i, maskname)
    gt = evaluation.get_mask(maskname, gt_dir)
    predict = evaluation.get_predict_mask(maskname, predict_dir)

    iou = evaluation.iou(predict, gt)
    acc_all = evaluation.accuracy_all(predict, gt)
    acc_mirror = evaluation.accuracy_mirror(predict, gt)
    ber = evaluation.ber(predict, gt)

    print("iou : {}".format(iou))
    print("acc : {}".format(acc_all))
    print("acc : {}".format(acc_mirror))
    print("ber : {}".format(ber))
    IOU.append(iou)
    ACC_all.append(acc_all)
    ACC_mirror.append(acc_mirror)
    BER.append(ber)

mean_IOU = 100 * sum(IOU)/len(IOU)
mean_ACC_all = 100 * sum(ACC_all)/len(ACC_all)
mean_ACC_mirror = 100 * sum(ACC_mirror)/len(ACC_mirror)
mean_BER = 100 * sum(BER)/len(BER)

print(len(IOU))
print(len(ACC_all))
print(len(ACC_mirror))
print(len(BER))

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f}".
      format("mean_IOU", mean_IOU, "mean_ACC_all", mean_ACC_all, "mean_ACC_mirror", mean_ACC_mirror, "mean_BER", mean_BER))

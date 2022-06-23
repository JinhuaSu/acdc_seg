# READ nii.gz data
# save to the common image
# calculate circle loss
#%%
import nibabel as nib
import model
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tfwrapper import losses
import os

nlabels = 4
dilation_filter = tf.ones((5, 5, 4), tf.float32)
#%%

data_names = [
    one.split(".")[0]
    for one in os.listdir("./model_prediction/baseline/prediction")
    # if "ED" in one
]
data_names
#%%
data_name = data_names[0]
#%%
from sklearn import metrics
from tqdm import tqdm


def process(data_name):
    # usage doc: https://nipy.org/nibabel/gettingstarted.html
    # ES:
    # ED:
    baseline_prediction = "./model_prediction/baseline/prediction/%s.nii.gz" % data_name
    ground_truth = "./model_prediction/baseline/ground_truth/%s.nii.gz" % data_name
    our_prediction = "./model_prediction/ours/prediction/%s.nii.gz" % data_name
    # (256, 256, 9)  {0,1,2,3} uint8

    raw_image = "./model_prediction/ours/image/%s.nii.gz" % data_name
    # (256, 256, 9)  [0, 445] mean=83 std:80
    nifty_img = nib.load(raw_image)
    raw_data = nifty_img.get_fdata()

    # nifty_img.shape

    # # %%
    # nifty_img.get_data_dtype()
    # # %%
    # nifty_img.affine.shape
    # # %%
    # # %%
    # raw_data.std()
    # >>> X.shape
    # (5, 212, 212, 1)
    # >>> X.max()
    # 8.266695
    # >>> X.min()
    # -1.0366102
    # >>> y.shape
    # (5, 212, 212)
    #%%
    our_pred_img = nib.load(our_prediction)
    our_pred = our_pred_img.get_fdata()
    #%%
    base_pred_img = nib.load(baseline_prediction)
    base_pred = base_pred_img.get_fdata()
    #%%
    ground_truth_label = nib.load(ground_truth)
    truth_label = ground_truth_label.get_fdata()
    # convert the raw data into the predict mode(use the same norm as the evaluate_patients.py)

    # Notice the cropping for predict (from 256 to 212)
    min_size = min(212, raw_data.shape[0], raw_data.shape[1])
    min_size
    #%%
    X = np.expand_dims(np.swapaxes(raw_data, 0, 2), -1)[:, :min_size, :min_size, :]
    y = np.swapaxes(our_pred, 0, 2)[:, :min_size, :min_size]
    y_base = np.swapaxes(base_pred, 0, 2)[:, :min_size, :min_size]
    label = np.swapaxes(truth_label, 0, 2)[:, :min_size, :min_size]
    if min_size < 212:
        pad_size = 212 - min_size
        X = np.pad(X, ((0, 0), (0, pad_size), (0, pad_size), (0, 0)), "minimum")
        y = np.pad(y, ((0, 0), (0, pad_size), (0, pad_size)), "minimum")
        y_base = np.pad(y_base, ((0, 0), (0, pad_size), (0, pad_size)), "minimum")
        label = np.pad(label, ((0, 0), (0, pad_size), (0, pad_size)), "minimum")

    one_hot_y = tf.one_hot(y, nlabels)
    one_hot_y_base = tf.one_hot(y_base, nlabels)
    one_hot_label = tf.one_hot(label, nlabels)
    our_loss_part_all = []
    base_loss_part_all = []
    gold_loss_part_all = []
    with tf.Session() as sess:

        # segmentation_loss = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=one_hot_y, labels=one_hot_label
        # )
        # score = sess.run(segmentation_loss)
        # # segmentation_loss = losses.pixel_wise_cross_entropy_loss(one_hot_y, one_hot_label)
        # segmentation_loss_base = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=one_hot_y_base, labels=one_hot_label
        # )
        # score_base = sess.run(segmentation_loss_base)
        # metrics.f1
        score = metrics.f1_score(
            label.flatten(), y.flatten(), [0, 1, 2, 3], average="macro"
        )
        score_base = metrics.f1_score(
            label.flatten(), y_base.flatten(), [0, 1, 2, 3], average="macro"
        )
        for i in tqdm(range(X.shape[0])):
            loss, _, _ = model.RAW_Student_Circle_Loss(
                tf.convert_to_tensor(X[i : (i + 1), ...], tf.float32),
                one_hot_y,
                dilation_filter,
            )
            loss_base, _, _ = model.RAW_Student_Circle_Loss(
                tf.convert_to_tensor(X[i : (i + 1), ...], tf.float32),
                one_hot_y_base,
                dilation_filter,
            )
            loss_gold, _, _ = model.RAW_Student_Circle_Loss(
                tf.convert_to_tensor(X[i : (i + 1), ...], tf.float32),
                one_hot_label,
                dilation_filter,
            )
            losses1 = sess.run(loss)
            losses_base = sess.run(loss_base)
            losses_gold = sess.run(loss_gold)
            base_loss_part_all.append(list(losses_base))
            our_loss_part_all.append(list(losses1))
            gold_loss_part_all.append(list(losses_gold))
        # cmasks = sess.run(cmask)
        # cmasks2 = sess.run(cmask2)
        # cmasks_base = sess.run(cmask_base)
        # cmasks2_base = sess.run(cmask2_base)
    # batch_size = raw_data.shape[2]
    # min_ = 0
    # max_ = 445
    # for i in range(batch_size):
    #     one = X[i, ..., 0]
    #     rgb_one = (one - min_) / (max_ - min_) * 256
    #     im = Image.fromarray(rgb_one)
    #     im.convert("L").save("./contrast_picture/" + data_name + "_%s.png" % i)

    return {
        "base_student_loss": base_loss_part_all,
        "our_student_loss": our_loss_part_all,
        "gold_student_loss": gold_loss_part_all,
        "base_score": score,
        "our_score": score_base,
    }


# %%
# TODO: get the Loss as X, get the simple Extropy as the y
# write the csv as the X_loss result and y, save id to reverse the data
# %%
from tqdm import tqdm

data_dict = {}
for data_name in tqdm(data_names[:1]):
    data_dict[data_name] = process(data_name)
# %%
import pandas as pd

df = pd.DataFrame(data_dict).T
df.to_csv("contrast_evaluate_data_all_part.csv")
# %%
# process(data_name)
# # %%
# data_dict["patient085_ES"]["base_score"][12].shape
# # %%
# data_dict["patient085_ES"]["our_student_loss"]
# # $$
# %%

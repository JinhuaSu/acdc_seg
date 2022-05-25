# READ nii.gz data
# save to the common image
# calculate circle loss
#%%
import nibabel as nib
import importlib
import model

importlib.reload(model)
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tfwrapper import losses
import os

#%%
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
# def process(data_name):
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
# %%
# (9, 212, 212, 1)
X = np.expand_dims(np.swapaxes(raw_data, 0, 2), -1)[:, :min_size, :min_size, :]
# (9, 212, 212)
y = np.swapaxes(our_pred, 0, 2)[:, :min_size, :min_size]
y_base = np.swapaxes(base_pred, 0, 2)[:, :min_size, :min_size]
label = np.swapaxes(truth_label, 0, 2)[:, :min_size, :min_size]
if min_size < 212:
    pad_size = 212 - min_size
    X = np.pad(X, ((0, 0), (0, pad_size), (0, pad_size), (0, 0)), "minimum")
    y = np.pad(y, ((0, 0), (0, pad_size), (0, pad_size)), "minimum")
    y_base = np.pad(y_base, ((0, 0), (0, pad_size), (0, pad_size)), "minimum")
    label = np.pad(label, ((0, 0), (0, pad_size), (0, pad_size)), "minimum")
#%%

X = np.zeros((1, 212, 212, 1))
X[:, 70:130, 70:130, :] = 3.5
y = np.zeros((1, 212, 212))
y[:, 70:130, 70:130] = 1
y_base = np.zeros((1, 212, 212))
y_base[:, 90:150, 90:150] = 1
label = np.zeros((1, 212, 212))
label[:, 70:130, 70:130] = 1
#%%
nlabels = 2
# dilation_filter = tf.ones((5, 5, nlabels), tf.float32)
dilation_filter = tf.ones((5, 5, 1), tf.float32)
#%%
one_hot_y = tf.one_hot(y, nlabels)
one_hot_y_base = tf.one_hot(y_base, nlabels)
one_hot_label = tf.one_hot(label, nlabels)
our_loss_part_all = []
base_loss_part_all = []
#%%
with tf.Session() as sess:

    segmentation_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=one_hot_y, labels=one_hot_label
    )
    score = sess.run(segmentation_loss)
    # segmentation_loss = losses.pixel_wise_cross_entropy_loss(one_hot_y, one_hot_label)
    segmentation_loss_base = tf.nn.softmax_cross_entropy_with_logits(
        logits=one_hot_y_base, labels=one_hot_label
    )
    score_base = sess.run(segmentation_loss_base)
    for i in range(X.shape[0]):
        loss, cmask, cmask2 = model.RAW_Student_Circle_Loss(
            tf.convert_to_tensor(X[i : (i + 1), ...], tf.float32),
            one_hot_y[..., 1:],
            dilation_filter,
        )
        loss_base, cmask_base, cmask2_base = model.RAW_Student_Circle_Loss(
            tf.convert_to_tensor(X[i : (i + 1), ...], tf.float32),
            one_hot_y_base[..., 1:],
            dilation_filter,
        )
        losses1 = sess.run(loss)
        losses_base = sess.run(loss_base)
        cmask = sess.run(cmask)
        cmask2 = sess.run(cmask2)
        # base_loss_part_all.append(list(losses_base))
        # our_loss_part_all.append(list(losses1))

# %%
losses1
#%%
losses_base
# TODO: get the Loss as X, get the simple Extropy as the y
# write the csv as the X_loss result and y, save id to reverse the data
# %%
score.mean()
# %%
score_base.mean()
# process(data_name)
# # %%
# data_dict["patient085_ES"]["base_score"][12].sha
# # %%
# data_dict["patient085_ES"]["our_student_loss"]
# # $$
# %%
with tf.Session() as sess:
    test = sess.run(one_hot_label - one_hot_y)
# %%
test

# %%
test.min()
# %%

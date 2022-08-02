#%%
import nibabel as nib
import model
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
# %%
from matplotlib import cm


def draw_plot(data_name):
    baseline_prediction = "./model_prediction/baseline/prediction/%s.nii.gz" % data_name
    ground_truth = "./model_prediction/baseline/ground_truth/%s.nii.gz" % data_name
    our_prediction = "./model_prediction/ours/prediction/%s.nii.gz" % data_name
    raw_image = "./model_prediction/ours/image/%s.nii.gz" % data_name
    nifty_img = nib.load(raw_image)
    raw_data = nifty_img.get_fdata()
    our_pred_img = nib.load(our_prediction)
    our_pred = our_pred_img.get_fdata()
    base_pred_img = nib.load(baseline_prediction)
    base_pred = base_pred_img.get_fdata()
    ground_truth_label = nib.load(ground_truth)
    truth_label = ground_truth_label.get_fdata()
    min_size = min(212, raw_data.shape[0], raw_data.shape[1])
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

    batch_size = raw_data.shape[2]
    min_ = 0
    max_ = 445
    for i in range(batch_size):
        one = X[i, ..., 0]
        rgb_one = (one - min_) / (max_ - min_) * 256
        im = Image.fromarray(rgb_one)
        # number to mask to color
        im_y = Image.fromarray(np.uint8(cm.gist_earth(y[i] / 4) * 255))
        im_y_base = Image.fromarray(np.uint8(cm.gist_earth(y_base[i] / 4) * 255))
        im_y_true = Image.fromarray(np.uint8(cm.gist_earth(label[i] / 4) * 255))
        #  (9, 212, 212)
        # 学习Image如何图层
        # im_y.convert("RGBA").save("./contrast_picture_plot/our_predict" + data_name + "_%s.png" % i)
        im.convert("L").save("./contrast_picture_plot/" + data_name + "_%s_raw.png" % i)
        our_pred_img = Image.blend(
            im.convert("L").convert("RGBA"), im_y.convert("RGBA"), 0.4
        )
        base_pred_img = Image.blend(
            im.convert("L").convert("RGBA"), im_y_base.convert("RGBA"), 0.4
        )
        ground_true_img = Image.blend(
            im.convert("L").convert("RGBA"), im_y_true.convert("RGBA"), 0.4
        )
        our_pred_img.save(
            "./contrast_picture_plot/" + data_name + "_%s_our_predict.png" % i
        )
        base_pred_img.save(
            "./contrast_picture_plot/" + data_name + "_%s_base_predict.png" % i
        )
        ground_true_img.save(
            "./contrast_picture_plot/" + data_name + "_%s_ground_true.png" % i
        )
        # 透明的颜色 叠加两张图片


#%%
for data_name in data_names:
    draw_plot(data_name)

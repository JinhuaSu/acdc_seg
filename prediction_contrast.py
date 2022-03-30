# READ nii.gz data
# save to the common image
# calculate circle loss
#%%
import nibabel as nib
import model
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

nlabels = 4
dilation_filter = tf.ones((5, 5, 4), tf.float32)
#%%

# usage doc: https://nipy.org/nibabel/gettingstarted.html
# ES:
# ED:
baseline_prediction = "./model_prediction/baseline/prediction/patient025_ED.nii.gz"
ground_truth = "./model_prediction/baseline/ground_truth/patient025_ED.nii.gz"
our_prediction = "./model_prediction/ours/prediction/patient025_ED.nii.gz"
# (256, 256, 9)  {0,1,2,3} uint8

raw_image = "./model_prediction/ours/image/patient025_ED.nii.gz"
# (256, 256, 9)  [0, 445] mean=83 std:80
nifty_img = nib.load(raw_image)

# %%
nifty_img.shape

# %%
nifty_img.get_data_dtype()
# %%
nifty_img.affine.shape
# %%
raw_data = nifty_img.get_fdata()
# %%
raw_data.std()
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
# %%
# convert the raw data into the predict mode(use the same norm as the evaluate_patients.py)
import numpy as np

X = np.expand_dims(np.swapaxes(raw_data, 0, 2), -1)[:, 22:-22, 22:-22, :]
# %%
y = np.swapaxes(our_pred, 0, 2)[:, 22:-22, 22:-22]

# %%
one_hot_y = tf.one_hot(y, nlabels)
# %%
with tf.Session() as sess:
    loss, cmask, cmask2 = model.RAW_Student_Circle_Loss(
        tf.convert_to_tensor(X, tf.float32), one_hot_y, dilation_filter
    )
    losses = sess.run(loss)
    cmasks = sess.run(cmask)
    cmasks2 = sess.run(cmask2)
# %%
batch_size = 9
min_ = 0
max_ = 445
for i in range(batch_size):
    one = X[i, ..., 0]
    rgb_one = (one - min_) / (max_ - min_) * 256
    im = Image.fromarray(rgb_one)
    im.convert("L").save("test" + ".png")

# %%

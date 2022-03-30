#%%
import imp
from numpy import imag
import acdc_data
import config.system as sys_config
from experiments import unet2D_bn_modified_wxent as exp_config

data = acdc_data.load_and_maybe_process_data(
    input_folder=sys_config.data_root,
    preprocessing_folder=sys_config.preproc_folder,
    mode=exp_config.data_mode,
    size=exp_config.image_size,
    target_resolution=exp_config.target_resolution,
    force_overwrite=False,
    split_test_train=True,
)

# %%
images_train = data["images_train"]
labels_train = data["masks_train"]

# %%
exp_config.use_data_fraction
images_train.shape
images_train.dtype
labels_train.shape
labels_train.dtype
# >>> images_train.shape
# (1516, 212, 212)
# >>> images_train.dtype
# dtype('<f4')
# >>> labels_train.shape
# (1516, 212, 212)
# >>> labels_train.dtype
# dtype('uint8')
#%%
import numpy as np

images = images_train
labels = labels_train
batch_size = exp_config.batch_size
random_indices = np.arange(images.shape[0])
np.random.shuffle(random_indices)

n_images = images.shape[0]
b_i = 0
# HDF5 requires indices to be in increasing order
batch_indices = np.sort(random_indices[b_i : b_i + batch_size])

X = images[batch_indices, ...]
y = labels[batch_indices, ...]

image_tensor_shape = [X.shape[0]] + list(exp_config.image_size) + [1]
X = np.reshape(X, image_tensor_shape)
# >>> X.shape
# (5, 212, 212, 1)
# >>> X.max()
# 8.266695
# >>> X.min()
# -1.0366102
# >>> y.shape
# (5, 212, 212)
# >>> y.max()
# 3
# >>> y.min()
# 0
# >>> (y == 2).sum()
# 3467
# >>> (y == 0).sum()
# 214561
# >>> (y == 1).sum()
# 2886
# >>> (y == 3).sum()
# 3806
#%%
import importlib
import model
import tensorflow as tf

importlib.reload(model)
nlabels = 4
import numpy as np

#%%
dilation_filter = tf.ones((5, 5, 4), tf.float32)

#%%
import matplotlib.pyplot as plt
from PIL import Image

save_dir = "picture/"
part_num = 10
max_pic_num = 1516
for id_ in range(max_pic_num // batch_size * batch_size):
    if id_ % batch_size == 0:
        b_i = id_
        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i : b_i + batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        image_tensor_shape = [X.shape[0]] + list(exp_config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)

        one_hot_y = tf.one_hot(y, nlabels)
        with tf.Session() as sess:
            loss, cmask, cmask2 = model.RAW_Student_Circle_Loss(
                tf.convert_to_tensor(X), one_hot_y, dilation_filter
            )
            losses = sess.run(loss)
            cmasks = sess.run(cmask)
            cmasks2 = sess.run(cmask2)
            batch_id = id_ % batch_size
    min_ = -2
    max_ = 8
    one = X[batch_id, ..., 0]
    rgb_one = (one - min_) / (max_ - min_) * 256
    im = Image.fromarray(rgb_one)
    im.convert("L").save(save_dir + str(id_) + ".png")

    for part_id in range(part_num):
        mask_one = (
            cmasks[part_id][batch_id, ..., 1]
            + cmasks[part_id][batch_id, ..., 2]
            + cmasks[part_id][batch_id, ..., 3]
            + cmasks2[part_id][batch_id, ..., 1]
            + cmasks2[part_id][batch_id, ..., 2]
            + cmasks2[part_id][batch_id, ..., 3]
        ) > 3
        mask_im = Image.fromarray(mask_one * rgb_one)
        mask_im.convert("L").save(
            save_dir
            + str(id_)
            + "_mask_label(part_%s)_%s.png" % (part_id, losses[part_id])
        )

    label_one = y[batch_id]
    label_im = Image.fromarray(label_one * 100)
    label_im.convert("L").save(save_dir + str(id_) + "_label.png")

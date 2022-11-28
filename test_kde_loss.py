import tensorflow as tf
from tfwrapper import losses
import numpy
import math
from PIL import Image, ImageDraw


def get_circle_masks(pred_label, part_num):
    circle_masks = []
    for i in range(pred_label.shape[0]):
        # there is some problem with the center point calculation
        multi_class_masks = []
        for j in range(pred_label.shape[-1]):
            one_center_point = tf.reduce_mean(tf.where(pred_label[i, ..., j]), axis=0)
            print(one_center_point.eval())
            one_circle_mask = get_circle_mask(
                one_center_point, pred_label.shape, part_num
            )
            print("one_circle_mask", tf.reduce_sum(one_circle_mask).eval())
            multi_class_masks.append(one_circle_mask)
        print("multi_class_masks", tf.stack(multi_class_masks, axis=2).get_shape())
        circle_masks.append(tf.stack(multi_class_masks, axis=2))
    print("circle_masks", tf.stack(circle_masks).get_shape())
    return tf.stack(circle_masks)


def get_raw_radia_mask(w, h, circle_n):
    img = Image.new("L", (w, h), 0)
    cx = w // 2
    cy = h // 2
    per_range = 2 * 3.14 / circle_n
    R = max(w, h)
    for i in range(1, circle_n):
        theta_start = i * per_range
        theta_end = (i + 1) * per_range
        x_start, y_start = cx + R * math.cos(theta_start), cy + R * math.sin(
            theta_start
        )
        x_end, y_end = cx + R * math.cos(theta_end), cy + R * math.sin(theta_end)
        polygon = [(cx, cy), (x_start, y_start), (x_end, y_end)]
        ImageDraw.Draw(img).polygon(polygon, outline=i, fill=i)
    mask = numpy.array(img)
    print("np,raw_mask", mask.sum(), tf.convert_to_tensor(mask).eval())
    return tf.convert_to_tensor(mask)


def get_circle_mask(one_center_point, shape, part_num=10):
    w, h = 212, 212
    raw_mask = get_raw_radia_mask(2 * w, 2 * h, part_num)
    print("raw_mask", raw_mask.eval())
    mask = raw_mask[
        (w - one_center_point[0]) : (2 * w - one_center_point[0]),
        (h - one_center_point[1]) : (2 * h - one_center_point[1]),
    ]
    print("mask", mask.eval())
    return mask

def kernel_dist(kernel_center, bandwith):
    # kernel center
    # (W, H, class_num) --> (W * H * class_num)
    return tf.contrib.distributions.Normal(kernel_center, bandwith)

bandwidth = 1.0
def map_fn_1(x, X_train, X_pred):
    return tf.div(tf.reduce_sum(
            tf.map_fn(lambda x_i: map_fn_2(x_i, x), X_train)),
            tf.multiply(tf.cast(X_pred.shape[0], dtype=tf.float32), bandwidth))

def map_fn_2(x_i, x):
    shape = x_i.shape
    X_for_map = tf.reshape(x_i, [-1])
    return tf.reshape(tf.map_fn(lambda x_j : kernel_dist(x_j, bandwidth).prob(x), X_for_map), shape)

def get_kde_predict(X_pred, X_train):
    return tf.map_fn(lambda x: map_fn_1(x, X_train,X_pred), X_pred)

def Plus_Minus_KDE_dist_test(y_true, y_pred, dilation_filter, part_num=10):
    y_pred = tf.sigmoid(y_pred)
    circle_masks = get_circle_masks(y_pred > 0.5, part_num)
    y_large = tf.nn.dilation2d(
        tf.cast(y_pred > 0.5, tf.float32),
        dilation_filter,
        strides=(1, 1, 1, 1),
        rates=(1, 1, 1, 1),
        padding="SAME",
    )
    y_small = 1 - tf.nn.dilation2d(
        tf.cast(y_pred < 0.5, tf.float32),
        dilation_filter,
        strides=(1, 1, 1, 1),
        rates=(1, 1, 1, 1),
        padding="SAME",
    )
    plus_mask = y_large - tf.cast(y_pred > 0.5, tf.float32)
    minus_mask = tf.cast(y_pred > 0.5, tf.float32) - y_small
    X_pred = tf.linspace(start=0.0, stop=1.0, num=10)
    kde_pred_plus = get_kde_predict(X_pred,tf.boolean_mask(y_pred, plus_mask))
    kde_pred_minus = get_kde_predict(X_pred,tf.boolean_mask(y_pred, minus_mask))

    return tf.reduce_mean(tf.square(kde_pred_plus - kde_pred_minus))

sess = tf.Session()
with sess.as_default():
    dilation_filter = tf.ones((5, 5, 4), tf.float32)
    # loss = Student_Circle_Loss(pred > 0.5, pred, dilation_filter)
    width_size, height_size = 10,10 # 212 x 212
    # y_true = tf.ones((5,width_size,height_size,4))
    # print(sess.run(tf.contrib.distributions.Normal(y_true, y_true)))
    pred = tf.random.uniform((5, width_size,height_size, 4))
    loss = Plus_Minus_KDE_dist_test(pred > 0.5, pred, dilation_filter)
    sess.run(loss)
    tf.print(loss)
    print(loss.eval())

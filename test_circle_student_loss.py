
import tensorflow as tf
from tfwrapper import losses
import numpy
import math
from PIL import Image, ImageDraw

def get_circle_masks(pred_label,part_num):
    circle_masks = []
    for i in range(pred_label.shape[0]):
        # there is some problem with the center point calculation
        multi_class_masks = []
        for j in range(pred_label.shape[-1]):
            one_center_point = tf.reduce_mean(tf.where(pred_label[i,...,j] ),axis =0)
            print(one_center_point.eval())
            one_circle_mask = get_circle_mask(one_center_point,pred_label.shape, part_num)
            print("one_circle_mask",tf.reduce_sum(one_circle_mask).eval())
            multi_class_masks.append(one_circle_mask)
        print("multi_class_masks",tf.stack(multi_class_masks, axis=2).get_shape())
        circle_masks.append(tf.stack(multi_class_masks, axis=2))
    print("circle_masks", tf.stack(circle_masks).get_shape())
    return tf.stack(circle_masks)


def get_raw_radia_mask(w,h,circle_n):
    img = Image.new('L', (w, h), 0)
    cx = w // 2
    cy = h // 2
    per_range = 2* 3.14 / circle_n
    R = max(w,h)
    for i in range(1,circle_n):
        theta_start = i * per_range
        theta_end = (i+1) * per_range
        x_start, y_start =cx+ R * math.cos(theta_start), cy+R *math.sin(theta_start) 
        x_end, y_end =cx+ R * math.cos(theta_end), cy+R *math.sin(theta_end) 
        polygon = [(cx,cy),(x_start,y_start),(x_end,y_end)] 
        ImageDraw.Draw(img).polygon(polygon, outline=i, fill=i)
    mask = numpy.array(img)
    print("np,raw_mask",mask.sum(),tf.convert_to_tensor(mask).eval())
    return tf.convert_to_tensor(mask)

def get_circle_mask(one_center_point, shape, part_num=10):
    w, h = 212 , 212
    raw_mask = get_raw_radia_mask(2*w, 2*h, part_num)
    print("raw_mask", raw_mask.eval())
    mask = raw_mask[(w - one_center_point[0]):(2*w - one_center_point[0]),(h - one_center_point[1]):(2*h - one_center_point[1])]
    print("mask", mask.eval())
    return mask
    # part = tf.zeros(shape)

    # use easy build and then cut( two size highth and width)
    # not api: init once use again and agian
    # 

    # return tf.stack([part for i in range(part_num)])

# TODO(sujinhua): 360 circle_part

def partial_Student_Loss(y_pred, plus_mask, minus_mask):
    n1 = tf.reduce_sum(plus_mask)
    n2 = tf.reduce_sum(minus_mask)
    print("n1", n1.eval())
    print("n2", n2.eval())
    # if tf.equal(n1,tf.zeros(1)) or tf.equal(n2,tf.zeros(1)):
    #     print("good")
    #     return tf.zeros(1)
    y_pred_plus = y_pred * plus_mask
    y_pred_minus = y_pred * minus_mask
    mu1 = tf.reduce_sum(y_pred_plus) / n1
    mu2 = tf.reduce_sum(y_pred_minus) / n2
    s1_square = (y_pred_plus - mu1) ** 2 / n1
    s2_square = (y_pred_minus - mu2) ** 2 / n2
    t =tf.contrib.distributions.StudentT(n1 + n2 - 2, 0.0, 1.0)
    loss_inv = t.cdf(tf.sigmoid(tf.reduce_mean(tf.abs(mu1 - mu2) / tf.sqrt(s1_square + s2_square))))
    return tf.cond(n1*n2 > 0, lambda: 1-loss_inv, lambda: n1*n2)

def Student_Circle_Loss(y_true, y_pred,dilation_filter, part_num=10):
    # norm
    # y_pred = tf.sigmoid(y_pred)
    circle_masks = get_circle_masks(y_pred > 0.5, part_num)
    y_large = tf.nn.dilation2d(tf.cast(y_pred > 0.5 , tf.float32), dilation_filter,strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    y_small = 1 - tf.nn.dilation2d(tf.cast(y_pred < 0.5 , tf.float32), dilation_filter,strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    plus_mask = y_large - y_pred
    minus_mask = y_pred - y_small
    loss_sum = tf.zeros(1)
    # print("circle_masks",circle_masks.eval())
    circle_masks2 = tf.one_hot(circle_masks,depth=part_num) # , depth=part_num
    print("circle_masks2" , circle_masks2.eval())
    for i in range(part_num):
        cmask = circle_masks2[...,i]
        print("cmask", cmask.eval())
        loss_sum+=partial_Student_Loss(y_pred, plus_mask * cmask, minus_mask *cmask)
        print("loss_sum", loss_sum.eval())
    return loss_sum

sess = tf.Session()
with sess.as_default():
    dilation_filter =tf.ones((5,5,4), tf.float32)
    pred = tf.random.uniform((5,212,212,4))
    loss = Student_Circle_Loss(pred> 0.5, pred, dilation_filter)
    sess.run(loss)
    tf.print(loss)
    print(loss.eval())
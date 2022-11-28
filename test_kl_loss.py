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
    # part = tf.zeros(shape)

def get_estimate_mu_sigma(X, mask):
    # X (B, W, H, class_num)
    # mask (B, W, H, class_num)
    n = tf.reduce_sum(mask, [1,2], keep_dims=True)
    # n shape (B,1,1,class_num)
    mu = tf.reduce_sum(X * mask, [1,2], keep_dims=True) / n 
    # mu shape (B,1,1,class_num)  --> (B,1,1,class_num)
    # unsqueeze
    s_square = tf.reduce_sum(((X - mu) * mask) ** 2,  [1,2], keep_dims=True) / n
    # squeeze

    return tf.squeeze(mu,[1,2]), tf.squeeze(s_square,[1,2])


def compute_kl(u1, sigma1, u2, sigma2, dim):
    """
    计算两个多元高斯分布之间KL散度KL(N1||N2)；

    所有的shape均为(B1,B2,...,dim),表示协方差为0的多元高斯分布
    这里我们假设加上Batch_size，即形状为(B,dim)

    dim:特征的维度
    """
    # mu shape (B, dim)
    # sigma shape (B, dim)
    # mu Tensor("truediv_60:0", shape=(), dtype=float32) Tensor("truediv_62:0", shape=(), dtype=float32)
    # sigma Tensor("truediv_61:0", shape=(5, 212, 212, 4), dtype=float32) Tensor("truediv_63:0", shape=(5, 212, 212, 4), dtype=float32)
    print("mu", u1, u2)
    print("sigma", sigma1, sigma2)
    sigma1_matrix = tf.matrix_diag(sigma1)  # (B,dim,dim)
    sigma1_matrix_det = tf.matrix_determinant(sigma1_matrix)  # (B,)

    sigma2_matrix = tf.matrix_diag(sigma2)  # (B,dim,dim)
    sigma2_matrix_det = tf.matrix_determinant(sigma2_matrix)  # (B,)
    sigma2_matrix_inv = tf.matrix_diag(1.0 / sigma2)  # (B,dim,dim)

    delta_u = tf.expand_dims((u2 - u1), axis=-1)  # (B,dim,1)
    print("delta_u",delta_u)
    # bug: tf.matrix_transpose 运行失败 ， ValueError: Argument 'a' should be a (batch) matrix, with rank >= 2.  Found: (1,)
    # delta_u 不是二维以上的矩阵
    delta_u_transpose = tf.matrix_transpose(delta_u)  # (B,1,dim)
    print("delta_u_transpose",delta_u_transpose)

    term1 = tf.reduce_sum((1.0 / sigma2) * sigma1, axis=-1)  # (B,) represent trace term
    print("term1", term1)
    term2 = delta_u_transpose @ sigma2_matrix_inv @ delta_u  # (B,)
    print("term2", term2)
    term3 = -dim
    print("term3", term3)
    # term4 = tf.math.log(sigma2_matrix_det) - tf.math.log(sigma1_matrix_det)
    # print("term4", term4)
    # 

    KL = 0.5 * (term1 + term3 + term2 ) # + term4
    print(KL)

    # if you want to compute the mean of a batch,then,
    KL_mean = tf.reduce_mean(KL)

    return KL_mean

def Plus_Minus_KL_Loss(y_true, y_pred, dilation_filter, part_num=10):
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
    # y_pred shape (B, W, H, class_num)
    # plus_mask shape (B, W, H, class_num)
    mu1, sigma1 = get_estimate_mu_sigma(y_pred, plus_mask)
    mu2, sigma2 = get_estimate_mu_sigma(y_pred, minus_mask)

    return compute_kl(mu1, sigma1, mu2, sigma2, dim=tf.ones((1,), tf.float32))

# case1 自行设计测试用例  输入  预期输出
sess = tf.Session()
with sess.as_default():
    dilation_filter = tf.ones((5, 5, 4), tf.float32)
    # 怎么实现B+ B-  膨胀卷积核
    # loss = Student_Circle_Loss(pred > 0.5, pred, dilation_filter)
    y_true = tf.ones((5,212,212,4))
    pred = tf.random.uniform((5, 212, 212, 4))
    loss = Plus_Minus_KL_Loss(y_true, pred, dilation_filter)
    sess.run(loss)
    tf.print(loss)
    print(loss.eval())

# case2



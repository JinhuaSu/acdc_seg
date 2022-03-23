# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
from tfwrapper import losses
import numpy
import math
from PIL import Image, ImageDraw

# import tensorflow.examples.tutorials.mnist


def inference(images, exp_config, training):
    """
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module.
    """

    return exp_config.model_handle(images, training, nlabels=exp_config.nlabels)


# The input tensor has shape [batch, in_height, in_width, depth] and the filters tensor has shape [filter_height, filter_width, depth], i.e., each input channel is processed independently of the others with its own structuring function. The output tensor has shape [batch, out_height, out_width, depth]. The spatial dimensions of the output tensor depend on the padding algorithm. We currently only support the default "NHWC" data_format.


def Student_Loss(y_true, y_pred, dilation_filter):
    # square(5)
    # y_pred = tf.sigmoid(y_pred)

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
    plus_mask = y_large - y_pred
    minus_mask = y_pred - y_small
    n1 = tf.reduce_sum(plus_mask)
    n2 = tf.reduce_sum(minus_mask)
    y_pred_plus = y_pred * plus_mask
    y_pred_minus = y_pred * minus_mask
    mu1 = tf.reduce_sum(y_pred_plus) / n1
    mu2 = tf.reduce_sum(y_pred_minus) / n2
    s1_square = (y_pred_plus - mu1) ** 2 / n1
    s2_square = (y_pred_minus - mu2) ** 2 / n2
    t = tf.contrib.distributions.StudentT(n1 + n2 - 2, 0.0, 1.0)
    loss_inv = t.cdf(
        tf.sigmoid(tf.reduce_mean(tf.abs(mu1 - mu2) / tf.sqrt(s1_square + s2_square)))
    )
    # to keep the x in the valid range
    # loss_inv = tf.abs(mu1 - mu2) / tf.sqrt(s1_square + s2_square)
    return -loss_inv


def get_circle_masks(pred_label, part_num):
    circle_masks = []
    for i in range(pred_label.shape[0]):
        # there is some problem with the center point calculation
        multi_class_masks = []
        for j in range(pred_label.shape[-1]):
            one_center_point = tf.reduce_mean(tf.where(pred_label[i, ..., j]), axis=0)
            one_circle_mask = get_circle_mask(
                one_center_point, pred_label.shape, part_num
            )
            multi_class_masks.append(one_circle_mask)
        circle_masks.append(tf.stack(multi_class_masks, axis=2))
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
    return tf.convert_to_tensor(mask)


def get_circle_mask(one_center_point, shape, part_num=10):
    w, h = 212, 212
    raw_mask = get_raw_radia_mask(2 * w, 2 * h, part_num)
    mask = raw_mask[
        (w - one_center_point[0]) : (2 * w - one_center_point[0]),
        (h - one_center_point[1]) : (2 * h - one_center_point[1]),
    ]
    return mask


# TODO(sujinhua): 360 circle_part


def partial_Student_Loss(y_pred, plus_mask, minus_mask):
    n1 = tf.reduce_sum(plus_mask)
    n2 = tf.reduce_sum(minus_mask)
    y_pred_plus = y_pred * plus_mask
    y_pred_minus = y_pred * minus_mask
    mu1 = tf.reduce_sum(y_pred_plus) / n1
    mu2 = tf.reduce_sum(y_pred_minus) / n2
    s1_square = (y_pred_plus - mu1) ** 2 / n1
    s2_square = (y_pred_minus - mu2) ** 2 / n2
    t = tf.contrib.distributions.StudentT(n1 + n2 - 2, 0.0, 1.0)
    loss_inv = t.cdf(
        tf.sigmoid(tf.reduce_mean(tf.abs(mu1 - mu2) / tf.sqrt(s1_square + s2_square)))
    )
    return tf.cond(n1 * n2 > 0, lambda: 1 - loss_inv, lambda: n1 * n2)


def Student_Circle_Loss(y_true, y_pred, dilation_filter, part_num=10):
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
    plus_mask = y_large - y_pred
    minus_mask = y_pred - y_small
    loss_sum = tf.zeros(1)
    circle_masks2 = tf.one_hot(circle_masks, depth=part_num)  # , depth=part_num
    for i in range(part_num):
        cmask = circle_masks2[..., i]
        loss_sum += partial_Student_Loss(y_pred, plus_mask * cmask, minus_mask * cmask)
    return loss_sum


# X (5, 212, 212, 1) y_pred (5, 212, 212, 4)
def RAW_Student_Circle_Loss(X, y_pred, dilation_filter, part_num=10):
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
    loss_sum = tf.zeros(1)
    circle_masks2 = tf.one_hot(circle_masks, depth=part_num)  # , depth=part_num
    loss_list = []
    cmask_list = []
    cmask_list2 = []
    for i in range(part_num):
        cmask = circle_masks2[..., i]
        cmask_list += [plus_mask * cmask]
        cmask_list2 += [minus_mask * cmask]
        loss_list += [partial_Student_Loss(X, plus_mask * cmask, minus_mask * cmask)]
    return loss_list, cmask_list, cmask_list2


def get_estimate_mu_sigma(X, mask):
    n = tf.reduce_sum(mask)
    mu = tf.reduce_sum(X) / n
    s_square = (mask - mu) ** 2 / n
    return mu, s_square


def compute_kl(u1, sigma1, u2, sigma2, dim):
    """
    计算两个多元高斯分布之间KL散度KL(N1||N2)；

    所有的shape均为(B1,B2,...,dim),表示协方差为0的多元高斯分布
    这里我们假设加上Batch_size，即形状为(B,dim)

    dim:特征的维度
    """
    sigma1_matrix = tf.matrix_diag(sigma1)  # (B,dim,dim)
    sigma1_matrix_det = tf.matrix_determinant(sigma1_matrix)  # (B,)

    sigma2_matrix = tf.matrix_diag(sigma2)  # (B,dim,dim)
    sigma2_matrix_det = tf.matrix_determinant(sigma2_matrix)  # (B,)
    sigma2_matrix_inv = tf.matrix_diag(1.0 / sigma2)  # (B,dim,dim)

    delta_u = tf.expand_dims((u2 - u1), axis=-1)  # (B,dim,1)
    delta_u_transpose = tf.matrix_transpose(delta_u)  # (B,1,dim)

    term1 = tf.reduce_sum((1.0 / sigma2) * sigma1, axis=-1)  # (B,) represent trace term
    term2 = delta_u_transpose @ sigma2_matrix_inv @ delta_u  # (B,)
    term3 = -dim
    term4 = tf.math.log(sigma2_matrix_det) - tf.math.log(sigma1_matrix_det)

    KL = 0.5 * (term1 + term2 + term3 + term4)

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
    mu1, sigma1 = get_estimate_mu_sigma(y_pred, plus_mask)
    mu2, sigma2 = get_estimate_mu_sigma(y_pred, minus_mask)

    return compute_kl(mu1, sigma1, mu2, sigma2, dim=1)


# TODO(sujinhua): random_part

# y_true (5, 212, 212, 4) y_pred (5, 212, 212, 4)
def Supervised_Student_Circle_Loss(y_true, y_pred, dilation_filter, part_num=10):
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
    y_large_true = tf.nn.dilation2d(
        tf.cast(y_true, tf.float32),
        dilation_filter,
        strides=(1, 1, 1, 1),
        rates=(1, 1, 1, 1),
        padding="SAME",
    )
    y_small_true = 1 - tf.nn.dilation2d(
        tf.cast(y_true, tf.float32),
        dilation_filter,
        strides=(1, 1, 1, 1),
        rates=(1, 1, 1, 1),
        padding="SAME",
    )
    plus_mask = y_large - tf.cast(y_pred > 0.5, tf.float32)
    minus_mask = tf.cast(y_pred > 0.5, tf.float32) - y_small
    plus_mask_true = y_large_true - y_true
    minus_mask_true = y_true - y_small_true
    loss_sum = tf.zeros(1)
    circle_masks2 = tf.one_hot(circle_masks, depth=part_num)  # , depth=part_num
    for i in range(part_num):
        cmask = circle_masks2[..., i]
        loss_sum += partial_Student_Loss(
            y_pred,
            plus_mask * plus_mask_true * cmask,
            minus_mask * minus_mask_true * cmask,
        )
    return loss_sum


def loss(
    logits,
    labels,
    nlabels,
    loss_type,
    weight_decay=0.0,
    warm_up_done=True,
    loss_k=1000000,
):
    """
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'weighted_crossentropy'/'crossentropy'/'dice'/'dice_onlyfg'/'crossentropy_and_dice'
    :param weight_decay: The weight for the L2 regularisation of the network paramters
    :return: The total loss including weight decay, the loss without weight decay, only the weight decay
    """
    with tf.variable_scope("weights_norm"):
        labels = tf.one_hot(labels, depth=nlabels)
        dilation_filter = tf.ones((5, 5, 4), tf.float32)

        weights_norm = tf.reduce_sum(
            input_tensor=weight_decay
            * tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection("weight_variables")]
            ),
            name="weights_norm",
        )

    if loss_type == "weighted_crossentropy":
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(
            logits, labels, class_weights=[0.1, 0.3, 0.3, 0.3]
        )
    elif loss_type == "crossentropy":
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
    elif loss_type == "dice":
        segmentation_loss = losses.dice_loss(logits, labels, only_foreground=False)
    elif loss_type == "dice_onlyfg":
        segmentation_loss = losses.dice_loss(logits, labels, only_foreground=True)
    elif loss_type == "crossentropy_and_dice":
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(
            logits, labels
        ) + 0.2 * losses.dice_loss(logits, labels)
    else:
        raise ValueError("Unknown loss: %s" % loss_type)

    # ac_loss = losses.Active_Contour_Loss(logits, labels)
    # student_loss = Student_Loss(labels, logits,dilation_filter)
    print(logits.shape, labels.shape)
    student_loss2 = Supervised_Student_Circle_Loss(labels, logits, dilation_filter)
    # kl_loss = Plus_Minus_KL_Loss(labels,logits,dilation_filter)
    # ac_loss += student_loss
    # + student_loss

    total_loss = tf.add(segmentation_loss, weights_norm)  # + ac_loss / 10
    if warm_up_done:
        # total_loss = kl_loss + total_loss
        total_loss = student_loss2 / loss_k + total_loss

    return total_loss, segmentation_loss, weights_norm


def predict(images, exp_config):
    """
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    """

    logits = exp_config.model_handle(
        images, training=tf.constant(False, dtype=tf.bool), nlabels=exp_config.nlabels
    )
    softmax = tf.nn.softmax(logits)
    mask = tf.arg_max(softmax, dimension=-1)

    return mask, softmax


def training_step(loss, optimizer_handle, learning_rate, **kwargs):
    """
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :param optimizer_handle: A handle to one of the tf optimisers
    :param learning_rate: Learning rate
    :param momentum: Optionally, you can also pass a momentum term to the optimiser.
    :return: The training operation
    """

    if "momentum" in kwargs:
        momentum = kwargs.get("momentum")
        optimizer = optimizer_handle(learning_rate=learning_rate, momentum=momentum)
    else:
        optimizer = optimizer_handle(learning_rate=learning_rate)

    # The with statement is needed to make sure the tf contrib version of batch norm properly performs its updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    return train_op


def evaluation(logits, labels, images, nlabels, loss_type, warm_up_done):
    """
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the
    current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    """

    mask = tf.arg_max(tf.nn.softmax(logits, dim=-1), dimension=-1)  # was 3
    mask_gt = labels

    tf.summary.image(
        "example_gt", prepare_tensor_for_summary(mask_gt, mode="mask", nlabels=nlabels)
    )
    tf.summary.image(
        "example_pred", prepare_tensor_for_summary(mask, mode="mask", nlabels=nlabels)
    )
    tf.summary.image("example_zimg", prepare_tensor_for_summary(images, mode="image"))

    total_loss, nowd_loss, weights_norm = loss(
        logits, labels, nlabels=nlabels, loss_type=loss_type, warm_up_done=warm_up_done
    )

    cdice_structures = losses.per_structure_dice(
        logits, tf.one_hot(labels, depth=nlabels)
    )
    cdice_foreground = cdice_structures[:, 1:]

    cdice = tf.reduce_mean(cdice_foreground)

    return nowd_loss, cdice


def prepare_tensor_for_summary(img, mode, idx=0, nlabels=None):
    """
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard.
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label..
    :return: Tensor ready to be used with tf.summary.image(...)
    """

    if mode == "mask":

        if img.get_shape().ndims == 3:
            V = img[idx, ...]
        elif img.get_shape().ndims == 4:
            V = img[idx, ..., 10]
        elif img.get_shape().ndims == 5:
            V = img[idx, ..., 10, 0]
        else:
            raise ValueError(
                "Dont know how to deal with input dimension %d"
                % (img.get_shape().ndims)
            )

    elif mode == "image":

        if img.get_shape().ndims == 3:
            V = img[idx, ...]
        elif img.get_shape().ndims == 4:
            V = img[idx, ..., 0]
        elif img.get_shape().ndims == 5:
            V = img[idx, ..., 10, 0]
        else:
            raise ValueError(
                "Dont know how to deal with input dimension %d"
                % (img.get_shape().ndims)
            )

    else:
        raise ValueError("Unknown mode: %s. Must be image or mask" % mode)

    if mode == "image" or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= nlabels - 1  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]

    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

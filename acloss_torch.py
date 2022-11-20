import torch


def active_contour_loss(y, y_pred):
    """
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    """
    shp_x = y_pred.shape
    shp_y = y.shape

    if len(shp_x) != len(shp_y):
        y = y.view((shp_y[0], 1, *shp_y[1:]))

    if all([i == j for i, j in zip(y_pred.shape, y.shape)]):
        # if this is the case then gt is probably already a one hot encoding
        y_onehot = y
    else:
        gt = y.long()
        y_onehot = torch.zeros(shp_x)
        if y_pred.device.type == "cuda":
            y_onehot = y_onehot.cuda(y_pred.device.index)
        y_onehot.scatter_(1, gt, 1)

    # length term
    delta_r = (
        y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    )  # horizontal gradient (B, C, H-1, W)
    delta_c = (
        y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    )  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    length = torch.mean(
        torch.sqrt(delta_pred + epsilon)
    )  # eq.(11) in the paper, mean is used instead of sum.
    # region term
    c_in = torch.ones_like(y_pred)
    c_out = torch.zeros_like(y_pred)

    # one hot这里有问题
    region_in = torch.abs(
        torch.mean(y_pred * (y_onehot - c_in) ** 2)
    )  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.abs(torch.mean((1 - y_pred) * (y_onehot - c_out) ** 2))
    # region_in = torch.abs(torch.mean(y_pred* ((y_onehot[:, 0, :, :] - c_in[:, 0, :, :]) ** 2)))  # equ.(12) in the paper
    # region_out = torch.abs(torch.mean((1 - y_pred[:, 0, :, :]) * ((y_onehot[:, 0, :, :] - c_out[:, 0, :, :]) ** 2)))  # equ.(12) in the paper

    region = region_in + region_out

    loss = length + region

    return loss

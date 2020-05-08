# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn import functional as F
from fvcore.nn import focal_loss,giou_loss,smooth_l1_loss,\
    sigmoid_focal_loss_star,sigmoid_focal_loss_jit,\
    sigmoid_focal_loss_star_jit,sigmoid_focal_loss
# from math import pi,atan
# __all__=[""]

# -----------------------------------------------
def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:

    p = torch.softmax(inputs,-1)
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")

    targets = F.one_hot(targets, inputs.size(-1)).to(targets.device)

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss.unsqueeze(-1)*targets * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


softmax_focal_loss_jit = torch.jit.script(
    softmax_focal_loss
)  # type: torch.jit.ScriptModule


def softmax_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:

    targets = F.one_hot(targets, inputs.size(-1)).to(targets.device)
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -F.log_softmax(shifted_inputs,-1) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()  # pyre-ignore
    elif reduction == "sum":
        loss = loss.sum()  # pyre-ignore

    return loss


softmax_focal_loss_star_jit = torch.jit.script(
    softmax_focal_loss_star
)  # type: torch.jit.ScriptModule



giou_loss_jit = torch.jit.script(
    giou_loss
)  # type: torch.jit.ScriptModule



smooth_l1_loss_jit = torch.jit.script(
    smooth_l1_loss
)  # type: torch.jit.ScriptModule


def ciou_loss(
    boxes1: torch.Tensor,# [x1,y1,x2,y2]
    boxes2: torch.Tensor,# # [x1,y1,x2,y2]
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # 对角线长度
    diag_len= (xc2-xc1)**2+(yc2-yc1)**2

    # 中心点距离平方
    x0 = (x1+x2)/2
    y0 = (y1+y2)/2
    x0g = (x1g+x2g)/2
    y0g = (y1g+y2g)/2
    center_len = (x0-x0g)**2+(y0-y0g)**2

    #
    w = x2-x1
    h = y2-y1
    wg = x2g-x1g
    hg = y2g-y1g
    pi = -4*torch.atan(torch.tensor(-1.,device=boxes1.device))
    v = 4/pi**2*(torch.atan(wg/hg)-torch.atan(w/h))**2
    alpha = v/(1-iouk+v)

    loss = 1 - iouk+center_len/diag_len+alpha*v

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

ciou_loss_jit = torch.jit.script(
    ciou_loss
)  # type: torch.jit.ScriptModule


def diou_loss(
    boxes1: torch.Tensor,# [x1,y1,x2,y2]
    boxes2: torch.Tensor,# # [x1,y1,x2,y2]
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    Gradient-friendly IoU loss with an additional penalty that is non-zero when the
    boxes do not overlap and scales with the size of their smallest enclosing box.
    This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    # 对角线长度
    diag_len= (xc2-xc1)**2+(yc2-yc1)**2

    # 中心点距离平方
    x0 = (x1+x2)/2
    y0 = (y1+y2)/2
    x0g = (x1g+x2g)/2
    y0g = (y1g+y2g)/2
    center_len = (x0-x0g)**2+(y0-y0g)**2

    loss = 1 - iouk+center_len/diag_len

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

diou_loss_jit = torch.jit.script(
    diou_loss
)  # type: torch.jit.ScriptModule

if __name__=="__main__":
    # pred = torch.rand([5,])
    # y = torch.randint(0,2,[5,],dtype=torch.float32)

    # loss = sigmoid_focal_loss(pred,y,reduction ="mean")
    # loss = sigmoid_focal_loss_star(pred,y,reduction ="mean")
    # loss = sigmoid_focal_loss_jit(pred,y,reduction ="mean")
    # loss = sigmoid_focal_loss_star_jit(pred,y,reduction ="mean")

    # pred = torch.rand([5,10])
    # y = torch.randint(0, 10, [5, ], dtype=torch.long)

    # loss = softmax_focal_loss(pred,y,reduction ="mean")
    # loss = softmax_focal_loss_jit(pred,y,reduction ="mean")
    # loss = softmax_focal_loss_star(pred,y,reduction ="mean")
    # loss = softmax_focal_loss_star_jit(pred,y,reduction ="mean")


    # box1 = torch.as_tensor([[10,20,30,40],[120,180,189,200]])
    # box2 = torch.as_tensor([[15,23,30,46],[116,175,194,207]])
    # loss = giou_loss(box1,box2,reduction="mean")

    pred = torch.rand([5, 10])
    y = torch.rand([5, 10])
    loss = smooth_l1_loss_jit(pred,y,1e-3,reduction="mean")

    print(loss)
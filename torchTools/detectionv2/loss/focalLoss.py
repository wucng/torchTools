# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn import functional as F
from fvcore.nn import focal_loss,giou_loss,smooth_l1_loss,\
    sigmoid_focal_loss_star,sigmoid_focal_loss_jit,\
    sigmoid_focal_loss_star_jit,sigmoid_focal_loss

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
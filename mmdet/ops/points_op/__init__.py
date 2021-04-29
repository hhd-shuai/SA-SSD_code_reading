from .points_ops import *
from mmdet.ops.points_op import points_op_cpu
import torch

def pts_in_boxes3d(pts, boxes3d):
    N = len(pts)        # 18426
    M = len(boxes3d)    # 15
    pts_in_flag = torch.IntTensor(M, N).fill_(0)        # torch.Size([15, 18426])
    reg_target = torch.FloatTensor(N, 3).fill_(0)       # torch.Size([18426, 3])
    points_op_cpu.pts_in_boxes3d(pts.contiguous(), boxes3d.contiguous(), pts_in_flag, reg_target)
    return pts_in_flag, reg_target


from .points_ops import *
from mmdet.ops.points_op import points_op_cpu
import torch

# 在 Auxiliary Network 中， 需要分割出 前景点 和 背景点，首先需要生成前景点和背景点的 label
def pts_in_boxes3d(pts, boxes3d):
    N = len(pts)        # 17839
    M = len(boxes3d)    # 15
    pts_in_flag = torch.IntTensor(M, N).fill_(0)        # [15, 17839]
    reg_target = torch.FloatTensor(N, 3).fill_(0)       # [17839, 3]
    points_op_cpu.pts_in_boxes3d(pts.contiguous(), boxes3d.contiguous(), pts_in_flag, reg_target)
    return pts_in_flag, reg_target


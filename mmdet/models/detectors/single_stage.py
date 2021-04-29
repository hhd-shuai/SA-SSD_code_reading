import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, rbbox2roi, bbox2result, multi_apply, kitti_bbox2results,\
                        tensor2points, delta2rbbox3d, weighted_binary_cross_entropy)
import torch.nn.functional as F


# BaseDetector是所有检测器的基类，是虚基类
# RPNTestMixin, BBoxTestMixin,MaskTestMixin ？？
class SingleStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    # 单阶段目标检测又Backbone，Neck，Bbox_head，Extra_head组成
    # 它们的实现需要设计者自己设计
    def __init__(self,
                 backbone, #{'type': 'SimpleVoxel', 'num_input_features': 4, 'use_norm': True, 'num_filters': [32, 64], 'with_distance': False}
                 neck=None, # {'type': 'SpMiddleFHD', 'output_shape': [40, 1600, 1408], 'num_input_features': 4, 'num_hidden_features': 320}
                 bbox_head=None, # {'type': 'SSDRotateHead', 'num_class': 1, 'num_output_filters': 256, 'num_anchor_per_loc': 2, 'use_sigmoid_cls': True, 'encode_rad_error_by_sin': True, 'use_direction_classifier': True, 'box_code_size': 7}
                 extra_head=None, # {'type': 'PSWarpHead', 'grid_offsets': (0.0, 40.0), 'featmap_stride': 0.4, 'in_channels': 256, 'num_class': 1, 'num_parts': 28}
                 train_cfg=None, # {'rpn': {'assigner': {'Car': {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_pos_iou': 0.45}, 'ignore_iof_thr': -1, 'similarity_fn': 'NearestIouSimilarity'}, 'anchor_thr': 0.1}, 'extra': {'assigner': {'pos_iou_thr': 0.7, 'neg_iou_thr': 0.7, 'min_pos_iou': 0.7, 'ignore_iof_thr': -1, 'similarity_fn': 'RotateIou3dSimilarity'}}}
                 test_cfg=None, # {'rpn': {'nms_across_levels': False, 'nms_pre': 2000, 'nms_post': 100, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'extra': {'score_thr': 0.3, 'nms': {'type': 'nms', 'iou_thr': 0.1}, 'max_per_img': 100}}
                 pretrained=None): # None
        super(SingleStageDetector, self).__init__()
        # 初始化backbone
        # backbone 通常通过全链接网络来提取特征映射图
        self.backbone = builder.build_backbone(backbone)

        # 初始化neck
        # neck 连接骨干和头的部分，例如：FPN、ASPP
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        # 初始化head
        # 用于特定任务。例如：候选框的预测（记做bbox_head）、掩膜的预测
        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

        # 初始化extra-head
        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)

        # 传入cfg中的参数
        # 加载训练参数和测试参数
        self.train_cfg = train_cfg # {'rpn': {'assigner': {'Car': {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_pos_iou': 0.45}, 'ignore_iof_thr': -1, 'similarity_fn': 'NearestIouSimilarity'}, 'anchor_thr': 0.1}, 'extra': {'assigner': {'pos_iou_thr': 0.7, 'neg_iou_thr': 0.7, 'min_pos_iou': 0.7, 'ignore_iof_thr': -1, 'similarity_fn': 'RotateIou3dSimilarity'}}}
        self.test_cfg = test_cfg # {'rpn': {'nms_across_levels': False, 'nms_pre': 2000, 'nms_post': 100, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'extra': {'score_thr': 0.3, 'nms': {'type': 'nms', 'iou_thr': 0.1}, 'max_per_img': 100}}

        #
        # 加载上次训练的模型
        self.init_weights(pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):       # 处理多个batch
        ret = {}
        for key, elems in batch_args.items():
            if key in ['voxels', 'num_points', ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key in ['coordinates', ]:
                coors = []
                for i, coor in enumerate(elems): # coor.shape : torch.Size([19480, 3])
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in ['img_meta', 'gt_labels', 'gt_bboxes', 'gt_types', ]:
                ret[key] = elems
            else:
                if isinstance(elems, dict):
                    ret[key] = {k: torch.stack(v, dim=0) for k, v in elems.items()}
                else:
                    ret[key] = torch.stack(elems, dim=0)
        return ret

    # img.shape [B, 3, 384, 1248]
    # img_meta: list
    #          img_meta[0]: dict
    #                      img_shape : tuple (375, 1242, 3)
    #                      sample_idx: 7251
    #                      calib
    # kwargs:
    #       1. anchors           list: len(anchors)      = B
    #       2. voxels            list: len(voxels)       = B
    #       3. coordinates       list: len(coordinates)  = B
    #       4. num_points        list: len(num_points)   = B
    #       5. anchor_mask       list: len(anchor_mask)  = B
    #       6. gt_labels         list: len(gt_labels)    = B
    #       7. gt_bboxes         list: len(gt_bboxes)    = B

    # img, img_meta, **kwargs都属于train_dataset push的数据
    def forward_train(self, img, img_meta, **kwargs):
        # img [1, 3, 384, 1248]
        batch_size = len(img_meta) # batch_size: 2
        # step1: 处理多batch情况
        ret = self.merge_second_batch(kwargs) # dict 'anchors'={dict:1}, 'voxels'={Tensor:34496}, 'coordinates'={Tensor:34496}, 'num_points'={Tensor:34496}, 'anchors_mask'={dict:1}, 'gt_labels'={list:2}, 'gt_bboxes'={list:2}, 'gt_types'={list:2}
        # step2: 
        vx = self.backbone(ret['voxels'], ret['num_points']) # torch.Size([34496, 4])
        # step3:
        # x.shape     = [2, 256, 200, 176]
        # conv6.shape = [2, 256, 200, 176]
        # point_misc  : tuple, shape = 3
        #             : 1. point_mean : shape [N,4] , [:,0] 是 Batch number
        #             : 2. point_cls  : shape [N,1]
        #             : 3. point_reg  : shape [N.3]
        x, conv6, point_misc = self.neck(vx, ret['coordinates'], batch_size, is_test=False)

        losses = dict()

        # point_misc--|
        #             |+ ---> neck.aux_loss ---> aux_loss
        # gt_bboxes---|
        aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
        losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:

            # rpn_outs    : tuple, size = 3
            #             : 1. box_preds      : shape [N, 200, 176, 14]
            #             : 2. cls_preds      : shape [N, 200, 176,  2]
            #             : 3. dir_cls_preds  : shape [N, 200, 176,  4]
            rpn_outs = self.rpn_head(x)
            # rpn_outs    : tuple, shape = 8
            rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['gt_types'],\
                            ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            # rpn_outs----|
            #             |
            # gt_bboxes---|
            #             |
            # gt_labels---|
            #             |+ ---> rpn_loss_inputs ---> rpn_head.loss ---> rpn_losses
            # anchors_mask|
            #             |
            # gt_types----|
            #             |
            # anchors-----|
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            # guided_anchors.shape :
            #                        [num_of_guided_anchors, 7]
            #                      + [num_of_gt_bboxes,      7]
            #                      ----------------------------
            #                      = [all_num,               7]

            guided_anchors, _ = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'],\
                        ret['anchors_mask'], ret['gt_bboxes'], ret['gt_labels'], thr=self.train_cfg.rpn.anchor_thr)
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            bbox_score = self.extra_head(conv6, guided_anchors)
            refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
            # bbox_score----|
            #               |
            # guided_anchors|
            #               |+ ---> refine_loss_inputs ---> extra_head.loss ---> refine_losses
            # gt_bboxes-----|
            #               |
            # gt_labels-----|
            refine_losses = self.extra_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)
        # 处理多个batch_size
        ret = self.merge_second_batch(kwargs)                   # ret['voxels'] [15470, 5, 4] 15470个体素，每个体素最多5个点 
                                                                # ret['num_points'] [15470] 记录每个体素包含的点云个数                                                
        vx = self.backbone(ret['voxels'], ret['num_points'])    # vx [15470, 4] 每个体素的特征  三个坐标和一个反射，体素内所有点云的平均值
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)    # ret['coordinates'] [15470, 4]

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors, anchor_labels = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                                       None, None, thr=.1)

        bbox_score = self.extra_head(conv6, guided_anchors, is_test=True)

        det_bboxes, det_scores, det_labels = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, anchor_labels, img_meta, self.test_cfg.extra)

        results = [kitti_bbox2results(*param, class_names=self.class_names) for param in zip(det_bboxes, det_scores, det_labels, img_meta)]

        return results




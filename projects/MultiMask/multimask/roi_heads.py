# -*- coding: utf-8 -*-
# Copyright (c) TechMagic K.K. and its affiliates. All Rights Reserved

import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY

@ROI_HEADS_REGISTRY.register()
class MultiMaskROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of MultiMask head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_multimask_head(cfg, input_shape)

    def _init_multimask_head(self, cfg, input_shape):
        # fmt: off
        self.multimask_on = cfg.MODEL.MULTIMASK_ON
        if not self.multimask_on:
            return
        # get the names of mask head
        self.head_names = cfg.MODEL.ROI_HEADS.MASK_HEADS
        # build mask heads
        # self.mask_heads = build_multimask_head(cfg.MODEL.ROI_MULTIMASK_HEADS, input_shape)

    def forward(self, images, features, proposals, targets=None):
        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_multimask(features, instances))
        else:
            instances = self._forward_multimask(features, instances)
        return instances, losses
    
    def _forward_multimask(self, features, instances):
        """
        Forward logic of the multimask prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            if len(proposals_dp) > 0:
                # NOTE may deadlock in DDP if certain workers have empty proposals_dp
                proposal_boxes = [x.proposal_boxes for x in proposals_dp]

                features_dp = self.densepose_pooler(features, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _ = self.densepose_predictor(densepose_head_outputs)
                densepose_loss_dict = self.densepose_losses(proposals_dp, densepose_outputs)
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _ = self.densepose_predictor(densepose_head_outputs)
            else:
                # If no detection occurred instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 4)

            densepose_inference(densepose_outputs, instances)
            return instances


def build_multimask_head(dict_list, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MULTIMASK_HEADS`.
    """
    return

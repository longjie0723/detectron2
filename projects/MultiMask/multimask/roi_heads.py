# -*- coding: utf-8 -*-
# Copyright (c) TechMagic K.K. and its affiliates. All Rights Reserved

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .multimask_head import build_multimask_head

from detectron2.modeling.roi_heads import select_foreground_proposals, StandardROIHeads
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler

logger = logging.getLogger(__name__)

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
        self.mask_poolers = {}
        self.mask_heads = {}
        for name in self.head_names:
            pooler_resolution = cfg.MODEL[name].POOLER_RESOLUTION
            pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
            sampling_ratio    = cfg.MODEL[name].POOLER_SAMPLING_RATIO
            pooler_type       = cfg.MODEL[name].POOLER_TYPE
            # fmt: on

            in_channels = [input_shape[f].channels for f in self.in_features][0]

            self.mask_poolers[name] = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            # build mask heads
            self.mask_heads[name] = build_multimask_head(
                cfg, name, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
            ).to(device=cfg.MODEL.DEVICE)
            # Aliasing to get along with nn.module.children()
            setattr(self, "mask_head_{}".format(name.lower()),self.mask_heads[name])

    def forward(self, images, features, proposals, targets=None):
        instances, losses = super().forward(images, features, proposals, targets)
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_multimask(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
        return pred_instances, {}
    
    def forward_with_given_boxes(self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_multimask(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return self.box_predictor.losses(predictions, proposals)
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_multimask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.multimask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            loss = {}
            for name in self.head_names:
                mask_features = self.mask_poolers[name](features, proposal_boxes)
                loss.update(self.mask_heads[name](mask_features, proposals))
            return loss
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            for name in self.head_names:
                mask_features = self.mask_poolers[name](features, pred_boxes)
                instances = self.mask_heads[name](mask_features, instances)
            return instances

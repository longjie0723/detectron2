# -*- coding = utf-8 -*-
# Copyright (c) TechMagic K.K. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_multimask_config(cfg):
    """
    Add config for multimask head.
    """
    _C = cfg
    _C.MODEL.__dict__[CN.NEW_ALLOWED] = True
    _C.NEW_ALLOWED = True
    _C.NEW_ALLOWED = True
    _C.MODEL.MULTIMASK_ON = True
    _C.MODEL.ROI_HEADS.MASK_HEADS = []

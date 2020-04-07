# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import numpy as np

from detectron2.structures import (PolygonMasks)


class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.multimask_on   = cfg.MODEL.MULTIMASK_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            self._transform_multimask(
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape)

        # Add secondary mask
        if len(annos) and "secondary_segmentation" in annos[0]:
            segms = [obj["secondary_segmentation"] for obj in annos]
            instances.gt_secondary_masks = PolygonMasks(segms)

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def _transform_multimask(self, annotation, transforms):
        if not self.multimask_on:
            return annotation

        # Handle multimask annotation (copy from segmentation)
        if "secondary_segmentation" in annotation:
            # each instance contains 1 or more polygons
            segm = annotation["secondary_segmentation"]
            if isinstance(segm, list):
                # polygons
                polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                annotation["secondary_segmentation"] = [
                    p.reshape(-1) for p in transforms.apply_polygons(polygons)
                ]
            else:
                raise ValueError(
                    "Cannot transform secondary_segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )
        return annotation

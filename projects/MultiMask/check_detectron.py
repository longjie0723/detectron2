import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from torchviz import make_dot, make_dot_from_trace

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import detectron2.export

im = cv2.imread("./input.jpg")

cfg = get_cfg()    # obtain detectron2's default config
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

detectron2.export.export_onnx_model(cfg, predictor, im)

#make_dot(outputs['instances'].model)

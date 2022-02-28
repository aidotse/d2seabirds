# Some basic setup:
import torch, torchvision

#print(torch.cuda.is_available())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops, find_contours

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pycocotools
from PIL import Image, ImageDraw
import numpy as np

# import ipdb
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data.datasets import register_coco_instances

import sys

# coco_path_ann = "/home/juan.vallado/data/annotations_SLU.json"
# img_path = "/home/juan.vallado/data/sequences_sampled/"
# coco_path_ann = "/home/joakim.eriksson/data/annotations_SLU.json"
# img_path = "/home/joakim.eriksson/data/testDataset/"
coco_path_ann = "/home/erik/data/annotations_SLU.json"
img_path = "/home/erik/data/images"

# Prepare dataloader for transformations
register_coco_instances("seabirds_train", {}, coco_path_ann, img_path)
seabirds_metadata = MetadataCatalog.get("seabirds_train")
datset = DatasetCatalog.get("seabirds_train")
dataloader = detectron2.data.build_detection_train_loader(
    dataset=datset,
    mapper=DatasetMapper(
        cfg=get_cfg(), is_train=True, augmentations=[T.RandomContrast(-1, 1)]
    ),
    total_batch_size=128,
)

def visualize(n):
    dataset_dicts = DatasetCatalog.get("seabirds_train")
    imgs = []
    for d in random.sample(dataset_dicts, n):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=ytvis_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        imgs.append(out.get_image()[:, :, ::-1])

    for i in range(0, len(imgs)):
        cv2.imwrite("{}v.jpg".format(i), imgs[i])


# visualize(5)

# RUN
cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("seabirds_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 10
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 8000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
cfg.INPUT.MASK_FORMAT = "bitmask"

# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
# cfg.OUTPUT_DIR = "/home/appuser/output"
# cfg.OUTPUT_DIR = "/home/joakim.eriksson/output"
cfg.OUTPUT_DIR = "/home/erik/data"

"""
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
"""
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth"
)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)

#sys.exit()

from detectron2.utils.visualizer import ColorMode

dataset_dicts = datset
imgs = []
filenames = []
res = []
# folder = "/home/juan.vallado/data/sequences_sampled/"
# folder = "/home/joakim.eriksson/data/testDataset/"
folder = "/home/erik/data/images"

ims = os.listdir(folder)

# ipdb.set_trace()
 

# for img in random.sample(ims, 100):
#for img in random.sample(ims, 100):
for img in ims:

    file = os.path.join(folder, img)
    filename = os.path.basename(file)
    print(filename)
    im = np.array(Image.open(file).convert("RGB"))
    im = im[:, :, ::-1]  # to bgr
    outputs = predictor(im)

    resi = dict()
    resi["pred_classes"] = outputs["instances"].pred_classes.cpu().detach().numpy()
    resi["scores"] = outputs["instances"].scores.cpu().detach().numpy()
    resi["pred_boxes"] = outputs["instances"].pred_boxes.tensor.cpu().detach().numpy()
    res.append(resi)

    v = Visualizer(
        im[:, :, ::-1],
        metadata=seabirds_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imgs.append(out.get_image()[:, :, ::-1])
    filenames.append(filename)


import pickle
from pathlib import Path

output_path = Path("/home/erik/output")

for i in range(0, len(imgs)):
    print(imgs[i].shape)
    # cv2.imwrite(filename.format(i), imgs[i])
    # filenamepkl = filename.replace("jpg", "pkl")
    # f = open(filenamepkl.format(i), "wb")
    cv2.imwrite(str(output_path.joinpath(filenames[i].format(i))), imgs[i])
    filenamepkl = filenames[i].replace("jpg", "pkl")
    f = open(output_path.joinpath(filenamepkl.format(i)), "wb")
    # cv2.imwrite("../erik/output/{}sb.jpg".format(i), imgs[i])
    # f = open("../erik.svensson/output/{}sb.pkl".format(i), "wb")
    pickle.dump(res[i], f)
    f.close()


"""
ino='/home/juan.vallado/data/YoutubeVIS/train/Annotations/0a8c467cc3/'
for image in os.listdir(ino):
    path = os.path.join(ino, image)
    print("With PIL:{}\nWith cv(Anydepth):{}\nWith cv GDAL:{}" \
    .format(
        len(np.unique(np.array(Image.open(path)))),
        len(np.unique(cv2.imread(path, cv2.IMREAD_ANYDEPTH))),
        len(np.unique(cv2.imread(path, cv2.IMREAD_LOAD_GDAL)))
    ))
"""

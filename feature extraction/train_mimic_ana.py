import torch
import detectron2
import json
from tqdm import tqdm
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import pickle

bboxlist = ['right lung', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone',
            'right hilar structures', 'right apical zone', 'right costophrenic angle',
            'right hemidiaphragm', 'left lung', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone',
            'left hilar structures', 'left apical zone', 'left costophrenic angle', 'left hemidiaphragm',
            'spine', 'aortic arch', 'mediastinum', 'upper mediastinum',
            'cardiac silhouette', 'left cardiac silhouette', 'right cardiac silhouette', 'cavoatrial junction'
]

diseaselist = ['lung opacity','pleural effusion','atelectasis','lobar/segmental collapse','enlarged cardiac silhouette','airspace opacity',
                'consolidation','tortuous aorta','calcified nodule','lung lesion','mass/nodule (not otherwise specified)',
               'pulmonary edema/hazy opacity','costophrenic angle blunting','vascular congestion','vascular calcification',
               'linear/patchy atelectasis','elevated hemidiaphragm','pleural/parenchymal scarring','hydropneumothorax','pneumothorax',
               'enlarged hilum','multiple masses/nodules','hyperaeration','mediastinal widening','vascular redistribution',
               'mediastinal displacement','spinal degenerative changes','superior mediastinal mass/enlargement','scoliosis',
               'sub-diaphragmatic air','hernia','spinal fracture','bone lesion','increased reticular markings/ild pattern',
               'infiltration','pneumomediastinum','bronchiectasis','cyst/bullae']

bbox_dict = {}
for i, bbox_class in enumerate(bboxlist):
    bbox_dict[bbox_class] = i

disease_dict = {}
for i, disease_class in enumerate(diseaselist):
    disease_dict[disease_class] = i

def get_mimic_ana_dicts(split): # ordered csv. chest-imagenome is required. https://physionet.org/content/chest-imagenome/1.0.0/
    split_dir = os.path.join('/mnt/data/wy/dataset/silver_dataset/splits', split+'.csv')
    ann_dir = '/mnt/data/wy/dataset/silver_dataset/annotation'
    with open(split_dir, 'r') as f:
        df = pd.read_csv(f)
        json_files = [image_id+'.json' for image_id in df['dicom_id']]

    data_dicts = []
    category_set = bbox_dict

    for idx in tqdm(range(len(json_files))):
        json_file = json_files[idx]
    # for idx, json_file in enumerate(json_files):
        record = {}
        objs = []
        path = os.path.join(ann_dir, json_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
            record['file_name'] = '/mnt/data/wy/dataset/mimic_cxr_png/' + data['image_id'] + '.png'
            record['image_id'] = idx
            record['height'] = 1024
            record['width'] = 1024
            for object in data['annotations']:
                if object['bbox_name'] not in category_set:
                    continue

                bbox = object['bbox_224']
                x1 = bbox[0] / float(224/1024)
                y1 = bbox[1] / float(224/1024)
                x2 = bbox[2] / float(224/1024)
                y2 = bbox[3] / float(224/1024)
                obj = {
                    'bbox': [x1,y1,x2,y2],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    "category_id": category_set[object['bbox_name']]
                }
                objs.append(obj)
            record['annotations'] = objs
        data_dicts.append(record)
    return data_dicts

for d in ["train", "valid"]:
    DatasetCatalog.register("mimic_dataset_" + d, lambda d=d: get_mimic_ana_dicts(d))
    MetadataCatalog.get("mimic_dataset_" + d).set(thing_classes=bboxlist)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("mimic_dataset_train")
cfg.DATASETS.TEST = ("mimic_dataset_valid",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.0002  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(bboxlist)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.WEIGHTS = str("results/mimic_ana_bbx/model_final_pre.pth")
# cfg.TEST.EVAL_PERIOD = 1000

cfg.OUTPUT_DIR = 'results/mimic_ana_bbx'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# predictor = DefaultPredictor(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# evalute model
val_loader = build_detection_test_loader(cfg, "mimic_dataset_valid")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
cfg.MODEL.WEIGHTS = os.path.join('results/mimic_ana_bbx', "model_final.pth")
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("mimic_dataset_valid", cfg, False, output_dir="output")
inference_on_dataset(predictor.model, val_loader, evaluator)